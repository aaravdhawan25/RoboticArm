# robot_find_and_pick_final_io_with_helpers.py
# RAW SERVO FIND MODE, non-blocking calibration

import cv2
import numpy as np
from pathlib import Path
import urllib.request
import serial
import time
import math
import sys
import threading
import queue
import os

# ---------------- USER SETTINGS ----------------
PORT = "/dev/cu.usbmodem1101"   # <<-- change to your Arduino port
BAUD = 115200

FRAME_W = 640
FRAME_H = 480

# Arm geometry (inches)
L1 = 6.5
L2 = 5.0
L3 = 6.5

# Camera offsets (relative to claw tip)
CAMERA_LEFT_RIGHT_OFFSET_IN = 0.0
CAMERA_BEHIND_CLAW_IN = 5.5
CAMERA_ABOVE_CLAW_IN = 2.5
CAMERA_TILT_DEG = 25.0

REAL_OBJ_WIDTH_IN = 2.7  # tennis ball
FOCAL_LENGTH = None
FOCAL_FILE = "focal_length.npy"

PICKUP_APPROACH_HEIGHT = 0.6
LIFT_AFTER_GRAB = 2.0

CLAW_OPEN_ANGLE = 90
CLAW_CLOSE_ANGLE = 30

# *** FIND MODE RAW SERVO ANGLES (YOUR VALUES) ***
FIND_SHOULDER_SERVO = 110
FIND_ELBOW_SERVO    = 40
FIND_WRIST_SERVO    = 24
FIND_CLAW_SERVO     = CLAW_OPEN_ANGLE
FIND_BASE_CENTER    = 90  # base straight ahead

# Base sweep while in find mode
BASE_SWEEP_MIN   = 40
BASE_SWEEP_MAX   = 140
BASE_SWEEP_STEP  = 3
BASE_SWEEP_DELAY = 0.06

# ---------------- Serial connect ----------------
print("Opening serial port:", PORT)
try:
    arduino = serial.Serial(PORT, BAUD, timeout=1)
    time.sleep(2)
    print("Serial OK")
except Exception as e:
    print("Could not open serial port:", e)
    sys.exit(1)

if os.path.exists(FOCAL_FILE):
    try:
        FOCAL_LENGTH = float(np.load(FOCAL_FILE))
        print(f"Loaded saved focal length: {FOCAL_LENGTH:.2f}")
    except Exception:
        FOCAL_LENGTH = None

# ---------------- ObjectDetectionPipeline (verbatim) ----------------
class ObjectDetectionPipeline:
    def __init__(self, confidence_threshold=0.5, nms_threshold=0.4):
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self._load_models()
    def _download_file(self, url, filepath):
        print(f"Downloading {filepath}...")
        urllib.request.urlretrieve(url, filepath)
    def _load_models(self):
        model_dir = Path("models")
        model_dir.mkdir(exist_ok=True)
        weights_path = model_dir / "yolov3-tiny.weights"
        config_path = model_dir / "yolov3-tiny.cfg"
        names_path = model_dir / "coco.names"
        weights_url = "https://pjreddie.com/media/files/yolov3-tiny.weights"
        config_url = "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg"
        names_url = "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"
        if not weights_path.exists():
            self._download_file(weights_url, str(weights_path))
        if not config_path.exists():
            self._download_file(config_url, str(config_path))
        if not names_path.exists():
            self._download_file(names_url, str(names_path))
        self.net = cv2.dnn.readNetFromDarknet(str(config_path), str(weights_path))
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        with open(names_path) as f:
            self.classes = [line.strip() for line in f.readlines()]
        self.colors = np.random.randint(0, 255, size=(len(self.classes), 3))
        print(f"Model loaded successfully ({len(self.classes)} classes)")
    def detect(self, image):
        height, width = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1/255.0, (416,416), swapRB=True, crop=False)
        self.net.setInput(blob)
        layer_names = self.net.getLayerNames()
        output_layers = [layer_names[i-1] for i in self.net.getUnconnectedOutLayers()]
        outputs = self.net.forward(output_layers)
        boxes, confidences, class_ids = [], [], []
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = int(np.argmax(scores))
                confidence = float(scores[class_id])
                if confidence > self.confidence_threshold:
                    cx, cy, w, h = (detection[0:4] * np.array([width,height,width,height])).astype(int)
                    x = cx - w//2
                    y = cy - h//2
                    boxes.append([x,y,w,h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        indices = cv2.dnn.NMSBoxes(
            boxes, confidences, self.confidence_threshold, self.nms_threshold
        )
        detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                x,y,w,h = boxes[i]
                cx = x + w//2
                cy = y + h//2
                detections.append({
                    'box': (x,y,w,h),
                    'conf': confidences[i],
                    'class': self.classes[class_ids[i]],
                    'id': class_ids[i],
                    'cx': cx,
                    'cy': cy
                })
        return detections
    def draw_boxes(self, image, detections):
        result = image.copy()
        for det in detections:
            x,y,w,h = det['box']
            class_id = det['id']
            color = tuple(map(int, self.colors[class_id]))
            cv2.rectangle(result,(x,y),(x+w,y+h),color,2)
            label = f"{det['class']}: {det['conf']:.2f}"
            cv2.putText(result,label,(x,y-5),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)
        return result

# ---------------- Distance estimation ----------------
def calibrate_focal_from_pixel_width(pixel_w, known_distance_in):
    global FOCAL_LENGTH
    FOCAL_LENGTH = (pixel_w * known_distance_in) / REAL_OBJ_WIDTH_IN
    np.save(FOCAL_FILE, np.array([FOCAL_LENGTH]))
    print(f"[calibrate] FOCAL_LENGTH = {FOCAL_LENGTH:.2f} px (saved)")

def estimate_distance_from_pixels(pixel_w):
    if FOCAL_LENGTH is None:
        raise RuntimeError(
            "FOCAL_LENGTH unset. Run 'cal <inches>' or press 'c' to see pixel width."
        )
    return (REAL_OBJ_WIDTH_IN * FOCAL_LENGTH) / max(1.0, pixel_w)

# ---------------- Camera ray -> robot coords ----------------
def pixels_to_camera_ray(cx, cy, pixel_w):
    slant_dist = estimate_distance_from_pixels(pixel_w)
    fx = FOCAL_LENGTH
    fy = FOCAL_LENGTH
    dx = cx - (FRAME_W/2)
    dy = (FRAME_H/2) - cy
    angle_x = math.atan2(dx, fx)
    angle_y = math.atan2(dy, fy)
    return slant_dist, angle_x, angle_y

def camera_ray_to_robot_coords(slant, ax, ay):
    forward_cam = slant * math.cos(ay)
    lateral_cam = slant * math.sin(ax)
    vertical_cam_down = slant * math.sin(ay)

    tilt_rad = math.radians(CAMERA_TILT_DEG)
    forward_rot = forward_cam*math.cos(tilt_rad) - vertical_cam_down*math.sin(tilt_rad)
    vertical_rot = forward_cam*math.sin(tilt_rad) + vertical_cam_down*math.cos(tilt_rad)

    obj_forward_from_claw = forward_rot - CAMERA_BEHIND_CLAW_IN
    obj_lateral_from_claw = lateral_cam + CAMERA_LEFT_RIGHT_OFFSET_IN
    obj_z = CAMERA_ABOVE_CLAW_IN - vertical_rot

    X_robot = max(obj_forward_from_claw, 0.25)
    Y_robot = obj_lateral_from_claw
    Z_robot = max(obj_z, 0.0)

    return X_robot, Y_robot, Z_robot

# ---------------- IK ----------------
def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def solve_ik(x,y,z):
    base_rad = math.atan2(y,x)
    base_deg = math.degrees(base_rad)

    planar = math.sqrt(x*x + y*y)
    reach = planar - L3
    D = math.sqrt(reach*reach + z*z)

    if D < 1e-6:
        raise ValueError("Target degenerate")

    cos_el = (L1*L1 + L2*L2 - D*D) / (2*L1*L2)
    cos_el = clamp(cos_el, -1.0, 1.0)
    el_internal = math.acos(cos_el)
    elbow_deg = math.degrees(math.pi - el_internal)

    alpha = math.atan2(z, reach)
    cos_beta = (L1*L1 + D*D - L2*L2) / (2*L1*D)
    cos_beta = clamp(cos_beta, -1.0, 1.0)
    beta = math.acos(cos_beta)
    shoulder_deg = math.degrees(alpha + beta)

    wrist_deg = 90.0 - (shoulder_deg - elbow_deg)

    if base_deg < 0:
        base_deg += 360

    return base_deg, shoulder_deg, elbow_deg, wrist_deg

# ---------------- Mapping IK -> servos ----------------
STRAIGHT_UP_SHOULDER = 90.0
STRAIGHT_UP_ELBOW = 90.0
STRAIGHT_UP_WRIST = 93.0
HOME_BASE = 90
HOME_SHOULDER = STRAIGHT_UP_SHOULDER
HOME_ELBOW = STRAIGHT_UP_ELBOW
HOME_WRIST = STRAIGHT_UP_WRIST
HOME_CLAW = 90
BASE_DIR = -1
SHOULDER_DIR = +1
ELBOW_DIR = 1
WRIST_DIR = 1
CLAW_DIR = +1

def map_ik_to_servo(joint, ik_angle):
    if joint == "base":
        home, d = HOME_BASE, BASE_DIR
    elif joint == "shoulder":
        home, d = HOME_SHOULDER, SHOULDER_DIR
    elif joint == "elbow":
        home, d = HOME_ELBOW, ELBOW_DIR
    elif joint == "wrist":
        home, d = HOME_WRIST, WRIST_DIR
    elif joint == "claw":
        home, d = HOME_CLAW, CLAW_DIR
    else:
        return int(clamp(round(ik_angle), 0, 180))
    servo = home + d * (ik_angle - home)
    return int(clamp(round(servo), 0, 180))

def send_mapped_csv(base_ik, shoulder_ik, elbow_ik, wrist_ik, claw_ang):
    sb = map_ik_to_servo("base", base_ik)
    ss = map_ik_to_servo("shoulder", shoulder_ik)
    se = map_ik_to_servo("elbow", elbow_ik)
    sw = map_ik_to_servo("wrist", wrist_ik)
    sc = map_ik_to_servo("claw", claw_ang)
    msg = f"{sb},{ss},{se},{sw},{sc}\n"
    arduino.write(msg.encode())
    time.sleep(0.03)

def send_all(base, shoulder, elbow, wrist, claw):
    send_mapped_csv(base, shoulder, elbow, wrist, claw)

def move_to_xyz(x,y,z,claw_angle=CLAW_OPEN_ANGLE):
    b,s,e,w = solve_ik(x,y,z)
    if b > 180:
        b = b - 360 if (b - 360) >= 0 else 180
    send_all(b,s,e,w,claw_angle)

# ---------- RAW send helper (no IK mapping) ----------
def send_raw_csv_rawvals(base_servo, shoulder_servo, elbow_servo, wrist_servo, claw_servo):
    def _clamp(a):
        return int(max(0, min(180, int(round(a)))))
    sb = _clamp(base_servo)
    ss = _clamp(shoulder_servo)
    se = _clamp(elbow_servo)
    sw = _clamp(wrist_servo)
    sc = _clamp(claw_servo)
    msg = f"{sb},{ss},{se},{sw},{sc}\n"
    try:
        arduino.write(msg.encode())
    except Exception as e:
        print("[serial] write failed:", e)
    time.sleep(0.03)

def send_raw_csv(base_servo, shoulder_servo, elbow_servo, wrist_servo, claw_servo):
    send_raw_csv_rawvals(base_servo, shoulder_servo, elbow_servo, wrist_servo, claw_servo)

# ---------- FIND MODE (RAW, Option A) ----------
_find_mode = False
_find_thread = None

def set_find_posture_raw():
    """Move arm once into the find posture."""
    send_raw_csv(FIND_BASE_CENTER,
                 FIND_SHOULDER_SERVO,
                 FIND_ELBOW_SERVO,
                 FIND_WRIST_SERVO,
                 FIND_CLAW_SERVO)
    time.sleep(0.25)

def _find_sweep_raw(min_servo=BASE_SWEEP_MIN,
                    max_servo=BASE_SWEEP_MAX,
                    step=BASE_SWEEP_STEP,
                    delay=BASE_SWEEP_DELAY):
    """
    Sweep ONLY the base servo left/right.
    Shoulder/Elbow/Wrist stay in the fixed find posture.
    """
    global _find_mode
    print("[find] raw sweep starting (base only).")
    base_angle = min_servo
    direction = 1
    # posture set ONCE
    set_find_posture_raw()
    while _find_mode:
        send_raw_csv(base_angle,
                     FIND_SHOULDER_SERVO,
                     FIND_ELBOW_SERVO,
                     FIND_WRIST_SERVO,
                     FIND_CLAW_SERVO)
        time.sleep(delay)
        base_angle += direction * step
        if base_angle >= max_servo:
            base_angle = max_servo
            direction = -1
        elif base_angle <= min_servo:
            base_angle = min_servo
            direction = 1
    print("[find] raw sweep stopped.")

def start_find_sweep_raw(min_a=BASE_SWEEP_MIN, max_a=BASE_SWEEP_MAX):
    global _find_mode, _find_thread
    if _find_mode:
        print("Find already running")
        return
    _find_mode = True
    _find_thread = threading.Thread(target=_find_sweep_raw,
                                    args=(min_a, max_a),
                                    daemon=True)
    _find_thread.start()
    print("[python] start_find_sweep_raw: started.")

def stop_find_sweep_raw():
    global _find_mode
    if not _find_mode:
        print("Find not running")
        return
    _find_mode = False
    time.sleep(BASE_SWEEP_DELAY*2)
    print("[python] stop_find_sweep_raw: stopped.")

# ---------- Public command wrappers ----------
def start_find_cmd():
    """Start find mode using raw servo angles."""
    start_find_sweep_raw()
    try:
        arduino.write(b"find\n")
    except:
        pass
    print("[python] start_find_cmd: raw find sweep started and 'find' sent to Arduino")

def stop_find_cmd():
    stop_find_sweep_raw()
    try:
        arduino.write(b"stop\n")
    except:
        pass
    print("[python] stop_find_cmd: raw find sweep stopped and 'stop' sent to Arduino")

def open_claw_cmd():
    """Open the claw."""
    try:
        arduino.write(b"CLAW_OPEN\n")
        time.sleep(0.05)
    except:
        pass
    send_raw_csv_rawvals(HOME_BASE,
                         FIND_SHOULDER_SERVO,
                         FIND_ELBOW_SERVO,
                         FIND_WRIST_SERVO,
                         CLAW_OPEN_ANGLE)
    print("[python] open_claw_cmd: sent CLAW_OPEN + raw fallback")

def close_claw_cmd():
    """Close the claw."""
    try:
        arduino.write(b"CLAW_CLOSE\n")
        time.sleep(0.05)
    except:
        pass
    send_raw_csv_rawvals(HOME_BASE,
                         FIND_SHOULDER_SERVO,
                         FIND_ELBOW_SERVO,
                         FIND_WRIST_SERVO,
                         CLAW_CLOSE_ANGLE)
    print("[python] close_claw_cmd: sent CLAW_CLOSE + raw fallback")

# ---------------- Console thread (non-blocking) ----------------
cmd_queue = queue.Queue()

def console_reader():
    while True:
        try:
            line = input()
        except EOFError:
            break
        cmd_queue.put(line.strip())

console_thread = threading.Thread(target=console_reader, daemon=True)
console_thread.start()

# ---------------- main loop ----------------
pipeline = ObjectDetectionPipeline(confidence_threshold=0.40)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

print("Commands: cal <distance_in_inches> | find | stop | open | close | quit")
print("In camera window: 'f' toggle find, 'c' show pixel width for calibration, 'q' quit.")
print("Calibration: put ball in view, press 'c' to see width, then type: cal <distance> in terminal.")

try:
    while True:
        # process console commands (non-blocking)
        while not cmd_queue.empty():
            line = cmd_queue.get()
            if not line:
                continue
            if line.startswith("cal"):
                parts = line.split()
                if len(parts) != 2:
                    print("Usage: cal <distance_in_inches>")
                    continue
                try:
                    known_dist = float(parts[1])
                except ValueError:
                    print("Invalid distance. Use a number, e.g. cal 8")
                    continue

                ret, frame = cap.read()
                if not ret:
                    print("Camera read failed for calibration.")
                    continue

                dets = pipeline.detect(frame)
                if len(dets) == 0:
                    print("No detection for calibration — make sure ball is visible with a green box.")
                else:
                    best = max(dets, key=lambda d: d['box'][2]*d['box'][3])
                    bx,by,bw,bh = best['box']
                    print(f"Calibration pixel width (px) = {bw}")
                    calibrate_focal_from_pixel_width(bw, known_dist)

            elif line == "find":
                start_find_cmd()
            elif line == "stop":
                stop_find_cmd()
            elif line == "open":
                open_claw_cmd()
            elif line == "close":
                close_claw_cmd()
            elif line == "quit":
                stop_find_sweep_raw()
                raise KeyboardInterrupt()
            else:
                # manual X Y Z move
                try:
                    parts = line.split()
                    if len(parts) >= 3:
                        x,y,z = map(float, parts[:3])
                        print("Manual move to:", x,y,z)
                        move_to_xyz(x,y,z, CLAW_OPEN_ANGLE)
                except Exception as ex:
                    print("Unknown command:", ex)

        # camera + detection
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        detections = pipeline.detect(frame)

        # draw
        disp = frame.copy()
        for det in detections:
            x,y,w,h = det['box']
            cx = det['cx']; cy = det['cy']
            cv2.rectangle(disp,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.circle(disp,(cx,cy),3,(0,0,255),-1)
            cv2.putText(disp,
                        f"{det['class']} {det['conf']:.2f}",
                        (x,y-6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255,200,0),
                        2)

        cv2.imshow("camera", disp)
        k = cv2.waitKey(1) & 0xFF

        if k == ord('q'):
            break
        if k == ord('f'):
            if not _find_mode:
                start_find_cmd()
            else:
                stop_find_cmd()
        if k == ord('c'):
            if len(detections) == 0:
                print("No detection for calibration - place tennis ball in view and try again.")
            else:
                best = max(detections, key=lambda d: d['box'][2]*d['box'][3])
                bx,by,bw,bh = best['box']
                print(f"[c] Current calibration pixel width (px) = {bw}")
                print("Now in terminal, type: cal <distance_in_inches>  e.g.  cal 8")

        # If in find mode and detection appears -> stop sweep and pick
        if _find_mode and len(detections) > 0:
            print("Found object during sweep; stopping sweep and picking.")
            stop_find_sweep_raw()
            time.sleep(BASE_SWEEP_DELAY*2)

        # If not sweeping and have detections -> pick largest
        if len(detections) > 0 and not _find_mode:
            best = max(detections, key=lambda d: d['box'][2]*d['box'][3])
            bx,by,bw,bh = best['box']
            cx = best['cx']; cy = best['cy']
            try:
                slant_dist, ang_x, ang_y = pixels_to_camera_ray(cx, cy, bw)
            except Exception as ex:
                print("Distance estimation failed:", ex)
                continue
            Xr, Yr, Zr = camera_ray_to_robot_coords(slant_dist, ang_x, ang_y)
            print(f"Pick target {best['class']} -> slant {slant_dist:.2f}in, X={Xr:.2f}in Y={Yr:.2f}in Z={Zr:.2f}in")
            if Xr < 0.5:
                print("Target too close; skipping.")
            else:
                try:
                    # approach above
                    move_to_xyz(Xr, Yr, Zr + PICKUP_APPROACH_HEIGHT, CLAW_OPEN_ANGLE)
                    time.sleep(0.6)
                    # drop down
                    move_to_xyz(Xr, Yr, Zr, CLAW_OPEN_ANGLE)
                    time.sleep(0.35)
                    # close claw
                    close_claw_cmd()
                    time.sleep(0.5)
                    # lift
                    move_to_xyz(Xr, Yr, Zr + LIFT_AFTER_GRAB, CLAW_CLOSE_ANGLE)
                    time.sleep(0.6)
                except Exception as ex:
                    print("Pickup failed:", ex)
                finally:
                    time.sleep(1.0)

finally:
    cap.release()
    cv2.destroyAllWindows()
    try:
        arduino.close()
    except:
        pass
    print("Exiting.")
