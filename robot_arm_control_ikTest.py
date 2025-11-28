import math
import serial
import time

# -------------------------------
# USER SETTINGS
# -------------------------------

# Serial port your Arduino is on:
# Examples:
#   Windows: "COM3"
#   Mac: "/dev/tty.usbmodem1101"
PORT = "/dev/cu.usbmodem1101"  # <<< CHANGE THIS
BAUD = 115200

# Arm segment lengths (change to match your real robot)
L1 = 5  # Shoulder → elbow
L2 = 6.5  # Elbow → wrist
L3 = 7.5   # Wrist → claw tip

# Claw presets
CLAW_OPEN   = 30
CLAW_CLOSED = 120

# Joint limits
LIMITS = {
    "base": (0, 180),
    "shoulder": (10, 170),
    "elbow": (0, 180),
    "wrist": (0, 180),
    "claw": (0, 180)
}

# -------------------------------
# CONNECT TO ARDUINO
# -------------------------------
print("Connecting to Arduino...")
arduino = serial.Serial(PORT, BAUD, timeout=1)
time.sleep(2)
print("Connected!\n")


# -------------------------------
# SEND ANGLES TO ARDUINO
# -------------------------------
def clamp(value, min_val, max_val):
    return max(min(value, max_val), min_val)


def send_servo_angles(base, shoulder, elbow, wrist, claw):
    # Apply joint limits
    base     = clamp(base,     *LIMITS["base"])
    shoulder = clamp(shoulder, *LIMITS["shoulder"])
    elbow    = clamp(elbow,    *LIMITS["elbow"])
    wrist    = clamp(wrist,    *LIMITS["wrist"])
    claw     = clamp(claw,     *LIMITS["claw"])

    msg = f"{base},{shoulder},{elbow},{wrist},{claw}\n"
    arduino.write(msg.encode())
    time.sleep(0.02)  # Smooth movement


# -------------------------------
# INVERSE KINEMATICS SOLVER
# -------------------------------
def solve_ik(x, y, z):
    """
    Returns: base, shoulder, elbow, wrist
    """

    # Base rotation
    base = math.degrees(math.atan2(y, x))

    # Distance on ground plane
    planar = math.sqrt(x*x + y*y)

    # Account for wrist (subtract L3)
    reach = planar - L3
    height = z

    # Distance from shoulder
    dist = math.sqrt(reach*reach + height*height)

    # Law of cosines
    cos_elbow = (L1*L1 + L2*L2 - dist*dist) / (2*L1*L2)
    cos_elbow = clamp(cos_elbow, -1, 1)
    elbow_angle = 180 - math.degrees(math.acos(cos_elbow))

    # Shoulder angle
    angle_a = math.degrees(math.atan2(height, reach))
    angle_b = math.degrees(math.acos(
        clamp((L1*L1 + dist*dist - L2*L2) / (2*L1*dist), -1, 1)
    ))
    shoulder_angle = angle_a + angle_b

    # Wrist tries to keep end effector level
    wrist_angle = 90 - (shoulder_angle - elbow_angle)

    return base, shoulder_angle, elbow_angle, wrist_angle


# -------------------------------
# MOVE TO A TARGET POINT
# -------------------------------
def move_to_xyz(x, y, z, claw_angle=CLAW_OPEN):
    base, shoulder, elbow, wrist = solve_ik(x, y, z)
    print(f"IK → base={base:.1f}, sh={shoulder:.1f}, el={elbow:.1f}, wr={wrist:.1f}")

    send_servo_angles(base, shoulder, elbow, wrist, claw_angle)


# -------------------------------
# CLAW CONTROL
# -------------------------------
def open_claw():
    print("Opening claw...")
    send_servo_angles(0, 90, 90, 90, 90)


def close_claw():
    print("Closing claw...")
    send_servo_angles(0, 90, 90, 30, 0)


# -------------------------------
# MAIN LOOP
# -------------------------------
print("READY!\n")
print("Type coordinates like: 10 5 8")
print("Or type: open, close, exit\n")

while True:
    user = input("> ")

    if user == "exit":
        break

    if user == "open":
        open_claw()
        continue

    if user == "close":
        close_claw()
        continue

    try:
        x, y, z = map(float, user.split())
        move_to_xyz(x, y, z)
    except:
        print("Invalid input. Type X Y Z or 'open' or 'close'.")
