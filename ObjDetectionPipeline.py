import cv2
import numpy as np
from pathlib import Path
import urllib.request


class ObjectDetectionPipeline:
    def __init__(self, confidence_threshold=0.5, nms_threshold=0.4):
        """
        Initialize object detection using OpenCV's built-in models
        Uses YOLOv3-tiny downloaded automatically from OpenCV sources

        Args:
            confidence_threshold: Detection confidence threshold (0-1)
            nms_threshold: Non-maximum suppression threshold
        """
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold

        # Download and load pre-trained models
        self._load_models()

    def _download_file(self, url, filepath):
        """Download file from URL"""
        print(f"Downloading {filepath}...")
        try:
            urllib.request.urlretrieve(url, filepath)
        except Exception as e:
            print(f"Error downloading {url}: {e}")
            raise

    def _load_models(self):
        """Download and load YOLOv3-tiny model files"""
        model_dir = Path("models")
        model_dir.mkdir(exist_ok=True)

        weights_path = model_dir / "yolov3-tiny.weights"
        config_path = model_dir / "yolov3-tiny.cfg"
        names_path = model_dir / "coco.names"

        # URLs for model files
        weights_url = "https://pjreddie.com/media/files/yolov3-tiny.weights"
        config_url = "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg"
        names_url = "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"

        # Download if not present
        if not weights_path.exists():
            self._download_file(weights_url, str(weights_path))

        if not config_path.exists():
            self._download_file(config_url, str(config_path))

        if not names_path.exists():
            self._download_file(names_url, str(names_path))

        # Load network
        self.net = cv2.dnn.readNetFromDarknet(str(config_path), str(weights_path))
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        # Load class names
        with open(names_path) as f:
            self.classes = [line.strip() for line in f.readlines()]

        self.colors = np.random.randint(0, 255, size=(len(self.classes), 3))
        print(f"Model loaded successfully ({len(self.classes)} classes)")

    def detect(self, image):
        """
        Run detection on image

        Args:
            image: BGR image from cv2.imread()

        Returns:
            List of detections with box, confidence, and class info
        """
        height, width = image.shape[:2]

        # Create blob
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)

        # Get predictions
        layer_names = self.net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        outputs = self.net.forward(output_layers)

        boxes, confidences, class_ids = [], [], []

        # Process detections
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > self.confidence_threshold:
                    cx, cy, w, h = (detection[0:4] * np.array([width, height, width, height])).astype(int)
                    x = cx - w // 2
                    y = cy - h // 2

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Apply NMS
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.nms_threshold)

        detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                detections.append({
                    'box': (x, y, w, h),
                    'conf': confidences[i],
                    'class': self.classes[class_ids[i]],
                    'id': class_ids[i]
                })

        return detections

    def draw_boxes(self, image, detections):
        """Draw detection boxes on image"""
        result = image.copy()

        for det in detections:
            x, y, w, h = det['box']
            class_id = det['id']
            color = tuple(map(int, self.colors[class_id]))

            cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
            label = f"{det['class']}: {det['conf']:.2f}"
            cv2.putText(result, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return result

    def process_image(self, image_path, output_path=None):
        """
        Process single image

        Args:
            image_path: Path to input image
            output_path: Optional path to save annotated image

        Returns:
            Tuple of (annotated_image, detections_list)
        """
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        detections = self.detect(image)
        result = self.draw_boxes(image, detections)

        if output_path:
            cv2.imwrite(str(output_path), result)
            print(f"Saved to {output_path}")

        return result, detections

    def process_video(self, video_path, output_path=None):
        """
        Process video file

        Args:
            video_path: Path to input video
            output_path: Optional path to save annotated video
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        frame_count = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                detections = self.detect(frame)
                result = self.draw_boxes(frame, detections)

                if writer:
                    writer.write(result)

                frame_count += 1
                if frame_count % 30 == 0:
                    print(f"Processed {frame_count} frames")

        finally:
            cap.release()
            if writer:
                writer.release()

        print(f"Complete. Processed {frame_count} frames")

    def process_webcam(self, display=True):
        """
        Real-time detection from webcam

        Args:
            display: Show live feed with detections
        """
        # Try different camera indices for Mac compatibility
        cap = None
        for i in range(10):
            cap = cv2.VideoCapture(1)
            if cap.isOpened():
                print(f"Using camera index {i}")
                break

        if cap is None or not cap.isOpened():
            raise ValueError("Could not open webcam")

        # Set camera properties for Mac
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        print("Webcam stream active. Press 'q' to quit.")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to read frame, retrying...")
                    continue

                detections = self.detect(frame)
                result = self.draw_boxes(frame, detections)

                if display:
                    cv2.imshow("Object Detection", result)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()


# Example usage
if __name__ == "__main__":
    # Initialize pipeline (downloads models on first run)
    pipeline = ObjectDetectionPipeline(confidence_threshold=0.4)

    # Real-time webcam
    pipeline.process_webcam()