import cv2
import logging
import numpy as np
from typing import Optional, Tuple
from mtcnn import MTCNN

logger = logging.getLogger(__name__)

class DocumentDetector:
    """
    Handles Layout Parsing, Document Corner Detection, and Portrait Extraction.
    Combines MTCNN for faces and YOLOv8 for layout/corners (Mocked ultralytics integration for prototype).
    """
    def __init__(self):
        logger.info("Initializing DocumentDetector (MTCNN & YOLOv8 integration)...")
        self.face_detector = MTCNN(device="CPU:0")
        
        # YOLOv8 model placeholder
        # self.yolo_model = YOLO('models/yolov8_corner_detector.pt')

    def extract_portrait(self, image: np.ndarray, output_path: str) -> Optional[str]:
        """
        Detects the citizen's portrait using MTCNN and saves the cropped image.
        Pads the bounding box slightly to avoid harsh cropping.
        """
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_detector.detect_faces(img_rgb)
        
        if not results:
            logger.warning("No face detected in document.")
            return None
            
        # Select largest face
        largest_face = max(results, key=lambda b: b['box'][2] * b['box'][3])
        x, y, w, h = largest_face['box']
        
        # Add 20% padding around the face
        padding = int(0.2 * max(w, h))
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image.shape[1] - x, w + 2 * padding)
        h = min(image.shape[0] - y, h + 2 * padding)
        
        face_crop = image[y:y+h, x:x+w]
        
        cv2.imwrite(output_path, face_crop)
        logger.info(f"Portrait extracted and saved to {output_path}")
        return output_path

    def detect_corners(self, image: np.ndarray) -> np.ndarray:
        """
        Mock for YOLOv8 Quadrilateral Corner Detection.
        Returns coordinates of 4 corners: [top-left, top-right, bottom-right, bottom-left]
        """
        # In production:
        # results = self.yolo_model(image)
        # return parse_corners(results)
        
        h, w = image.shape[:2]
        return np.array([
            [0, 0],
            [w, 0],
            [w, h],
            [0, h]
        ], dtype="float32")
