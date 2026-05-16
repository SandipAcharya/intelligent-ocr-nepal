import cv2
from mtcnn import MTCNN
import os

class FaceDetector:
    def __init__(self):
        self.detector = MTCNN()

    def extract_face(self, image, output_path):
        """
        Detects a face in the document and saves it to output_path.
        Returns the path to the saved face image, or None if no face is found.
        """
        # Convert BGR to RGB for MTCNN
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.detector.detect_faces(img_rgb)
        
        if not results:
            return None
        
        # Take the largest face found
        bounding_box = results[0]['box']
        x, y, w, h = bounding_box
        
        # Add padding
        padding = int(0.2 * max(w, h))
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image.shape[1] - x, w + 2 * padding)
        h = min(image.shape[0] - y, h + 2 * padding)
        
        face_crop = image[y:y+h, x:x+w]
        
        cv2.imwrite(output_path, face_crop)
        return output_path

def detect_document_corners(image_path):
    """
    Mock layout analysis/document corner detection.
    In a full implementation, YOLOv8 would predict the 4 corners of the citizenship/ID card.
    """
    pass
