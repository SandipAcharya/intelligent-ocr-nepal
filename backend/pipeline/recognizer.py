import easyocr

class DevanagariRecognizer:
    def __init__(self):
        # Using EasyOCR as a robust hybrid OCR engine for Nepali/Hindi
        # In a full deployment, PaddleOCR or TrOCR could be used for specific fields.
        self.reader = easyocr.Reader(['ne', 'en'], gpu=False)

    def extract_text(self, image_path):
        """
        Extracts raw text strings from the image.
        Returns a list of extracted texts and their confidences.
        """
        results = self.reader.readtext(image_path)
        
        extracted_data = []
        for (bbox, text, prob) in results:
            extracted_data.append({
                "text": text,
                "confidence": prob,
                "bbox": bbox
            })
            
        return extracted_data

    def recognize(self, image_path):
        """
        Returns a single concatenated string of extracted text.
        """
        extracted = self.extract_text(image_path)
        return " ".join([item['text'] for item in extracted])
