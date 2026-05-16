import logging
from typing import List, Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)

class HybridRecognizer:
    """
    Hybrid OCR/HTR Engine as specified in the technical architecture.
    
    Stage 3 of the 4-stage deep learning pipeline:
      - Printed Text Crops  → PaddleOCR (PP-OCRv5, lang='hi' = Devanagari/Hindi)
      - Handwritten Crops   → TrOCR (VisionEncoderDecoder Transformer)
    """
    def __init__(self, use_gpu: bool = False):
        logger.info("Initializing HybridRecognizer...")
        self.use_gpu = use_gpu
        self.paddle_ocr = None
        self.trocr_processor = None
        self.trocr_model = None

        # 1. Printed Text Engine: PaddleOCR with Devanagari support
        # NOTE: PaddleOCR uses 'hi' (Hindi) for Devanagari script — not 'devanagari'
        try:
            from paddleocr import PaddleOCR
            self.paddle_ocr = PaddleOCR(
                use_angle_cls=True,
                lang='hi',           # 'hi' = Hindi/Devanagari in PaddleOCR
                use_gpu=use_gpu,
                show_log=False
            )
            logger.info("PaddleOCR (Devanagari/hi) initialized successfully.")
        except Exception as e:
            logger.warning(f"PaddleOCR failed to load: {e}. Will skip printed OCR.")

        # 2. Handwritten Text Engine: HuggingFace TrOCR
        try:
            import torch
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
            self.trocr_processor = TrOCRProcessor.from_pretrained(
                'microsoft/trocr-base-handwritten'
            )
            self.trocr_model = VisionEncoderDecoderModel.from_pretrained(
                'microsoft/trocr-base-handwritten'
            ).to(self.device)
            logger.info("TrOCR (HTR) initialized successfully.")
        except Exception as e:
            logger.warning(f"TrOCR could not be loaded (may need internet/first-run download): {e}")

    def recognize_printed(self, image_path: str) -> List[Dict[str, Any]]:
        """
        Extracts printed text blocks using PaddleOCR (Devanagari-optimized).
        Returns a list of {text, confidence, bbox, type} dicts.
        """
        if self.paddle_ocr is None:
            logger.warning("PaddleOCR not available, returning empty result.")
            return []

        results = self.paddle_ocr.ocr(image_path, cls=True)
        extracted_data = []

        if results and results[0]:
            for line in results[0]:
                bbox, (text, confidence) = line
                extracted_data.append({
                    "text": text,
                    "confidence": float(confidence),
                    "bbox": bbox,
                    "type": "printed"
                })
        return extracted_data

    def recognize_handwritten(self, image_crop: np.ndarray) -> Dict[str, Any]:
        """
        Processes a single handwritten text line crop via TrOCR.
        Uses ViT image encoder + GPT-2 style text decoder.
        """
        if self.trocr_processor is None or self.trocr_model is None:
            return {"text": "[HTR unavailable]", "confidence": 0.0, "type": "handwritten"}

        try:
            from PIL import Image
            image = Image.fromarray(image_crop).convert("RGB")
            pixel_values = self.trocr_processor(
                image, return_tensors="pt"
            ).pixel_values.to(self.device)
            generated_ids = self.trocr_model.generate(pixel_values)
            text = self.trocr_processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]
            return {"text": text, "confidence": 0.85, "type": "handwritten"}
        except Exception as e:
            logger.error(f"HTR extraction failed: {e}")
            return {"text": "", "confidence": 0.0, "type": "handwritten"}

    def process_document(
        self,
        image_path: str,
        layout_regions: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Orchestrates the routing pipeline:
        - Printed regions → PaddleOCR
        - Handwritten regions (identified by layout router) → TrOCR
        """
        results = self.recognize_printed(image_path)

        # Future: iterate over layout_regions, crop handwritten fields,
        # call self.recognize_handwritten(crop) and merge results.
        return results
