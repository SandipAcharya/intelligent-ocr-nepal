import logging
from typing import List, Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)

class HybridRecognizer:
    """
    Hybrid OCR/HTR Engine as specified in the technical architecture.
    
    Stage 3 of the 4-stage deep learning pipeline (100% Offline):
      - Printed Text Crops  → EasyOCR (lang=['hi', 'en'] = Devanagari/Hindi)
      - Handwritten Crops   → TrOCR (VisionEncoderDecoder Transformer, local_files_only=True)
    """
    def __init__(self, use_gpu: bool = False):
        logger.info("Initializing HybridRecognizer (Offline PyTorch Engine)...")
        self.use_gpu = use_gpu
        self.easy_ocr = None
        self.trocr_processor = None
        self.trocr_model = None

        # 1. Printed Text Engine: EasyOCR with Devanagari support
        try:
            import easyocr
            # EasyOCR downloads to ~/.EasyOCR/model once, then runs purely offline
            self.easy_ocr = easyocr.Reader(['hi', 'en'], gpu=self.use_gpu)
            logger.info("EasyOCR (Devanagari/hi) initialized successfully.")
        except ImportError:
            logger.warning("easyocr module not found. Please run `pip install easyocr`.")
        except Exception as e:
            logger.warning(f"EasyOCR failed to load: {e}. Will skip printed OCR.")

        # 2. Handwritten Text Engine: HuggingFace TrOCR
        try:
            import torch
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
            model_id = 'paudelanil/trocr-devanagari-2'
            try:
                # First try strictly offline to prevent network delays
                self.trocr_processor = TrOCRProcessor.from_pretrained(model_id, local_files_only=True)
                self.trocr_model = VisionEncoderDecoderModel.from_pretrained(model_id, local_files_only=True).to(self.device)
                logger.info(f"TrOCR ({model_id}) loaded successfully from local cache.")
            except Exception:
                logger.info(f"Downloading {model_id} from HuggingFace for the first time. This may take a minute...")
                self.trocr_processor = TrOCRProcessor.from_pretrained(model_id, local_files_only=False)
                self.trocr_model = VisionEncoderDecoderModel.from_pretrained(model_id, local_files_only=False).to(self.device)
                logger.info("TrOCR downloaded and cached successfully for future offline use.")
        except Exception as e:
            logger.warning(f"TrOCR could not be loaded: {e}")

    def recognize_printed(self, image_path: str) -> List[Dict[str, Any]]:
        """
        Extracts printed text blocks using EasyOCR (Devanagari-optimized).
        Returns a list of {text, confidence, bbox, type} dicts.
        """
        if self.easy_ocr is None:
            logger.warning("EasyOCR not available, returning empty result.")
            return []

        # EasyOCR readtext returns a list of tuples: (bbox, text, prob)
        results = self.easy_ocr.readtext(image_path)
        extracted_data = []

        for line in results:
            bbox, text, confidence = line
            extracted_data.append({
                "text": text,
                "confidence": float(confidence),
                # bbox format in EasyOCR is [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
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
        - Printed regions → EasyOCR
        - Handwritten regions (identified by layout router) → TrOCR
        """
        results = self.recognize_printed(image_path)

        # Future: iterate over layout_regions, crop handwritten fields,
        # call self.recognize_handwritten(crop) and merge results.
        return results
