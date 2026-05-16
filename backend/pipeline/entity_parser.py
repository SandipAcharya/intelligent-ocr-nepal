import re
import logging
from typing import Dict, List, Any
import spacy

logger = logging.getLogger(__name__)

class KYCParser:
    """
    Named Entity Recognition (NER) pipeline for Devanagari KYC details.
    Combines SpaCy NLP with RegEx fallback heuristics.
    """
    def __init__(self):
        logger.info("Initializing KYC Parser (SpaCy + RegEx heuristics)...")
        # In a real environment, load a fine-tuned 'ne_core_news_sm'
        try:
            self.nlp = spacy.load("en_core_web_sm") # Placeholder for Devanagari Model
        except OSError:
            logger.warning("SpaCy model not found. Using RegEx heuristics only.")
            self.nlp = None

        self.patterns = {
            "citizenship_no": r"(?:ना[\.॰\s]*प्र[\.॰\s]*नं[\.॰\s]*|ना[\.॰\s]*प्र[\.॰\s]*प[\.॰\s]*नं[\.॰\s]*|No\.|Number)\s*[:\-ः]*\s*([०-९0-9\-/\s]+)",
            "date_of_birth": r"(?:साल|Year)\s*[:\-ः]*\s*([०-९0-9]{4})\s*(?:महिना|Month)\s*[:\-ः]*\s*([०-९0-9]{1,2})\s*(?:गते|Day)\s*[:\-ः]*\s*([०-९0-9]{1,2})",
            "name": r"(?:नाम\s*थर|Full\s*Name)\s*[:\-ः]*\s*([^\n]+)",
        }

    def parse(self, text_blocks: List[Dict[str, Any]], doc_type: str = 'unknown') -> Dict[str, Any]:
        """
        Parses raw text blocks to map to administrative fields based on doc_type.
        """
        # Join text with newlines so regex can match until end of line
        full_text = "\n".join([block['text'] for block in text_blocks])
        
        parsed_data = {
            "first_name": "",
            "last_name": "",
            "citizenship_no": "",
            "date_of_birth": "",
            "address": "",
            "confidence_score": 1.0
        }

        if text_blocks:
            parsed_data["confidence_score"] = sum([b['confidence'] for b in text_blocks]) / len(text_blocks)

        # Extract Citizenship No
        cit_match = re.search(self.patterns["citizenship_no"], full_text)
        if cit_match:
            raw_no = cit_match.group(1).strip()
            parsed_data["citizenship_no"] = re.sub(r'\s+', '-', raw_no)
            
        # Extract DOB (combining year, month, day if matched)
        dob_match = re.search(self.patterns["date_of_birth"], full_text)
        if dob_match:
            parsed_data["date_of_birth"] = f"{dob_match.group(1)}-{dob_match.group(2)}-{dob_match.group(3)}"

        # Extract Name
        name_match = re.search(self.patterns["name"], full_text)
        if name_match:
            full_name = name_match.group(1).strip()
            # Clean up trailing artifacts
            full_name = re.sub(r'लिङ्ग.*', '', full_name).strip()
            name_parts = full_name.split()
            if len(name_parts) > 0:
                parsed_data["first_name"] = name_parts[0]
                parsed_data["last_name"] = " ".join(name_parts[1:])

        # Fallback block-by-block parsing if regex misses due to newline splits
        if not parsed_data["first_name"] or parsed_data["first_name"] == "ः":
            for i, block in enumerate(text_blocks):
                text = block['text'].strip()
                if "नाम थर" in text or "Full Name" in text:
                    # The name might be in the same block or the next block
                    # Also replace Visarga with colon for uniform splitting
                    text_normalized = text.replace("ः", ":")
                    if ":" in text_normalized and len(text_normalized.split(":")) > 1 and text_normalized.split(":")[1].strip():
                        full_name = text_normalized.split(":")[1].strip()
                    elif i + 1 < len(text_blocks):
                        full_name = text_blocks[i+1]['text'].strip()
                    else:
                        continue
                        
                    full_name = re.sub(r'लिङ्ग.*', '', full_name).strip()
                    name_parts = full_name.split()
                    if name_parts:
                        parsed_data["first_name"] = name_parts[0]
                        parsed_data["last_name"] = " ".join(name_parts[1:])
                    break

        return parsed_data
