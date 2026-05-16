import re

class KYCParser:
    def __init__(self):
        # Regular expressions for Nepali Citizenship and similar documents
        self.patterns = {
            "citizenship_no": r"(?:नागरिकता|ना\.प्र\.प\.नं|No\.|Number)\s*[:\-]?\s*([\d\-/]+)",
            "date_of_birth": r"(?:जन्म मिति|DOB|Date of Birth)\s*[:\-]?\s*([\d]{4}[/-][\d]{1,2}[/-][\d]{1,2})",
            # Fallbacks or heuristic patterns
        }

    def parse(self, text_blocks):
        """
        Parses raw text blocks to map to administrative fields.
        """
        full_text = " ".join([block['text'] for block in text_blocks])
        
        parsed_data = {
            "first_name": "",
            "last_name": "",
            "citizenship_no": "",
            "date_of_birth": "",
            "address": "",
            "confidence_score": 1.0 # default to 1.0, update based on average conf
        }

        # Calculate average confidence
        if text_blocks:
            parsed_data["confidence_score"] = sum([b['confidence'] for b in text_blocks]) / len(text_blocks)

        # Simple Regex extraction
        cit_match = re.search(self.patterns["citizenship_no"], full_text)
        if cit_match:
            parsed_data["citizenship_no"] = cit_match.group(1).strip()
            
        dob_match = re.search(self.patterns["date_of_birth"], full_text)
        if dob_match:
            parsed_data["date_of_birth"] = dob_match.group(1).strip()

        # Heuristic approach to names: Just extracting random capitalized/Nepali words as mock if regex fails.
        # In a robust implementation, use SpaCy NER for 'ne'
        words = full_text.split()
        if len(words) > 0 and not parsed_data["first_name"]:
            # Mock
            parsed_data["first_name"] = "Sample"
            parsed_data["last_name"] = "Name"
            
        return parsed_data
