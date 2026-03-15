import re
from typing import List, Optional

class PIIMasker:
    """
    Professional-grade PII Masking layer for sanitizing user queries and documents.
    Prevents leakage of sensitive information to external LLM providers.
    """

    def __init__(self):
        """
        Initializes common PII patterns for detection and replacement.
        """
        self.pii_patterns = {
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "phone": r'\b(\+\d{1,2}\s?)?1?\-?\.?\s?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b',
            "credit_card": r'\b(?:\d[ -]*?){13,16}\b',
            "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
            "ipv4": r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'
        }

    def mask(self, text: str, labels: Optional[List[str]] = None) -> str:
        """
        Detects and masks PII in the given text based on specified labels.

        Args:
            text (str): Input text containing potential PII.
            labels (Optional[List[str]]): List of PII types to mask (e.g., ['email', 'phone']).
                                         If None, masks all known PII types.

        Returns:
            str: Masked text where PII is replaced by labels (e.g., "[EMAIL_MASKED]").
        """
        masked_text = text
        labels_to_mask = labels or self.pii_patterns.keys()

        for label in labels_to_mask:
            if label in self.pii_patterns:
                pattern = self.pii_patterns[label]
                replacement = f"[{label.upper()}_MASKED]"
                masked_text = re.sub(pattern, replacement, masked_text)

        return masked_text

    def unmask(self, text: str) -> str:
        """
        Note: True unmasking requires stateful storage of original values.
        This implementation is for redaction purposes, but could be extended
        to support a vault-based tokenization strategy for recovery.
        """
        # Placeholder for extension to stateful unmasking
        return text

if __name__ == "__main__":
    # Example usage
    masker = PIIMasker()
    query = "Hi, my email is john.doe@example.com and my phone number is 123-456-7890. My IP is 192.168.1.1."
    
    masked_query = masker.mask(query)
    # print(f"Original: {query}")
    # print(f"Masked: {masked_query}")
