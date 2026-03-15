from typing import Union
from .reranker import CrossEncoderReranker

class RerankerFactory:
    """
    Factory class to create different types of rerankers.
    Follows the Factory Method design pattern.
    """

    @staticmethod
    def get_reranker(reranker_type: str = "cross-encoder", **kwargs) -> Union[CrossEncoderReranker]:
        """
        Returns a reranker instance based on the specified type.

        Args:
            reranker_type (str): Type of reranker ('cross-encoder').
            **kwargs: Additional arguments for the reranker initialization.

        Returns:
            An instance of a reranker.
        """
        if reranker_type == "cross-encoder":
            return CrossEncoderReranker(**kwargs)
        else:
            raise ValueError(f"Unknown reranker type: {reranker_type}")
