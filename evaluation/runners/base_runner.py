from abc import ABC, abstractmethod

from evaluation.models import RAGResponse
from pathlib import Path
from typing import List


class BaseRAGRunner(ABC):
    """
    Abstract base class for all RAG runners.
    """


    @abstractmethod
    def ask(self, question: str) -> RAGResponse:
        """
        Ask a question to the RAG system.
        """
        pass

    @abstractmethod
    def close(self):
        """
        Release all resources.
        """
        pass

    @abstractmethod
    def initialize(self, documents: List[Path]) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()