"""
RAG Runner.

Responsible for interacting with the RAG system.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from app.documents.processor import DocumentProcessor

from evaluation.config import config
from evaluation.logger import logger
from evaluation.models import RAGResponse


class RAGRunner:
    """
    Wrapper around DocumentProcessor.

    Responsible for:

    - Loading documents
    - Indexing documents
    - Asking questions
    """

    def __init__(self) -> None:
        """
        Create an empty runner.

        Heavy initialization is intentionally deferred
        until initialize() is called.
        """

        self.processor: DocumentProcessor | None = None
        self.documents: List[Path] = []

    def initialize(
        self,
        documents: List[Path],
    ) -> None:
        """
        Initialize the RAG system.

        Parameters
        ----------
        documents
            Documents to ingest.
        """

        logger.info("Initializing DocumentProcessor...")

        self.documents = documents

        self.processor = DocumentProcessor(
            session_id=config.SESSION_ID
        )

        logger.info(
            "Indexing %d document(s)...",
            len(documents),
        )

        self.processor.ingest_multiple_documents(
            [str(doc) for doc in documents]
        )

        logger.info("Document indexing completed.")

    def ask(
        self,
        question: str,
    ) -> RAGResponse:
        """
        Ask a question to the RAG.
        """

        if self.processor is None:
            raise RuntimeError(
                "RAGRunner has not been initialized."
            )

        logger.info("Running question: %s", question)

        response = self.processor.ask(question)

        logger.info("Question completed.")

        return response

    def close(self) -> None:
        """
        Release resources.
        """

        logger.info("Closing RAG Runner...")

        if self.processor is not None:
            self.processor.close()
            self.processor = None

        logger.info("RAG Runner closed.")

    def __enter__(self):
        """
        Context manager support.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Cleanup on exit.
        """
        self.close()