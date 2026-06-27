"""
Utility for loading and validating evaluation datasets.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from pydantic import ValidationError

from evaluation.config import config
from evaluation.logger import logger
from evaluation.models import EvaluationQuestion


class DatasetLoader:
    """
    Loads and validates the evaluation dataset.
    """

    def __init__(self, dataset_path: Path | None = None):
        self.dataset_path = dataset_path or config.DATASET_PATH

    def load(self) -> List[EvaluationQuestion]:
        """
        Load and validate all evaluation questions.

        Returns
        -------
        List[EvaluationQuestion]

        Raises
        ------
        FileNotFoundError
            If dataset file does not exist.

        RuntimeError
            If JSON is invalid or question validation fails.
        """

        logger.info("Loading evaluation dataset...")

        if not self.dataset_path.exists():
            logger.error(f"Dataset not found: {self.dataset_path}")
            raise FileNotFoundError(self.dataset_path)

        try:
            with self.dataset_path.open(
                "r",
                encoding="utf-8"
            ) as file:

                raw_questions = json.load(file)

        except json.JSONDecodeError as e:

            logger.exception("Invalid JSON format.")
            raise RuntimeError("Dataset JSON is invalid.") from e

        questions: List[EvaluationQuestion] = []

        for index, question in enumerate(raw_questions, start=1):

            try:

                questions.append(
                    EvaluationQuestion.model_validate(question)
                )

            except ValidationError as e:

                logger.error(
                    f"Question {index} failed validation.\n{e}"
                )

                raise RuntimeError(
                    f"Dataset validation failed on question {index}."
                ) from e

        logger.info(
            f"Successfully loaded {len(questions)} questions."
        )

        return questions