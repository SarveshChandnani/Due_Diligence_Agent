"""
Report Generator.

Writes evaluation reports to disk.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pandas as pd

from evaluation.config import config
from evaluation.logger import logger
from evaluation.models import (
    EvaluationResult,
    EvaluationSummary,
)


class ReportGenerator:
    """
    Generates CSV and JSON reports for evaluation results.
    """

    def __init__(
        self,
        output_dir: Path | None = None,
    ):

        self.output_dir = output_dir or config.OUTPUT_DIR

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def generate(
        self,
        results: List[EvaluationResult],
        summary: EvaluationSummary,
    ) -> None:
        """
        Generate all evaluation reports.
        """

        logger.info("Generating evaluation reports...")

        self._write_csv(results)

        self._write_results_json(results)

        self._write_summary_json(summary)

        logger.info(
            "Reports saved to %s",
            self.output_dir.resolve(),
        )

    # ----------------------------------------------------

    def _write_csv(
        self,
        results: List[EvaluationResult],
    ) -> None:

        rows = []

        for result in results:

            row = {
                "id": result.id,
                "category": result.category,
                "difficulty": result.difficulty,
                "question": result.question,
                "answer": result.answer,
                "execution_time": result.execution_time,
                "error": result.error,
            }

            for metric in result.metrics:

                row[f"{metric.name}_score"] = metric.score
                row[f"{metric.name}_passed"] = metric.passed

            rows.append(row)

        df = pd.DataFrame(rows)

        df.to_csv(
            self.output_dir / "evaluation_results.csv",
            index=False,
        )

    # ----------------------------------------------------

    def _write_results_json(
        self,
        results: List[EvaluationResult],
    ) -> None:

        with open(
            self.output_dir / "evaluation_results.json",
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(

                [
                    result.model_dump()
                    for result in results
                ],

                file,

                indent=4,

                ensure_ascii=False,

            )

    # ----------------------------------------------------

    def _write_summary_json(
        self,
        summary: EvaluationSummary,
    ) -> None:

        with open(
            self.output_dir / "evaluation_summary.json",
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(

                summary.model_dump(),
                file,
                indent=4,
                ensure_ascii=False,
            )