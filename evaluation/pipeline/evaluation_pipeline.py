"""
Evaluation Pipeline.

Coordinates the complete evaluation workflow.

This class contains orchestration logic only.
It does not create dependencies or implement
business logic.
"""

from __future__ import annotations

from evaluation.logger import logger

from evaluation.models import (
    EvaluationRequest,
    EvaluationSummary,
)

from evaluation.runners.rag_runner import RAGRunner
from evaluation.runners.deep_eval_runner import DeepEvalRunner

from evaluation.utils.dataset_loader import DatasetLoader

from evaluation.metrics.summary import SummaryGenerator

from evaluation.reports.report_generator import ReportGenerator


class EvaluationPipeline:
    """
    Orchestrates the complete evaluation workflow.
    """

    def __init__(
        self,
        dataset_loader: DatasetLoader,
        rag_runner: RAGRunner,
        evaluator: DeepEvalRunner,
        summary_generator: SummaryGenerator,
        report_generator: ReportGenerator,
    ) -> None:

        self.dataset_loader = dataset_loader
        self.rag_runner = rag_runner
        self.evaluator = evaluator
        self.summary_generator = summary_generator
        self.report_generator = report_generator

    def run(
        self,
        request: EvaluationRequest,
    ) -> EvaluationSummary:
        """
        Execute the complete evaluation pipeline.

        Parameters
        ----------
        request
            Evaluation request.

        Returns
        -------
        EvaluationSummary
        """

        logger.info("=" * 80)
        logger.info("Starting Evaluation Pipeline")
        logger.info("=" * 80)

        try:

            # --------------------------------------------------
            # Load Dataset
            # --------------------------------------------------

            self.dataset_loader.dataset_path = request.dataset_path

            dataset = self.dataset_loader.load()

            logger.info(
                "Loaded %d evaluation question(s).",
                len(dataset),
            )

            # --------------------------------------------------
            # Initialize RAG
            # --------------------------------------------------

            self.rag_runner.initialize(
                request.documents
            )

            # --------------------------------------------------
            # Evaluate Dataset
            # --------------------------------------------------

            results = self.evaluator.evaluate_dataset(
                dataset
            )

            # --------------------------------------------------
            # Generate Summary
            # --------------------------------------------------

            summary = self.summary_generator.generate(
                results
            )

            # --------------------------------------------------
            # Generate Reports
            # --------------------------------------------------

            self.report_generator.output_dir = (
                request.output_directory
            )

            self.report_generator.generate(
                results=results,
                summary=summary,
            )

            logger.info("=" * 80)
            logger.info("Evaluation Pipeline Completed")
            logger.info("=" * 80)

            return summary

        except Exception:

            logger.exception(
                "Evaluation pipeline failed."
            )

            raise

        finally:

            self.rag_runner.close()