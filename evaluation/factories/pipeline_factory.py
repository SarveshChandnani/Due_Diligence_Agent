"""
Pipeline Factory.

Responsible for constructing the complete evaluation pipeline.

This is the only place where dependencies should be instantiated.
"""

from evaluation.pipeline.evaluation_pipeline import EvaluationPipeline

from evaluation.runners.rag_runner import RAGRunner
from evaluation.runners.deep_eval_runner import DeepEvalRunner

from evaluation.utils.dataset_loader import DatasetLoader

from evaluation.metrics.summary import SummaryGenerator

from evaluation.reports.report_generator import ReportGenerator


class PipelineFactory:
    """
    Factory responsible for constructing the evaluation pipeline.
    """

    @staticmethod
    def create() -> EvaluationPipeline:
        """
        Build and return a fully configured EvaluationPipeline.
        """

        # -----------------------------
        # Infrastructure
        # -----------------------------

        dataset_loader = DatasetLoader()

        rag_runner = RAGRunner()

        # -----------------------------
        # Evaluation
        # -----------------------------

        evaluator = DeepEvalRunner(
            rag_runner=rag_runner
        )

        summary_generator = SummaryGenerator()

        # -----------------------------
        # Reporting
        # -----------------------------

        report_generator = ReportGenerator()

        # -----------------------------
        # Pipeline
        # -----------------------------

        return EvaluationPipeline(
            dataset_loader=dataset_loader,
            rag_runner=rag_runner,
            evaluator=evaluator,
            summary_generator=summary_generator,
            report_generator=report_generator,
        )