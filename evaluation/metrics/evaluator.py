"""
Metric Evaluator

Responsible for executing DeepEval metrics.
"""

from __future__ import annotations

from typing import List

from deepeval.test_case import LLMTestCase

from evaluation.logger import logger
from evaluation.metrics.rag_metrics import RAGMetrics
from evaluation.models import MetricScore


class MetricEvaluator:
    """
    Executes all configured DeepEval metrics.
    """

    def __init__(self):

        self.metrics = RAGMetrics.get_metrics()

    def evaluate(
        self,
        test_case: LLMTestCase,
    ) -> List[MetricScore]:
        """
        Execute every configured metric.
        """

        scores: List[MetricScore] = []

        for metric in self.metrics:

            logger.info(
                "Running %s",
                metric.__class__.__name__,
            )

            for metric in self.metrics:
                try:
                    metric.measure(test_case)

                    scores.append(

                        MetricScore(
                            name=metric.__class__.__name__,
                            score=metric.score,
                            threshold=metric.threshold,
                            passed=metric.is_successful(),
                            reason=metric.reason,
                        )
                    )

                except Exception as e:
                    logger.exception(e)

                    scores.append(
                        MetricScore(
                            name=metric.__class__.__name__,
                            score=0.0,
                            passed=False,
                            reason=str(e),
                        )
                    )

        return scores