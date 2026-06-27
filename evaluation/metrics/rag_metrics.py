"""
RAG Metrics Configuration

Only reference-free metrics are used.

These metrics do NOT require expected answers.
"""

from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    HallucinationMetric,
)


class RAGMetrics:
    """
    Factory class for all DeepEval metrics.
    """

    @staticmethod
    def get_metrics():

        return [

            FaithfulnessMetric(
                threshold=0.75,
                model="gpt-4.1-mini",
                include_reason=True
            ),

            AnswerRelevancyMetric(
                threshold=0.75,
                model="gpt-4.1-mini",
                include_reason=True
            ),

            # ContextualPrecisionMetric(
            #     threshold=0.75,
            #     model="gpt-4.1-mini",
            #     include_reason=True
            # ),

            HallucinationMetric(
                threshold=0.75,
                model="gpt-4.1-mini",
                include_reason=True
            ),
        ]