"""
Summary Generator.

Generates aggregate statistics from evaluation results.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from evaluation.models import (
    EvaluationResult,
    EvaluationSummary,
)


class SummaryGenerator:
    """
    Generates an evaluation summary from all evaluation results.
    """

    @staticmethod
    def generate(
        results: List[EvaluationResult],
    ) -> EvaluationSummary:
        """
        Generate an EvaluationSummary.

        Parameters
        ----------
        results
            List of evaluation results.

        Returns
        -------
        EvaluationSummary
        """

        total_questions = len(results)

        successful = sum(
            result.error is None
            for result in results
        )

        failed = total_questions - successful

        success_rate = (
            round(successful / total_questions * 100, 2)
            if total_questions
            else 0.0
        )

        metric_scores: Dict[str, List[float]] = defaultdict(list)

        metric_passes: Dict[str, int] = defaultdict(int)

        metric_totals: Dict[str, int] = defaultdict(int)

        # ----------------------------------------
        # Aggregate every metric dynamically
        # ----------------------------------------

        for result in results:

            if result.error is not None:
                continue

            for metric in result.metrics:

                metric_scores[
                    metric.name
                ].append(metric.score)

                metric_totals[
                    metric.name
                ] += 1

                if metric.passed:

                    metric_passes[
                        metric.name
                    ] += 1

        # ----------------------------------------
        # Average score per metric
        # ----------------------------------------

        metric_averages = {

            name: round(
                sum(scores) / len(scores),
                3,
            )

            for name, scores in metric_scores.items()

            if scores

        }

        # ----------------------------------------
        # Pass percentage per metric
        # ----------------------------------------

        metric_pass_rate = {

            name: round(
                metric_passes[name]
                / metric_totals[name]
                * 100,
                2,
            )

            for name in metric_totals

        }

        return EvaluationSummary(

            total_questions=total_questions,

            successful=successful,

            failed=failed,

            success_rate=success_rate,

            metric_averages=metric_averages,

            metric_pass_rate=metric_pass_rate,

        )