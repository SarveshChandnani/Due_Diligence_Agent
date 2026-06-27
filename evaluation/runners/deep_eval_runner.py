"""
DeepEval Runner

Coordinates the evaluation pipeline.
"""

from __future__ import annotations

import time
from typing import List

from deepeval.test_case import LLMTestCase

from evaluation.logger import logger
from evaluation.metrics.evaluator import MetricEvaluator
from evaluation.models import (
    EvaluationQuestion,
    EvaluationResult,
    RAGResponse,
)
from evaluation.runners.rag_runner import RAGRunner


class DeepEvalRunner:

    def __init__(
        self,
        rag_runner: RAGRunner,
    ):

        self.rag_runner = rag_runner

        self.metric_evaluator = MetricEvaluator()

    def evaluate_question(
        self,
        question: EvaluationQuestion,
    ) -> EvaluationResult:

        logger.info(
            "Evaluating Question %d",
            question.id,
        )

        start = time.perf_counter()

        try:

            rag_response: RAGResponse = self.rag_runner.ask(
                question.question
            )

            test_case = LLMTestCase(

               input=rag_response.question,
               actual_output=rag_response.answer,
               retrieval_context=rag_response.contexts,
               context=rag_response.contexts,

            )

            metric_scores = self.metric_evaluator.evaluate(
                test_case
            )

            execution_time = round(
                time.perf_counter() - start,
                3,
            )

            return EvaluationResult(

                id=question.id,

                category=question.category,

                difficulty=question.difficulty,

                question=question.question,

                answer=rag_response.answer,

                contexts=rag_response.contexts,

                metrics=metric_scores,

                execution_time=execution_time,

            )

        except Exception as e:

            logger.exception(e)

            execution_time = round(
                time.perf_counter() - start,
                3,
            )

            return EvaluationResult(

                id=question.id,

                category=question.category,

                difficulty=question.difficulty,

                question=question.question,

                answer="",

                contexts=[],

                metrics=[],

                execution_time=execution_time,

                error=str(e),

            )

    def evaluate_dataset(
        self,
        dataset: List[EvaluationQuestion],
    ) -> List[EvaluationResult]:

        results = []

        for question in dataset:

            results.append(

                self.evaluate_question(question)

            )

        return results