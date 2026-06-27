"""
Central configuration for the evaluation framework.
"""

from pathlib import Path
from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationConfig:
    """
    Global configuration for evaluation.
    """

    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

    EVALUATION_DIR: Path = PROJECT_ROOT / "evaluation"

    DATASET_PATH: Path = (
        EVALUATION_DIR
        / "datasets"
        / "questions.json"
    )

    DOCUMENTS_DIR: Path = (
        EVALUATION_DIR
        / "docs"
    )

    REPORTS_DIR: Path = (
        EVALUATION_DIR
        / "reports"
    )

    LOGS_DIR: Path = (
        EVALUATION_DIR
        / "logs"
    )


    OUTPUT_DIR: Path = (
        EVALUATION_DIR
        /"output"
    )

    # ------------------------------
    # RAG Configuration
    # ------------------------------

    SESSION_ID: str = "evaluation"

    TOP_K: int = 5

    # ------------------------------
    # DeepEval Thresholds
    # ------------------------------

    FAITHFULNESS_THRESHOLD: float = 0.80

    ANSWER_RELEVANCY_THRESHOLD: float = 0.80

    CONTEXTUAL_PRECISION_THRESHOLD: float = 0.75

    HALLUCINATION_THRESHOLD: float = 0.80

    # ------------------------------
    # Output Files
    # ------------------------------

    RESULTS_JSON: Path = (
        REPORTS_DIR
        / "results.json"
    )

    RESULTS_CSV: Path = (
        REPORTS_DIR
        / "results.csv"
    )

    SUMMARY_JSON: Path = (
        REPORTS_DIR
        / "summary.json"
    )



config = EvaluationConfig()

# Create required directories automatically
config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
config.LOGS_DIR.mkdir(parents=True, exist_ok=True)