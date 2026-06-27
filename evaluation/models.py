"""
Pydantic models used throughout the evaluation framework.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from pathlib import Path


class EvaluationQuestion(BaseModel):
    """
    Represents a single evaluation question from the dataset.
    """

    id: int = Field(..., gt=0)

    difficulty: str = Field(
        ...,
        description="Easy | Medium | Hard"
    )

    category: str = Field(
        ...,
        min_length=2
    )

    question: str = Field(
        ...,
        min_length=5
    )


class RAGResponse(BaseModel):
    """
    Response returned by DocumentProcessor.ask().
    """

    question: str

    answer: str

    contexts: List[str]

    sources: Optional[List[dict]] = None


class MetricScore(BaseModel):

    name: str

    score: float = Field(
        ...,
        ge=0.0,
        le=1.0
    )

    threshold: float

    passed: bool

    reason: str | None = None


class EvaluationResult(BaseModel):
    """
    Complete evaluation result for one question.
    """

    id: int

    category: str

    difficulty: str

    question: str

    answer: str

    contexts: List[str]

    metrics: List[MetricScore] = Field(default_factory=list)

    execution_time: Optional[float] = None

    error: Optional[str] = None


class EvaluationSummary(BaseModel):
    """
    Overall evaluation summary.
    """

    total_questions: int

    successful: int

    failed: int

    success_rate: float = Field(
        ...,
        ge=0.0,
        le=100.0
    )

    metric_averages: Dict[str, float] = Field(
        default_factory=dict,
        description="Average score of every metric."
    )

    metric_pass_rate: Dict[str, float] = Field(
        default_factory=dict,
        description="Pass percentage of every metric."
    )

class EvaluationRequest(BaseModel):
    """
    Represents a complete evaluation request.

    This object contains everything required
    to execute an evaluation pipeline.
    """

    documents: List[Path] = Field(
        ...,
        min_length=1,
        description="Documents to ingest into the RAG."
    )

    dataset_path: Path = Field(
        ...,
        description="Path to the evaluation dataset."
    )

    output_directory: Path = Field(
        ...,
        description="Directory where reports will be generated."
    )