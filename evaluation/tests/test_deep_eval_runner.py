from pathlib import Path

from evaluation.runners.rag_runner import RAGRunner
from evaluation.runners.deep_eval_runner import DeepEvalRunner
from evaluation.utils.dataset_loader import DatasetLoader


documents = [
    Path("evaluation/docs/Zepto_pitchdeck.pdf")
]

# Load Dataset
dataset = DatasetLoader().load()

# Test only one question first
dataset = dataset[:1]

with RAGRunner(documents) as rag_runner:

    evaluator = DeepEvalRunner(rag_runner)

    results = evaluator.evaluate_dataset(dataset)

    for result in results:

        print("=" * 80)

        print(result.model_dump())

        print("=" * 80)