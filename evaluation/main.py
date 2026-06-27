from pathlib import Path

from evaluation.config import config

from evaluation.models import EvaluationRequest

from evaluation.factories.pipeline_factory import PipelineFactory
from evaluation.logger import logger


def main():

    pipeline = PipelineFactory.create()

    request = EvaluationRequest(
        documents=[
            Path("evaluation/docs/Zepto_pitchdeck.pdf")
        ],
        dataset_path=config.DATASET_PATH,
        output_directory=config.OUTPUT_DIR,
    )

    summary = pipeline.run(request)

    logger.info("=" * 80)
    logger.info("Evaluation Completed Successfully")
    logger.info("=" * 80)

    logger.info("Total Questions      : %d", summary.total_questions)
    logger.info("Successful           : %d", summary.successful)
    logger.info("Failed               : %d", summary.failed)


if __name__ == "__main__":
    main()