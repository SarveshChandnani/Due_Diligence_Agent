# from evaluation.logger import logger
from evaluation.logger import logger
# from evaluation.utils.dataset_loader import DatasetLoader
from evaluation.utils.dataset_loader import DatasetLoader

loader = DatasetLoader()

questions = loader.load()

logger.info(f"Loaded {len(questions)} questions.")

logger.info(questions[0])