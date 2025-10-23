from config.paths_config import *
from config.data_ingestion_config import *
from src.logger import get_logger
from src.custom_exception import CustomException
from src.data_ingestion import DataIngestion
from src.data_processing import DataProcessing
from src.model_training import ModelTraining

logger = get_logger(__name__)


if __name__ == "__main__":
    try:
        ingest = DataIngestion(DATASET_NAME, TARGET_DIR)
        ingest.run()

        processor = DataProcessing(INPUT_FILE_PATH, OUTPUT_FILE_PATH)
        processor.run()

        trainer = ModelTraining(PROCESSED_DATA_PATH)
        trainer.run()

    except Exception as e:
        logger.error(f"Pipeline terminated due to error: {e}")
        raise CustomException("Pipeline terminated")