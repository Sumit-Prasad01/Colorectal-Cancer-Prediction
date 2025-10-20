import os
import zipfile
import shutil
import kagglehub
from src.logger import get_logger
from config.data_ingestion_config import DATASET_NAME, TARGET_DIR
from src.custom_exception import CustomException

logger = get_logger(__name__)


class DataIngestion:
    def __init__(self, dataset_name: str, target_dir: str):
        """
        Handles downloading and extracting datasets from KaggleHub.
        """
        self.dataset_name = dataset_name
        self.target_dir = target_dir

    def create_raw_dir(self) -> str:
        """
        Creates (if needed) a 'raw' directory inside the target directory.
        """
        raw_dir = os.path.join(self.target_dir, "raw")
        try:
            os.makedirs(raw_dir, exist_ok=True)
            logger.info(f"Created or verified directory: {raw_dir}")
            return raw_dir
        except Exception as e:
            logger.error(f"Error creating {raw_dir}: {e}")
            raise CustomException(f"Failed to create {raw_dir}: {e}")

    def extract_csv(self, source_path: str, raw_dir: str):
        """
        Extracts CSVs from the downloaded dataset path (ZIP or directory).
        """
        try:
            # Case 1: Dataset is a ZIP file
            if source_path.endswith(".zip"):
                logger.info("ZIP file detected — extracting CSV files...")

                with zipfile.ZipFile(source_path, "r") as zip_ref:
                    csv_files = [f for f in zip_ref.namelist() if f.endswith(".csv")]

                    if not csv_files:
                        raise CustomException("No CSV files found in the ZIP archive.")

                    for csv_file in csv_files:
                        zip_ref.extract(csv_file, raw_dir)
                        logger.info(f"Extracted: {csv_file} → {raw_dir}")

                logger.info("CSV extraction from ZIP completed successfully.")

            # Case 2: Dataset is a directory
            elif os.path.isdir(source_path):
                logger.info("Directory detected — searching for CSV files...")
                csv_found = False

                for root, _, files in os.walk(source_path):
                    for file in files:
                        if file.endswith(".csv"):
                            csv_found = True
                            src_file = os.path.join(root, file)
                            shutil.copy(src_file, os.path.join(raw_dir, file))
                            logger.info(f"Copied: {file} → {raw_dir}")

                if not csv_found:
                    raise CustomException("No CSV files found in the downloaded dataset folder.")

                logger.info("CSV extraction from directory completed successfully.")

            else:
                raise CustomException("Invalid dataset format. Expected .zip or directory.")

        except Exception as e:
            logger.error(f"Error while extracting CSVs from {source_path}: {e}")
            raise CustomException(f"Failed to extract CSVs: {e}")

    def download_dataset(self, raw_dir: str):
        """
        Downloads the dataset using KaggleHub and extracts CSV files.
        """
        try:
            logger.info(f"Downloading dataset '{self.dataset_name}' from KaggleHub...")
            dataset_path = kagglehub.dataset_download(self.dataset_name)

            if not dataset_path or not os.path.exists(dataset_path):
                raise CustomException(f"KaggleHub download failed — path not found: {dataset_path}")

            logger.info(f"Dataset downloaded successfully to: {dataset_path}")
            self.extract_csv(dataset_path, raw_dir)

        except Exception as e:
            logger.error(f"Error while downloading dataset '{self.dataset_name}': {e}")
            raise CustomException(f"Failed to download dataset '{self.dataset_name}': {e}")

    def run(self):
        """
        Executes the full data ingestion pipeline.
        """
        try:
            raw_dir = self.create_raw_dir()
            self.download_dataset(raw_dir)
            logger.info("Data ingestion pipeline completed successfully.")
        except Exception as e:
            logger.error(f"Data ingestion pipeline failed: {e}")
            raise CustomException(f"Failed to run data ingestion pipeline: {e}")


if __name__ == "__main__":
    try:
        ingest = DataIngestion(DATASET_NAME, TARGET_DIR)
        ingest.run()
    except Exception as e:
        logger.error(f"Pipeline terminated due to error: {e}")
