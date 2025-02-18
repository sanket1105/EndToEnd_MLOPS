from src.MLOPS_Project import logger
from src.MLOPS_Project.pipeline.DataIngestion import DataIngestionTrainingPipeline
from src.MLOPS_Project.pipeline.DataValidation import DataValidationTrainingPipeline

STAGE_NAME = "Data Ingestion Stage"

if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataIngestionTrainingPipeline()
        obj.initiate_data_ingestion()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e


STAGE_NAME = "Data Validation Stage"

if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataValidationTrainingPipeline()
        obj.initiate_data_validation()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
