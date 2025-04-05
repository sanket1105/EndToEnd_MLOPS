import os
import urllib.request as request
import zipfile

import pandas as pd
from sklearn.model_selection import train_test_split

from src.MLOPS_Project import logger
from src.MLOPS_Project.components.DataTransformation import DataTransformation
from src.MLOPS_Project.config.config import ConfigurationManager
from src.MLOPS_Project.entity.config_entity import DataTransformationConfig

STAGE_NAME = "Data Transformation Stage"


class DataTransformationTrainingPipeline:
    def __init__(self):
        pass

    def initiate_data_transformation(self):

        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config)
        data_transformation.train_test_splitting()


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataTransformationTrainingPipeline()
        obj.initiate_data_transformation()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
