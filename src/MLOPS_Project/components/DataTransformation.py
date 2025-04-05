import os
import urllib.request as request
import zipfile

import pandas as pd
from sklearn.model_selection import train_test_split

from src.MLOPS_Project import logger
from src.MLOPS_Project.entity.config_entity import DataTransformationConfig


class DataTransformation:

    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def train_test_splitting(self):
        data = pd.read_csv(self.config.data_path)

        train, test = train_test_split(data)

        train.to_csv(f"{self.config.root_dir}/train.csv", index=False)
        test.to_csv(f"{self.config.root_dir}/test.csv", index=False)

        logger.info("Did split the data into train and test")
