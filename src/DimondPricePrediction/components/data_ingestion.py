from src.DimondPricePrediction.logger import logging
from src.DimondPricePrediction.exception import customexception
import sys
import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
@dataclass
class DataIngestionConfig:
    raw_data_path:str = os.path.join("artifacts","raw.csv")
    train_data_path:str = os.path.join("artifacts","train.csv")
    test_data_path:str = os.path.join("artifacts","test.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data ingestion started")
        try:
            data = pd.read_csv(os.path.join("notebooks/data","gemstone.csv"))
            logging.info("I have successful ready data in dataframe df")

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            data.to_csv(self.ingestion_config.raw_data_path,index=False)
            logging.info("Raw data saved in artifacts")

            train_data,test_data = train_test_split(data,test_size=0.25,random_state=26)

            train_data.to_csv(self.ingestion_config.train_data_path,index=False)
            logging.info("Train data saved in artifacts")

            test_data.to_csv(self.ingestion_config.test_data_path,index=False)
            logging.info("Test data saved in artifacts")

            logging.info("Data Ingestion completed successfully")
            
            return (self.ingestion_config.train_data_path,self.ingestion_config.test_data_path)

        except Exception as e:
            logging.info("An error occured in Data Ingestion step")
            raise customexception(e,sys)
