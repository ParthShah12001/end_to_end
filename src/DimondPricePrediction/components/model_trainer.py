from src.DimondPricePrediction.logger import logging
from src.DimondPricePrediction.exception import customexception
from src.DimondPricePrediction.utils.utils import save_object
from src.DimondPricePrediction.utils.utils import evaluate_model    
import pandas as pd
import os
import sys 
from sklearn.linear_model import LinearRegression
from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self,train_path,test_path):
        try:
            logging.info("Splitting into test and train data")
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)
            x_train = train_data.iloc[:,:-1]
            y_train = train_data.iloc[:,-1]
            x_test = test_data.iloc[:, :-1]
            y_test = test_data.iloc[:, -1]
            
            models = {
                "LinearRegression":LinearRegression()
            }

            model_report:dict=evaluate_model(x_train,y_train,x_test,y_test,models)

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]
            print('\n====================================================================================\n')
            print(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )


            logging.info("Model created successfully")

        except Exception as e:
            logging.info('Exception occured at Model Training')
            raise customexception(e,sys)