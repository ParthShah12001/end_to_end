import os
import sys
import pandas as pd
from src.DimondPricePrediction.logger import logging
from src.DimondPricePrediction.exception import customexception
from src.DimondPricePrediction.utils.utils import load_object

class PredictPipeline:

    def predict(self,features):
        try:
            preprocessor_path = os.path.join("artifacts","preprocessor.pkl")
            model_path = os.path.join("artifacts","model.pkl")

            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)
            scaled_data = preprocessor.transform(features)
            pred = model.predict(scaled_data)
            return pred
        except Exception as e:
            raise customexception(e,sys)
        
class CustomData:
    def __init__(self,carat,depth,table,cut,color,clarity):
        self.carat = carat
        self.depth = depth
        self.table = table
        self.cut = cut
        self.color = color
        self.clarity = clarity

    def get_as_dataframe(self):
        try:
            custom_input_data = {
        'carat': [self.carat],   # Wrap the scalar values in lists
        'depth': [self.depth],
        'table': [self.table],
        'cut': [self.cut],
        'color': [self.color],
        'clarity': [self.clarity]
    }

            df = pd.DataFrame(custom_input_data)
            logging.info("DataFrame created successfully")
            return df
        except Exception as e:
            logging.info("AN Error occured while creating DataFrame")
            raise customexception(e,sys)
        