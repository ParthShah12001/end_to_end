from src.DimondPricePrediction.logger import logging
from src.DimondPricePrediction.exception import customexception
from src.DimondPricePrediction.utils.utils import save_object
import os
import sys
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder,MinMaxScaler
from dataclasses import dataclass
import pandas as pd

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts","preprocessor.pkl")
    train_final_data_path:str = os.path.join("artifacts","train_final.csv")
    test_final_data_path:str = os.path.join("artifacts","test_final.csv")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation(self,numeric_col,categorical_col):
        logging.info("Data Transformation initiated")

        try:
            cut_categories = ["Fair","Good","Very Good","Premium","Ideal"]
            color_categories = ["D","E","F","G","H","I","J"]
            clarity_categories = ["I1","SI2","SI1","VS2","VS1","VVS2","VVS1","IF"]

            logging.info("Initiating Pipeline")

            num_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",MinMaxScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("ordinalencoding",OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories]))
                ]
            )

            preprocessor = ColumnTransformer([
                ('num_pipeline',num_pipeline,numeric_col),
                ('cat_pipeline',cat_pipeline,categorical_col)
            ])

            return preprocessor
        
        except Exception as e:
            logging.info("An Error occured in setting up the data transformation pipeline")
            raise customexception(e,sys)
        
    def initialize_data_transformation(self,train_path,test_path):
        try:
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)

            logging.info("Successfully read training and testing data \n")
            logging.info("Train data \n",train_data.head(5))
            logging.info("Test data \n",test_data.head(5))

            target_column = "price"
            drop_columns = [target_column,"id","x","y","z"]

            input_training_data = train_data.drop(drop_columns,axis=1)
            training_target_data = train_data[target_column]

            input_testing_data = test_data.drop(drop_columns,axis=1)
            testing_target_data = test_data[target_column]

            numeric_col = input_training_data.columns[input_training_data.dtypes != "object"]
            categorical_col = input_testing_data.columns[input_testing_data.dtypes == "object"]

            preprocessing_obj = self.get_data_transformation(numeric_col,categorical_col)

            logging.info("Final Transformation of data start")

            input_features_train = pd.DataFrame(preprocessing_obj.fit_transform(input_training_data),columns = preprocessing_obj.get_feature_names_out())
            input_features_test = pd.DataFrame(preprocessing_obj.transform(input_testing_data),columns = preprocessing_obj.get_feature_names_out())
            

            logging.info("Transformation successfully completed")

            final_training_data = pd.concat([input_features_train, pd.DataFrame(training_target_data, columns=[target_column])], axis=1)
            final_testing_data = pd.concat([input_features_test, pd.DataFrame(testing_target_data, columns=[target_column])], axis=1)

            final_training_data.to_csv(self.data_transformation_config.train_final_data_path,index=False)
            final_testing_data.to_csv(self.data_transformation_config.test_final_data_path,index=False)

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return(self.data_transformation_config.train_final_data_path,self.data_transformation_config.test_final_data_path)
        
        except Exception as e:
            logging.info("Error occured while transforming the data and saving it in csv format")
            raise customexception(e,sys)