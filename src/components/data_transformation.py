import sys
from dataclasses import dataclass
import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from src.logger import logging
from src.exception import CustomError
from src.utils import save_object
import pickle

@dataclass
class DataTransformationConfig:
    transformed_data_path: str = os.path.join('artifacts', 'transformed_data.csv')
    preprocessor_path: str = os.path.join('artifacts', 'preprocessor.pkl')
    
class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()
        
    def get_data_transformer_object(self):
        try:
            num_col=['reading score','writing score']
            cat_col=['gender','race/ethenicty',
                     'parental level of education','lunch',
                     'test preparation course']
            num_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )
            cat_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('onehotencoder',OneHotEncoder()),
                    ('scaler',StandardScaler())
                ]
            )
            
            logging.info('numerical columns scaling completed')
            logging.info('categorical columns encoding completed')
            
            preprocessor=ColumnTransformer(
                [
                    ('num_pipelines',num_pipeline,num_col),
                    ('cat_pipeline',cat_pipeline,cat_col),
                
                ]
            )
        except Exception as e:
            raise CustomError(e,sys)
    
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            
            logging.info('reading train and test data completed')
            logging.info('obtaining preprocessing object')
            
            target_col_name='math_score'
            numerical_col=['writing_score','reading_score']
            
            preprocessing_obj=self.get_data_transformer_object()
            
            input_feature_test_df=test_df.drop(columns=[target_col_name],axis=1)
            target_feature_test_df=test_df[target_col_name]
            
            input_feature_train_df=train_df.drop(columns=[target_col_name],axis=1)
            target_feature_train_df=train_df[target_col_name]
            
            logging.info(
                f'Applying preprocessing object on trainig dataframe and testing dataframe.'
            )
            
            input_feature_train_array=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_array=preprocessing_obj.fit_transform(input_feature_test_df)
            
            train_array=np.c_(
                input_feature_train_array,np.array(target_feature_train_df)
            )
            test_array=np.c_(
                input_feature_test_array,np.array(target_feature_test_df)
            )
            
            logging.info('saved preprocessing object.')
            
            save_object(
                file_path=self.data_transformation_config.processor_obj_file_path,
                obj=preprocessing_obj
            )
            
            return (
                train_array,
                test_array,
                self.data_transformation_config.processor_obj_file_path
            )
        except Exception as e:
            
            raise CustomError(e,sys)