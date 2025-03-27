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
            num_col = ['reading score', 'writing score']
            cat_col = ['gender', 'race/ethnicity',
                       'parental level of education', 'lunch',
                       'test preparation course']
            
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )
            
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehotencoder', OneHotEncoder(handle_unknown='ignore')),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )

            logging.info('Numerical columns scaling completed')
            logging.info('Categorical columns encoding completed')

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num_pipeline', num_pipeline, num_col),
                    ('cat_pipeline', cat_pipeline, cat_col),
                ]
            )

            return preprocessor  # ✅ Fix: Returning the preprocessor object

        except Exception as e:
            raise CustomError(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Reading train and test data completed')
            logging.info('Obtaining preprocessing object')

            target_col_name = 'math score'
            numerical_col = ['reading score', 'writing score']  # ✅ Fixed naming

            preprocessing_obj = self.get_data_transformer_object()

            input_feature_train_df = train_df.drop(columns=[target_col_name], axis=1)
            target_feature_train_df = train_df[target_col_name]

            input_feature_test_df = test_df.drop(columns=[target_col_name], axis=1)
            target_feature_test_df = test_df[target_col_name]

            logging.info('Applying preprocessing object on training and testing data')

            input_feature_train_array = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_array = preprocessing_obj.transform(input_feature_test_df)  # ✅ Fix: transform() instead of fit_transform()

            train_array = np.c_[
                input_feature_train_array, np.array(target_feature_train_df)
            ]
            test_array = np.c_[
                input_feature_test_array, np.array(target_feature_test_df)
            ]

            logging.info('Saving preprocessing object')

            save_object(
                file_path=self.transformation_config.preprocessor_path,  # ✅ Fix: Correct attribute reference
                obj=preprocessing_obj
            )

            return (
                train_array,
                test_array,
                self.transformation_config.preprocessor_path  # ✅ Fix: Correct attribute reference
            )

        except Exception as e:
            raise CustomError(e, sys)
