import os
import sys
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from src.logger import logging
from src.exception import CustomError
from src.utils import save_object
from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    preprocessor_path: str = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            num_col = ['reading score', 'writing score']
            cat_col = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']
            
            num_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            
            cat_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehotencoder', OneHotEncoder(handle_unknown='ignore')),
                ('scaler', StandardScaler(with_mean=False))
            ])

            logging.info('Preprocessing pipelines created.')

            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, num_col),
                ('cat_pipeline', cat_pipeline, cat_col),
            ])

            return preprocessor 

        except Exception as e:
            raise CustomError(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            target_col = 'math score'
            input_features_train = train_df.drop(columns=[target_col])
            target_train = train_df[target_col]

            input_features_test = test_df.drop(columns=[target_col])
            target_test = test_df[target_col]

            preprocessing_obj = self.get_data_transformer_object()

            input_train_array = preprocessing_obj.fit_transform(input_features_train)
            input_test_array = preprocessing_obj.transform(input_features_test)

            train_array = np.c_[input_train_array, target_train]
            test_array = np.c_[input_test_array, target_test]

            save_object(self.transformation_config.preprocessor_path, preprocessing_obj)

            return train_array, test_array

        except Exception as e:
            raise CustomError(e, sys)
