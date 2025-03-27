import os
import sys
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from src.exception import CustomError
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')  # ✅ Fixed attribute type
    model_report_file_path: str = os.path.join('artifacts', 'model_report.txt')  # ✅ Fixed attribute type

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and testing input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1], train_array[:, -1], test_array[:, :-1], test_array[:, -1]
            )
            
            models = {
                'RandomForestRegressor': RandomForestRegressor(),
                'DecisionTreeRegressor': DecisionTreeRegressor(),
                'GradientBoostingRegressor': GradientBoostingRegressor(),
                'LinearRegression': LinearRegression(),
                'KNeighborsRegressor': KNeighborsRegressor(),
                'CatBoostRegressor': CatBoostRegressor(verbose=0),
                'XGBRegressor': XGBRegressor(),
                'AdaBoostRegressor': AdaBoostRegressor()
            }
            
            logging.info("Evaluating models")
            model_report = evaluate_models(X_train, y_train, X_test, y_test, models)
            
            best_model_score = max(model_report.values())  # ✅ Removed unnecessary `sorted()`
            
            best_model_name = max(model_report, key=model_report.get)  # ✅ More Pythonic way to get best model name
            best_model = models[best_model_name]
            
            logging.info(f"Best model found: {best_model_name} with score: {best_model_score}")
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            predicted = best_model.predict(X_test)
            model_r2_score = r2_score(y_test, predicted)  # ✅ Renamed variable
            
            return model_r2_score  # ✅ Now correctly returning the model performance
                
        except Exception as e:
            logging.error("Error occurred during model training")  # ✅ Moved logging before exception
            raise CustomError(e, sys)
