import os
import sys
from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from src.exception import CustomError
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'best_model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Model training process started...")

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

            param_grid = {
                'RandomForestRegressor': {'n_estimators': [100, 200], 'max_depth': [10, 20]},
                'DecisionTreeRegressor': {'max_depth': [10, 20]},
                'GradientBoostingRegressor': {'n_estimators': [100, 200], 'learning_rate': [0.1, 0.2]},
                'LinearRegression': {},  # No hyperparameters for Linear Regression
                'KNeighborsRegressor': {'n_neighbors': [3, 5]},
                'CatBoostRegressor': {'iterations': [500], 'depth': [6]},
                'XGBRegressor': {'n_estimators': [100], 'learning_rate': [0.1]},
                'AdaBoostRegressor': {'n_estimators': [100], 'learning_rate': [0.1]}
            }

            logging.info("Performing hyperparameter tuning...")
            model_report = evaluate_models(X_train, y_train, X_test, y_test, models, param_grid)

            # Find the best model based on the highest R² score
            best_model_name = max(model_report, key=model_report.get)
            best_model = models[best_model_name]

            logging.info(f"Best model identified: {best_model_name}")

            # Retrain the best model on full training data
            best_model.fit(X_train, y_train)
            
            # Save the best model
            save_object(self.model_trainer_config.trained_model_file_path, best_model)

            # Evaluate on test set
            y_pred = best_model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            
            logging.info(f"Final R² score on test set: {r2:.4f}")
            print(f"🏆 Best Model: {best_model_name} | Final R² Score: {r2:.4f}")

            return r2

        except Exception as e:
            logging.error("Error occurred during model training", exc_info=True)
            raise CustomError(e, sys)
