import os
import sys
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomError

def save_object(file_path, obj):
    """Saves a Python object using dill serialization."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomError(e, sys)

def evaluate_models(x_train, y_train, x_test, y_test, models, param_grid):
    """Evaluates models using GridSearchCV and returns the R² score report."""
    try:
        report = {}
        
        for model_name, model in models.items():
            grid_search = GridSearchCV(model, param_grid[model_name], cv=3, n_jobs=-1, verbose=1)
            grid_search.fit(x_train, y_train)

            best_model = grid_search.best_estimator_
            best_model.fit(x_train, y_train)

            y_test_pred = best_model.predict(x_test)
            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_model_score
            
        return report
    except Exception as e:
        raise CustomError(e, sys)
