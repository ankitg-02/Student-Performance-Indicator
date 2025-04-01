import sys
import pandas as pd
from src.exception import CustomError
import dill

class PredictPipeline():
    def __init__(self):
        pass
    
    def predict(self, features):
        try:
            model_path = 'artifacts/model.pkl'
            preprocessor_path = 'artifacts/preprocessor.pkl'
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features)
            predictions = model.predict(data_scaled)
            return predictions
        except Exception as e:
            raise CustomError(e, sys)
    
class CustomData:
    def __init__(self,
                 gender:str,
                 race_ethincity:str,
                 parental_level_of_education:str,
                 lunch:str,
                 test_preparation_course:str,
                 reading_score:int,
                 writing_score:str):
        self.gender=gender
        self.race_ethincity=race_ethincity
        self.parental_level_of_education=parental_level_of_education
        self.lunch=lunch
        self.test_preparation_course=test_preparation_course
        self.reading_score=reading_score
        self.writing_score=writing_score
        
    def get_data_as_df(self):
        try:
            custom_data_input_dict={
                'gender':[self.gender],
                'race_ethinicity':[self.race_ethincity],
                'parental level of education':[self.parental_level_of_education],
                'lunch':[self.lunch],
                'test preparation course':[self.test_preparation_course],
                'reading score':[self.reading_score],
                'writing score':[self.writing_score]
            }
            df=pd.DataFrame(custom_data_input_dict)
            return df
        
        except Exception as e:
            raise CustomError(e, sys) from e
    
def load_object(file_path):
    try:
        with open(file_path, 'rb') as file:
            return dill.load(file)
    except Exception as e:
        raise CustomError(e, sys)