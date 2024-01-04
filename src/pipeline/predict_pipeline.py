import sys 
import pandas as pd 

import os


sys.path.append("/home/fsgg7/projects")
from src.utils import load_object
from src.exception import customeException
class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path='src/components/artifacts/model.pkl'
            preprocessor_path='src/components/artifacts/proprocessor.pkl'
            #print("current working directory",os.getcwd())
            model=load_object(file_path=model_path)

            
            preprocessor=load_object(file_path=preprocessor_path)

            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        except Exception as e:
            raise customeException(e,sys)

class CustomData:
    def __init__(self,
            gender:str,
            race_ethnicity: str,
            parental_level_of_education:str,
            lunch:str,
            test_preparation_course:str,
            reading_score:int,
            writing_score:int        
                
                 ):
        self.gender=gender
        self.race_ethnicity=race_ethnicity
        self.parental_level_of_education=parental_level_of_education
        self.lunch=lunch
        self.test_preparation_course=test_preparation_course
        self.reading_score=reading_score
        self.writing_score=writing_score
    

    def get_data_as_frame(self):
        try:
            custom_data_input_dict={
                "gender":[self.gender],
                'race_ethnicity':[self.race_ethnicity],
                'parental_level_of_education':[self.parental_level_of_education],
                'lunch':[self.lunch],
                'test_preparation_course':[self.test_preparation_course],
                'reading_score':[self.reading_score],
                'writing_score':[self.writing_score]

            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise customeException(e,sys)

