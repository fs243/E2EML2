import os
import sys 
from catboost import CatBoostRegressor
from dataclasses import dataclass
from sklearn.ensemble import (AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor)
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
sys.path.append('/home/fsgg7/projects/')
from src.exception import customeException
from src.logger import logging
from src.utils import save_object

from src.components.utils import evaluate_models

class modelTrainerConfig():
    trained_model_path=os.path.join("artifacts","model.pkl")

class modelTrainer:
    def __init__(self):
        self.model_trainer_config= modelTrainerConfig()


    def initiate_model_trainer(self,train_array, test_array):

        try:
           logging.info("splitting training and test input data")
           X_train,y_train, X_test, y_test = \
           (train_array[:,:-1], train_array[:,-1], test_array[:,:-1], test_array[:,-1])

           models={
            "adaBoost": AdaBoostRegressor(),
            "decision tree": DecisionTreeRegressor(),
            "kneighborns": KNeighborsRegressor(),
            "gradient boosting":GradientBoostingRegressor(),
            "linear regression": LinearRegression(),
            "xgbRegressor": XGBRegressor(),
            "catboost":CatBoostRegressor(verbose=False),
            "adaBoost":AdaBoostRegressor()

            }
           
           model_report:dict= evaluate_models(X_train=X_train, y_train=y_train,X_test=X_test ,y_test=y_test, models=models)
           best_model_score= max(sorted(model_report.values()) )

           best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]
           best_model=models[best_model_name]
           if best_model_score < 0.6:
               raise customeException('no best model found')
           logging.info("best model on both training and testing dataset")


           save_object(file_path=self.model_trainer_config.trained_model_path,obj=best_model)
           predicted=best_model.predict(X_test)
           r2_score1=r2_score(predicted,y_test)
           return r2_score1
        except Exception as e:
            raise customeException(e,sys)

if __name__=='__main__':        
    print("hello fahime")