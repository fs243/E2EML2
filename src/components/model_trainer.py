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

from src.utils import evaluate_models

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
            'random forest':RandomForestRegressor()

            }
           params={
                "decision tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                'random forest':{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "gradient boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    # 'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "linear regression":{},
                "kneighborns":{
                    'n_neighbors':[5,7,9,11],
                    # 'weights':['uniform','distance'],
                    # 'algorithm':['ball_tree','kd_tree','brute']
                },
                "xgbRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "catboost":{
                    'depth': [6,8,10],
                    # 'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "adaBoost":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }
           
           model_report:dict= evaluate_models(X_train=X_train, y_train=y_train,X_test=X_test ,y_test=y_test, models=models,params=params)
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