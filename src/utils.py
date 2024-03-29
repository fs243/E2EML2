import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from catboost import CatBoostRegressor
from dataclasses import dataclass
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression 
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
sys.path.append('/home/fsgg7/projects/')
from src.exception import customeException
from src.logger import logging




def save_object(file_path, obj):
    try:
        
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise customeException(e,sys)   
def load_object(file_path):
    
    print("loading file path:", file_path)
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise customeException(e, sys)
    
def evaluate_models(X_train, y_train,X_test,y_test,models,params):
    try:
        report = {}
        
        for i in range(len(list(models))):
            model = list(models.values())[i]
            
            para = params[list(models.keys())[i]]
          
            gs=GridSearchCV(model,para,cv=3)

            gs.fit(X_train, y_train)  # Train model
            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)
            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise customeException(e, sys)
    
