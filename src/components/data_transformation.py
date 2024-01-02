import sys 
from dataclasses import dataclass 
import numpy as np 
import pandas as pd 
from sklearn.compose import ColumnTransformer
#Sys.path.append('/home/fsgg7/projects')
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import customeException
from src.logger import logging
import os
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_object_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:

    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    
    def get_data_transformer_obj(self):
        try:
            numerical_columns=['writing_score','reading_score']
            categorical_columns=['gender','race_ethnicity', 'parental_level_of_education','lunch','test_preparation_course']
            num_pipeline=Pipeline(steps=[('imputer',SimpleImputer(strategy='median') ),( 'scaler', StandardScaler())])
            logging.info("numerical column transformation completed")
            cat_pipeline=Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),('onehotencoder', OneHotEncoder(sparse_output=False)),('scaler', StandardScaler(with_mean=False))]
                                  )
            logging.info("categorical column transformation completed")

            preprocessor= ColumnTransformer([('num pipeline', num_pipeline, numerical_columns),('cat pipeline', cat_pipeline,categorical_columns)])
            return preprocessor
        except Exception as e:
            raise customeException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info("reading train and test raw files and converting to dataframe completed")
            preprocessing_obj=self.get_data_transformer_obj()
            target_column_name='math_score'
            numerical_columns=['writing_score','reading_score']
            input_features_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_features_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]
            logging.info("applying feature transform to dataframes")

            input_features_train_arr=preprocessing_obj.fit_transform(input_features_train_df)
            input_features_test_arr=preprocessing_obj.transform(input_features_test_df)

            train_arr=np.c_[input_features_train_arr,np.array(target_feature_train_df)]
            test_arr=np.c_[input_features_test_arr,np.array(target_feature_test_df)]
            
            logging.info("saving preprocessing object")
            save_object(file_path=self.data_transformation_config.preprocessor_object_file_path,obj=preprocessing_obj)

            return (train_arr,test_arr,self.data_transformation_config.preprocessor_object_file_path)


        except Exception as e:
            raise customeException(e,sys)
            
