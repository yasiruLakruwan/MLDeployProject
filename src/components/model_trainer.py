import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exeption import CustomExeption
from src.logger import logging
from src.utils import save_object, evaluate_model

@dataclass

class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')


class ModelTrainer:
    def __init__(self):
        self.modle_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Spliting training and testing data...!")
            # train test spliting.......!

            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Dission Tree": DecisionTreeRegressor(),
                "Gradiant Boosting":GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbour Classifier":KNeighborsRegressor(),
                "XGBoost Classifier":XGBRegressor(),
                "Catboosting Classifier":CatBoostRegressor(),
                "Adaboost Classifier": AdaBoostRegressor()
            }

            model_report:dict=evaluate_model(X_train=X_train,X_test=X_test,y_train=y_train,y_test=y_test,
                                             models= models)
            
            # to get the best model score from the dictionary 
            best_model_score = max(sorted(model_report.values()))

            # to get the best model name from dictionary
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomExeption("No best model found")
            
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.modle_trainer_config.trained_model_file_path,
                obj=best_model
            )
            predicted = best_model.predict(X_test)

            r2_square = r2_score(y_test,predicted)
            
            return r2_square

        except Exception as e:
            raise CustomExeption(e,sys)   