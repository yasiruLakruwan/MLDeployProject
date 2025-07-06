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
from src.utils import save_object

@dataclass

class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')
    # model saving path and model saving path added here................
    # move to the model training........................................

class ModelTrainer:
    def __init__(self):
        self.modle_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array,preprocessor_path):
        try:
            pass
        except Exception as e:
            raise CustomExeption(e,sys)   