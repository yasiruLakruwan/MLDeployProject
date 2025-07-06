# common function add here..................!

import os
import sys            
import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score
#from sklearn.model_selection import GridSearchCV

from src.exeption import CustomExeption

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomExeption(e, sys)

def evaluate_model(X_train,y_train,X_test,y_test,models):
    try:
        report = {}

        for model_name, model in models.items():
            #model = list(models.values())[i]

            model.fit(X_train, y_train) #train the model

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train,y_train_pred)
            test_model_score = r2_score(y_test,y_test_pred)

            report[model_name] = test_model_score

        return report
    except Exception as e:
        raise CustomExeption(e,sys)