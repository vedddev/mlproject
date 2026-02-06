import sys
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "8"
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor,RandomForestRegressor,GradientBoostingRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score

from src.utils import save_object,evaluate_models
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass


@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts',"model.pkl")
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Spliting training and test data")
            X_train = train_array[:, :-1]
            y_train = train_array[:, -1]

            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]
            models={
                "Random Forest":RandomForestRegressor(n_jobs=-1),
                "Decision Tree":DecisionTreeRegressor(),
                "XGB":XGBRegressor(n_jobs=-1),
                "Linear Regression":LinearRegression(),
                "Support Vector":SVR(),
                "Adaboost":AdaBoostRegressor(),
                "Gradient Boost":GradientBoostingRegressor(),
                "CatBoosting Regressor":CatBoostRegressor(verbose=False),
                "KNN":KNeighborsRegressor(n_jobs=-1)
            }
            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)
            
            best_model_score=max(sorted(model_report.values()))
            
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model=models[best_model_name]
            
            if best_model_score<0.6:
                raise CustomException("No best model Found")
            logging.info(f"Best found model on both training and testing dataset")
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            predicted=best_model.predict(X_test)
            final_r2=r2_score(y_test,predicted)
            return final_r2
        except Exception as e:
            raise CustomException(e,sys)