import os
import joblib
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from config.model_params import *

import mlflow
import mlflow.sklearn


logger = get_logger(__name__)

class ModelTraining:

    def __init__(self, processed_data_path):
        
        self.processed_data_path = processed_data_path
        self.model_dir = MODEL_DIR

        self.params_dist = LIGHTGBM_PARAMS
        self.random_search_params = RANDOM_SEARCH_PARAMS


        os.makedirs(self.model_dir, exist_ok = True)

        logger.info("Model training initialized")

    
    def load_data(self):
        try:
            self.X_train = joblib.load(os.path.join(self.processed_data_path, "X_train.pkl"))
            self.X_test = joblib.load(os.path.join(self.processed_data_path, "X_test.pkl"))
            self.y_train = joblib.load(os.path.join(self.processed_data_path, "y_train.pkl"))
            self.y_test = joblib.load(os.path.join(self.processed_data_path, "y_test.pkl"))

            logger.info("Data loaded for model trainig.")
        
        except Exception as e:
            logger.error(f"Error while loading data for model {e}")
            raise CustomException("Failed to load data for model.")
    
    
    def tarin_lgbm(self):
        try:
            logger.info("Initializing Our Model.")

            self.model =  lgb.LGBMClassifier(random_state = self.random_search_params["random_state"])

            logger.info("Starting our HyperParameter Tuning")

            random_search = RandomizedSearchCV(
                    estimator = self.model,
                    param_distributions = self.params_dist,
                    n_iter = self.random_search_params["n_iter"],
                    cv = self.random_search_params["cv"],
                    n_jobs = self.random_search_params["n_jobs"],
                    verbose = self.random_search_params["verbose"],
                    random_state = self.random_search_params["random_state"],
                    scoring = self.random_search_params["scoring"],
            )

            logger.info("Starting our HyperParameter Tunning")

            random_search.fit(self.X_train, self.y_train)

            logger.info("HyperParameter Tunning Completed")

            best_params = random_search.best_params_
            self.best_lgbm_model = random_search.best_estimator_

            logger.info(f"Best Parameters are {best_params}")

            joblib.dump(self.best_lgbm_model, os.path.join(self.model_dir, "model.pkl"))

            logger.info("Model trained and saved successfully.")
        
        except Exception as e:
            logger.error(f"Error During training model {e}")
            raise CustomException("Failed to train model")
    

    def evaluate_model(self):
        try:

            y_pred = self.best_lgbm_model.predict(self.X_test)

            y_proba = self.best_lgbm_model.predict_proba(self.X_test)[:,1] if len(self.y_test.unique()) == 2 else None

            accuracy =  accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, average = "weighted")
            recall = recall_score(self.y_test, y_pred, average = "weighted")
            f1 = f1_score(self.y_test, y_pred, average = "weighted")

            mlflow.log_metric("Accuracy", accuracy)
            mlflow.log_metric("Precision", precision)
            mlflow.log_metric("Recall", recall)
            mlflow.log_metric("F1 Score", f1)

            logger.info(f"Accuracy : {accuracy} ")
            logger.info(f"Precision : {precision} ")
            logger.info(f"Recall : {recall} ")
            logger.info(f"F1 : {f1} ")

            roc_auc = roc_auc_score(self.y_test, y_proba)

            mlflow.log_metric("Roc Auc Score", roc_auc)

            logger.info(f"Roc-Aur_Score : {roc_auc}")
        
            logger.info("Model Evaluation Done.")
        
        except Exception as e:
            logger.error(f"Error During evaluating model {e}")
            raise CustomException("Failed to evaluate model")
    

    def run(self):
        self.load_data()
        self.tarin_lgbm()
        self.evaluate_model()



if __name__ ==  "__main__":
    with mlflow.start_run():
        trainer = ModelTraining(PROCESSED_DATA_PATH)
        trainer.run()