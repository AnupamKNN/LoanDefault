import sys, os
import numpy as np

from loandefault.entity.config_entity import ModelTrainerConfig
from loandefault.entity.artifact_entity import ModelTrainerArtifact, DataTransformationArtifact

from loandefault.exception.exception import LoanDefaultException
from loandefault.logging.logger import logging

from loandefault.utils.main_utils.utils import save_object, load_object
from loandefault.utils.main_utils.utils import load_numpy_array_data, evaluate_models
from loandefault.utils.ml_utils.metric.classification_metric import get_classification_score
from loandefault.utils.ml_utils.model.estimator import LoanModel


from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

import mlflow


import dagshub
dagshub.init(repo_owner='AnupamKNN', repo_name='LoanDefault', mlflow=True)



class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_train_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise LoanDefaultException(e, sys)
        
    def track_mlflow(self, best_model, classification_metric):
        with mlflow.start_run():
            m_f1_score = classification_metric.f1_score
            m_precision_score = classification_metric.precision_score
            m_recall_score = classification_metric.recall_score
            m_accuracy_score = classification_metric.accuracy_score

            mlflow.log_metric("Accuracy", m_accuracy_score)
            mlflow.log_metric("f1 score", m_f1_score)
            mlflow.log_metric("precision", m_precision_score)
            mlflow.log_metric("Recall", m_recall_score)
            mlflow.sklearn.log_model(best_model, "model")
        

    def train_model(self, x_train, y_train, x_test, y_test) -> LoanModel:
        try:
            models = {
                "LogisticRegression": LogisticRegression(class_weight='balanced'),
                "KNeighbors": KNeighborsClassifier(),
                "RandomForest": RandomForestClassifier(class_weight='balanced'),
                "AdaBoost": AdaBoostClassifier(),
                "GradientBoosting": GradientBoostingClassifier(),
                "XGBoost": XGBClassifier(eval_metric='logloss'),
                "SVC": SVC(probability=True, class_weight='balanced'),
                "DecisionTree": DecisionTreeClassifier(class_weight='balanced')
            }

            param_grid = {
                "LogisticRegression": [
                    {
                        'solver': ['liblinear'],
                        'penalty': ['l1', 'l2'],
                        'C': [0.01, 0.1, 1],
                        'max_iter': [100]
                    },
                    {
                        'solver': ['lbfgs'],
                        'penalty': ['l2'],
                        'C': [0.01, 0.1, 1],
                        'max_iter': [100]
                    },
                    {
                        'solver': ['saga'],
                        'penalty': ['elasticnet'],
                        'C': [0.01, 0.1],
                        'max_iter': [100],
                        'l1_ratio': [0.0, 0.5]
                    }
                ],

                "KNeighbors": {
                    'n_neighbors': [3, 5],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan']
                },

                "RandomForest": {
                    'n_estimators': [100],
                    'criterion': ['gini'],
                    'max_depth': [3, 5],
                    'min_samples_split': [5],
                    'min_samples_leaf': [2]
                },

                "AdaBoost": {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.01, 0.1]
                },

                "GradientBoosting": {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.01, 0.1],
                    'max_depth': [3, 5],
                    'subsample': [0.8]
                },

                "XGBoost": {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.01, 0.1],
                    'max_depth': [3, 5],
                    'subsample': [0.8],
                    'colsample_bytree': [0.8]
                },

                "SVC": [
                    {
                        'kernel': ['linear'],
                        'C': [0.1, 1]
                    },
                    {
                        'kernel': ['rbf'],
                        'C': [0.1, 1],
                        'gamma': ['scale']
                    }
                ],

                "DecisionTree": {
                    'criterion': ['gini'],
                    'max_depth': [3, 5],
                    'min_samples_split': [5],
                    'min_samples_leaf': [2]
                }
            }

            model_report: dict = evaluate_models(X_train=x_train, y_train=y_train, X_test=x_test, y_test=y_test,
                                                models=models, param=param_grid)

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            y_train_pred = best_model.predict(x_train)
            regression_train_metric = get_classification_score(y_true=y_train, y_pred=y_train_pred)
            self.track_mlflow(best_model, regression_train_metric)

            y_test_pred = best_model.predict(x_test)
            regression_test_metric = get_classification_score(y_true=y_test, y_pred=y_test_pred)
            self.track_mlflow(best_model, regression_test_metric)

            preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
            model_dir_path = os.path.dirname(self.model_train_config.trained_model_file_path)
            os.makedirs(model_dir_path, exist_ok=True)

            Loan_Model = LoanModel(preprocessor=preprocessor, model=best_model)
            save_object(self.model_train_config.trained_model_file_path, obj=Loan_Model)
            save_object("final_model/model.pkl", best_model)

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_train_config.trained_model_file_path,
                train_metric_artifact=regression_train_metric,
                test_metric_artifact=regression_test_metric
            )

            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact

        except Exception as e:
            raise LoanDefaultException(e, sys)




    def initate_model_trainer(self)-> ModelTrainerArtifact:
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            # loading train array and test array
            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1]
            )

            model_trainer_artifact = self.train_model(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
            return model_trainer_artifact


        except Exception as e:
            raise LoanDefaultException(e, sys)