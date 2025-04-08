import sys
import os
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from loandefault.entity.config_entity import DataTransformationConfig
from loandefault.entity.artifact_entity import (
    DataValidationArtifact,
    DataTransformationArtifact
)

from loandefault.exception.exception import LoanDefaultException
from loandefault.logging.logger import logging
from loandefault.constant.training_pipeline import TARGET_COLUMN
from loandefault.utils.main_utils.utils import save_numpy_array_data, save_object
from loandefault.constant.training_pipeline import SCHEMA_FILE_PATH
from loandefault.utils.main_utils.utils import read_yaml_file


class DataTransformation:
    def __init__(self, data_validation_artifact: DataValidationArtifact, 
                 data_transformation_config: DataTransformationConfig):
        try:
            self.data_validation_artifact:DataValidationArtifact = data_validation_artifact
            self.data_transformation_config:DataTransformationConfig = data_transformation_config
        except Exception as e:
            raise LoanDefaultException(e, sys)
        
    @staticmethod
    def read_data(file_path)->pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise LoanDefaultException(e, sys)
        
    def get_data_transformer_object(self) -> Pipeline:
        """
        Returns:
            A Pipeline object that applies Simple Imputation, Standard Scaling, 
            and One-Hot Encoding (with first column dropped).
        """
        logging.info("Entered get_data_transformer_object method of Transformation class")
        
        try:
            # Load schema configuration
            self.schema_config = read_yaml_file(SCHEMA_FILE_PATH)

            # Identify numerical and categorical features
            num_features = [list(d.keys())[0] for d in self.schema_config["numerical_columns"]]
            cat_features = [list(d.keys())[0] for d in self.schema_config["categorical_columns"]]

            # Define numerical transformer pipeline with SimpleImputer
            numeric_transformer = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),  # Handle NaNs
                ("scaler", StandardScaler())
            ])

            # Define categorical transformer pipeline with SimpleImputer & OneHotEncoder
            categorical_transformer = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),  # Handle NaNs in categorical columns
                ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore"))
            ])

            # Combine transformers into a ColumnTransformer
            preprocessor = ColumnTransformer(transformers=[
                ("categorical", categorical_transformer, cat_features),
                ("numerical", numeric_transformer, num_features)
            ])

            # Wrap the ColumnTransformer in a pipeline
            transformation_pipeline = Pipeline(steps=[
                ("preprocessor", preprocessor)
            ])
            
            return transformation_pipeline

        except Exception as e:
            logging.error("Error in get_data_transformer_object method")
            raise LoanDefaultException(e, sys)


    def initiate_data_transformation(self) -> DataTransformationArtifact:
        logging.info("Entered initiate_data_transformation method of DataTransformation class")
        try:
            logging.info("Start data transformation")

            # Read training and testing data
            train_df = DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)

            # Extract input features and target features
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN].values.reshape(-1, 1)

            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN].values.reshape(-1, 1)

            # Load schema and extract expected columns
            self.schema_config = read_yaml_file(SCHEMA_FILE_PATH)
            expected_columns = [list(col.keys())[0] for col in self.schema_config["numerical_columns"] + self.schema_config["categorical_columns"]]

            # Handle and log any column mismatches
            missing_train = set(expected_columns) - set(input_feature_train_df.columns)
            extra_train = set(input_feature_train_df.columns) - set(expected_columns)
            if missing_train:
                logging.warning(f"Missing columns in training data: {missing_train}")
            if extra_train:
                logging.info(f"Extra columns in training data (will be dropped): {extra_train}")

            missing_test = set(expected_columns) - set(input_feature_test_df.columns)
            extra_test = set(input_feature_test_df.columns) - set(expected_columns)
            if missing_test:
                logging.warning(f"Missing columns in test data: {missing_test}")
            if extra_test:
                logging.info(f"Extra columns in test data (will be dropped): {extra_test}")

            # Drop extra columns to align with schema
            input_feature_train_df = input_feature_train_df[expected_columns]
            input_feature_test_df = input_feature_test_df[expected_columns]

            # Apply preprocessing
            preprocessor = self.get_data_transformer_object()
            preprocessor_object = preprocessor.fit(input_feature_train_df)

            transformed_input_train_feature = preprocessor_object.transform(input_feature_train_df)
            transformed_input_test_feature = preprocessor_object.transform(input_feature_test_df)

            # Convert to dense arrays if sparse
            if hasattr(transformed_input_train_feature, "toarray"):
                transformed_input_train_feature = transformed_input_train_feature.toarray()
            if hasattr(transformed_input_test_feature, "toarray"):
                transformed_input_test_feature = transformed_input_test_feature.toarray()

            # Diagnostic prints
            print("Shape of transformed_input_train_feature:", transformed_input_train_feature.shape)
            print("Shape of target_feature_train_df:", target_feature_train_df.shape)

            print("Shape of transformed_input_test_feature:", transformed_input_test_feature.shape)
            print("Shape of target_feature_test_df:", target_feature_test_df.shape)

            print(f"The type of transformed_input_train_feature is: {type(transformed_input_train_feature)}")
            print(f"The type of target_feature_train_df is: {type(target_feature_train_df)}") 

            print(f"The type of transformed_input_test_feature is: {type(transformed_input_test_feature)}")
            print(f"The type of target_feature_test_df is: {type(target_feature_test_df)}")

            # Check class distribution before balancing
            print("Class distribution before SMOTETomek:")
            unique, counts = np.unique(target_feature_train_df, return_counts=True)
            print(dict(zip(unique, counts)))

            # Apply SMOTE-Tomek for balancing classes
            from imblearn.combine import SMOTETomek
            smt = SMOTETomek(random_state=42)
            resampled_input_train_feature, resampled_target_train = smt.fit_resample(
                transformed_input_train_feature, target_feature_train_df.ravel()
            )

            # Check class distribution after balancing
            print("Class distribution after SMOTETomek:")
            unique, counts = np.unique(resampled_target_train, return_counts=True)
            print(dict(zip(unique, counts)))

            # Stack features and target column
            train_arr = np.c_[resampled_input_train_feature, resampled_target_train]
            test_arr = np.c_[transformed_input_test_feature, target_feature_test_df]

            # Save transformed numpy arrays
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)

            # Save the preprocessing object
            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor_object)
            save_object("final_model/preprocessor.pkl", preprocessor_object)

            # Prepare and return artifacts
            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )
            return data_transformation_artifact

        except Exception as e:
            raise LoanDefaultException(e, sys)


        except Exception as e:
            raise LoanDefaultException(e, sys)



