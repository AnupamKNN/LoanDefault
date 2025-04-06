from loandefault.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from loandefault.entity.config_entity import DataValidationConfig
from loandefault.exception.exception import LoanDefaultException
from loandefault.logging.logger import logging
from loandefault.constant.training_pipeline import SCHEMA_FILE_PATH
from scipy.stats import ks_2samp
import pandas as pd
import sys, os
from loandefault.utils.main_utils.utils import read_yaml_file, write_yaml_file



class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, 
                 data_validation_config: DataValidationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise LoanDefaultException(e,  sys)
    
    @staticmethod
    def read_data(file_path)-> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise LoanDefaultException(e, sys)
        

    def validate_number_of_columns(self, dataframe: pd.DataFrame)->bool:
        try:
            number_of_columns = len(self._schema_config)
            logging.info(f"Required number of columns: {number_of_columns}")
            logging.info(f"Dataframe has number of columns: {len(dataframe.columns)}")
            if len(dataframe.columns) == number_of_columns:
                return True
            return False
        except Exception as e:
            raise LoanDefaultException(e, sys)
        

    def detect_dataset_drift(self, base_df, current_df, thershold = 0.05)-> bool:
        try:
            status = True
            report = {}
            for column in base_df.columns:
                d1 = base_df[column]
                d2 = current_df[column]
                is_same_dist = ks_2samp(d1, d2)
                if thershold <= is_same_dist.pvalue:
                    is_found = False
                else:
                    is_found = True
                    status = False
                report.update({column:{
                    "p_value": float(is_same_dist.pvalue),
                    "drift_status": is_found
                }})
            drift_report_file_path = self.data_validation_config.drift_report_file_path

            # Create directory
            dir_path = os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path, exist_ok=True)
            write_yaml_file(file_path = drift_report_file_path, content = report)
        except Exception as e:
            raise LoanDefaultException(e, sys)
        
    def is_numeric_column(self, dataframe: pd.DataFrame, column_name: str)->bool:
        try:
            return dataframe[column_name].dtypes in ["int64", "float64"]
        except Exception as e:
            raise LoanDefaultException(e, sys)
        
        

    def initiate_data_validation(self)->DataValidationArtifact:
        try:
            train_file_path = self.data_ingestion_artifact.trained_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            # read the data from train and test file
            train_dataframe = DataValidation.read_data(train_file_path)
            test_dataframe = DataValidation.read_data(test_file_path)

            # validate number of columns

            status = self.validate_number_of_columns(dataframe = train_dataframe)
            if not status:
                error_message = f"Train dataframe does not contain all columns.\n"
            status = self.validate_number_of_columns(dataframe = test_dataframe)
            if not status:
                error_message = f"Test dataframe does not contain all columns.\n"

            # Check if train and test dataset has numeric columns
            error_message = []
            if not self.is_numeric_column(dataframe= train_dataframe, column_name= train_dataframe.columns[0]):
                error_message.append (f"Train dataframe does not have numeric columns.\n")

            if not self.is_numeric_column(dataframe= test_dataframe, column_name= test_dataframe.columns[0]):
                error_message.append(f"Test dataframe does not have numeric columns.\n")

            if len(error_message) > 0:
                raise Exception("\n".join(error_message))
            else:
                logging.info("Numerical columns existance validation completed successfully. No errors found.")
            

            # Let's check datadrift
            status = self.detect_dataset_drift(base_df = train_dataframe, current_df = test_dataframe)
            dir_path = os.path.dirname(self.data_validation_config.valid_train_file_path)
            os.makedirs(dir_path, exist_ok=True)

            train_dataframe.to_csv(
                self.data_validation_config.valid_train_file_path, index = False, header = True
            )

            test_dataframe.to_csv(
                self.data_validation_config.valid_test_file_path, index = False, header= True
            )

            data_validation_artifact = DataValidationArtifact(
                validation_status = status,
                valid_train_file_path= self.data_ingestion_artifact.trained_file_path,
                valid_test_file_path= self.data_ingestion_artifact.test_file_path,
                invalid_train_file_path= None,
                invalid_test_file_path= None,
                drift_report_file_path= self.data_validation_config.drift_report_file_path
            )

            return data_validation_artifact

        except Exception as e:
            raise LoanDefaultException(e, sys)