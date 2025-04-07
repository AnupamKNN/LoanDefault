import os
import sys

from loandefault.exception.exception import LoanDefaultException
from loandefault.logging.logger import logging

from loandefault.components.data_ingestion import DataIngestion
from loandefault.components.data_validation import DataValidation
from loandefault.components.data_transformation import DataTransformation
from loandefault.components.model_trainer import ModelTrainer

from loandefault.entity.config_entity import (
    TrainingPipelineConfig,
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig
    
    
)

from loandefault.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact
)

import sys

DAGSHUB_USERNAME = os.getenv("DAGSHUB_USERNAME")
DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN")

if DAGSHUB_USERNAME and DAGSHUB_TOKEN:
    os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USERNAME
    os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN
    os.environ["MLFLOW_TRACKING_URI"] = f"https://dagshub.com/{DAGSHUB_USERNAME}/LoanDefault.mlflow"
else:
    print("❗❗❗ AUTHORIZATION REQUIRED ❗❗❗")
    sys.exit(1)

class TrainingPipeline:
    def __init__(self):
        self.training_pipeline_config = TrainingPipelineConfig()

    def start_data_ingestion(self):
        try:
            self.data_ingestion_config = DataIngestionConfig(training_pipeline_config=self.training_pipeline_config)
            logging.info("Start data ingestion")
            data_ingestion = DataIngestion(data_ingestion_config= self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initia_data_ingestion()
            logging.info(f"Data Ingestion completed and srtifact: {data_ingestion_artifact}")
            return data_ingestion_artifact

        except Exception as e:
            raise LoanDefaultException(e, sys)
        
    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact):
        try:
            data_validation_config = DataValidationConfig(training_pipeline_config= self.training_pipeline_config)
            data_validation = DataValidation(data_ingegestion_artifact = data_ingestion_artifact, data_validation_config=data_validation_config)
            logging.info("Initiated the data validation")
            data_validation_artifact = data_validation.initiate_data_validation()
            return data_validation_artifact
        except Exception as e:
            raise LoanDefaultException(e, sys)
        
    def sart_data_transformation(self, data_validation_artifact: DataValidationArtifact):
        try:
            data_transformation_config = DataTransformationConfig(training_pipeline_config=self.training_pipeline_config)
            data_transformation = DataTransformation(data_validation_artifact= data_validation_artifact,
                                                    data_transformation_config= data_transformation_config)
            
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            return data_transformation_artifact
        except Exception as e:
            raise LoanDefaultException(e, sys)
        
    def start_model_trainer(self, data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config:ModelTrainerConfig = ModelTrainerConfig(training_pipeline_config=self.training_pipeline_config)
            model_trainer = ModelTrainer(
                data_transformation_artifact= data_transformation_artifact,
                model_trainer_config = self.model_trainer_config
            )

            model_trainer_artifact = model_trainer.initate_model_trainer()
            return model_trainer_artifact
        except Exception as e:
            raise LoanDefaultException(e, sys)


    def run_pipeline(self):
        try:
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
            data_transformation_artifact = self. sart_data_transformation(data_validation_artifact= data_validation_artifact)
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact= data_transformation_artifact)
            return model_trainer_artifact
        except Exception as e:
            raise LoanDefaultException(e, sys)