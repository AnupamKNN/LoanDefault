from loandefault.components.data_ingestion import DataIngestion
from loandefault.components.data_validation import DataValidation
from loandefault.components.data_transformation import DataTransformation
from loandefault.components.model_trainer import ModelTrainer
from loandefault.exception.exception import LoanDefaultException
from loandefault.logging.logger import logging
from loandefault.entity.config_entity import DataIngestionConfig, DataValidationConfig, DataTransformationConfig,  ModelTrainerConfig
from loandefault.entity.config_entity import TrainingPipelineConfig
from loandefault.entity.config_entity import TrainingPipelineConfig

import sys
if __name__ == "__main__":
    try:
        training_pipeline_config = TrainingPipelineConfig()

        # Data Ingestion
        data_ingestion_config = DataIngestionConfig(training_pipeline_config=training_pipeline_config)
        data_ingestion = DataIngestion(data_ingestion_config)
        logging.info("Initiated data ingestion")
        data_ingestion_artifact = data_ingestion.initia_data_ingestion()
        logging.info("Data Ingestion completed")
        # print(data_ingestion_artifact)

        # Data Validation
        data_validation_config = DataValidationConfig(training_pipeline_config=training_pipeline_config)
        data_validation = DataValidation(data_ingestion_artifact, data_validation_config)
        logging.info("Initiated data validation")
        data_validation_artifact = data_validation.initiate_data_validation()
        logging.info("Data validation completed")
        # print(data_validation_artifact)

        # Data Transformation
        data_transformation_config = DataTransformationConfig(training_pipeline_config=training_pipeline_config)
        data_transformation = DataTransformation(data_validation_artifact, data_transformation_config)
        logging.info("Initiated data transformation")
        data_transformation_artifact = data_transformation.initiate_data_transformation()
        logging.info("Data transformation completed")

        # Model Training
        logging.info("Model Training started")
        model_trainer_config = ModelTrainerConfig(training_pipeline_config=training_pipeline_config)
        model_trainer = ModelTrainer(model_trainer_config=model_trainer_config, 
                                     data_transformation_artifact=data_transformation_artifact)
        model_trainer_artifact = model_trainer.initate_model_trainer()
        logging.info("Model Training completed successfully")


    except Exception as e:
        raise LoanDefaultException(e, sys)