from loandefault.components.data_ingestion import DataIngestion
from loandefault.exception.exception import LoanDefaultException
from loandefault.logging.logger import logging
from loandefault.entity.config_entity import DataIngestionConfig
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

    except Exception as e:
        raise LoanDefaultException(e, sys)