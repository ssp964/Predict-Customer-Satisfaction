from zenml import pipeline
from PredictCustomerSatisfaction.steps.data_ingestion import ingestData
from PredictCustomerSatisfaction.steps.data_cleaning import cleanData
from PredictCustomerSatisfaction.steps.model_trainer import trainModel
from PredictCustomerSatisfaction.steps.model_evaluate import evaluateModel
from PredictCustomerSatisfaction.logging import logger


@pipeline(enable_cache=False)
def trainPipeline(data_path: str):
    """
    A pipeline to train a model on customer satisfaction data.
    """
    logger.info("Pipeline started")
    df = ingestData(data_path)
    cleanData(df)
    trainModel(df)
    evaluateModel(df)
