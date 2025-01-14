from zenml import pipeline
from PredictCustomerSatisfaction.steps.data_ingestion import ingestData
from PredictCustomerSatisfaction.steps.data_cleaning import cleanData
from PredictCustomerSatisfaction.steps.model_trainer import trainModel
from PredictCustomerSatisfaction.steps.model_evaluate import evaluateModel
from PredictCustomerSatisfaction.logging import logger


@pipeline(enable_cache=False)
def trainPipeline(data_path: str, model_name: str):
    """
    A pipeline to train a model on customer satisfaction data.
    """
    logger.info("Pipeline started")
    df = ingestData(data_path)
    X_train, X_test, y_train, y_test = cleanData(df)
    model = trainModel(X_train, X_test, y_train, y_test, model_name)
    r2_score, rmse = evaluateModel(model, X_test, y_test)
