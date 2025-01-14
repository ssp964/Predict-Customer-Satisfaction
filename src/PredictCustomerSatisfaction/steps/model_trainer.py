from PredictCustomerSatisfaction.logging import logger
import pandas as pd
from zenml import step


@step
def trainModel(df: pd.DataFrame) -> None:
    """
    Train a model on the data.

    Args:
        df (pd.DataFrame): The data to train the model on.
    """
    logger.info("Model training starteds")
    pass
