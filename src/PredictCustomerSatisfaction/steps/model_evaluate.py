from PredictCustomerSatisfaction.logging import logger
import pandas as pd
from zenml import step


@step
def evaluateModel(df: pd.DataFrame) -> None:
    """
    Evaluate the model's performance.

    Args:
        df (pd.DataFrame): The data used for evaluation.
    """
    logger.info("Model evaluation started")

    pass
