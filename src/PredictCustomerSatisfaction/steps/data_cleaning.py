from PredictCustomerSatisfaction.logging import logger
import pandas as pd
from zenml import step


@step
def cleanData(df: pd.DataFrame) -> None:
    logger.info("Data cleaning started")
    pass
