from PredictCustomerSatisfaction.logging import logger

from PredictCustomerSatisfaction.components.DataCleaning import (
    DataCleaning,
    DataDivideStrategy,
    DatePreProcessStrategy,
)

import pandas as pd
from zenml import step
from typing_extensions import Annotated
from typing import Tuple


@step
def cleanData(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.DataFrame, "y_train"],
    Annotated[pd.DataFrame, "y_test"],
]:
    """
    Clean the data by applying date preprocessing and data division.

    Args:
        df (pd.DataFrame): The raw data to be cleaned.

    Returns:
        X_train: Training data
        X_test: Testing data
        y_train: Training labels
        y_test: Testing labels

    """
    try:
        process_strategy = DatePreProcessStrategy()
        data_cleaning = DataCleaning(df, process_strategy)
        processed_data = data_cleaning.handle_data()

        divide_strategy = DataDivideStrategy()
        final_data = DataCleaning(df, divide_strategy)
        X_train, X_test, y_train, y_test = final_data.handle_data()
        logger.info("Data cleaning started")
    except Exception as e:
        logger.error(f"Error in data cleaning: {e}")
        raise e
