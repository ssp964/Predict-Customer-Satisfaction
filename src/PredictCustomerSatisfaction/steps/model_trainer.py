from PredictCustomerSatisfaction.logging import logger

from PredictCustomerSatisfaction.components.modelDev import LinearRegressionModel
from sklearn.base import RegressorMixin

import pandas as pd
from zenml import step


@step
def trainModel(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    modelName: str,
) -> RegressorMixin:
    """
    Train a model on the data.

    Args:
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.DataFrame,
        y_test: pd.DataFrame,
    """
    model = None

    if modelName == "LinearRegression":
        model = LinearRegressionModel()
        trained_model = model.train(X_train, y_train)
        return trained_model
    else:
        logger.error(f"Unsupported model: %s", modelName)
        raise ValueError(f"Unsupported model: %s", modelName)
