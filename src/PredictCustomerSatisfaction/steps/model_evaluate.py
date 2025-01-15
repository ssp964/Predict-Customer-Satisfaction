from PredictCustomerSatisfaction.logging import logger
from PredictCustomerSatisfaction.components.modelEval import MSE, RMSE, R2Score

import pandas as pd
import mlflow
from typing import Tuple
from typing_extensions import Annotated
from sklearn.base import RegressorMixin
from zenml import step
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker


@step(experiment_tracker=experiment_tracker.name)
def evaluateModel(
    model: RegressorMixin, X_test: pd.DataFrame, y_test: pd.DataFrame
) -> Tuple[Annotated[float, "RMSEscore"], Annotated[float, "R2score"]]:
    """
    Evaluate the model's performance.

    Args:
        df (pd.DataFrame): The data used for evaluation.
    """
    try:
        logger.info("Model evaluation started")

        prediction = model.predict(X_test)
        MSEscoreFn = MSE()
        RMSEscoreFn = RMSE()
        R2scoreFn = R2Score()

        MSEscore = MSEscoreFn.calculate_score(y_test, prediction)
        mlflow.log_metric("mse", MSEscore)

        RMSEscore = RMSEscoreFn.calculate_score(y_test, prediction)
        mlflow.log_metric("rmse", RMSEscore)

        R2score = R2scoreFn.calculate_score(y_test, prediction)
        mlflow.log_metric("r2score", R2score)

        logger.info(f"Evaluation scores calculated successfully")

        return RMSEscore, R2score
    except Exception as e:
        logger.error(f"Error in model evaluation: {e}")
        raise e
