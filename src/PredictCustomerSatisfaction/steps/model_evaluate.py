from PredictCustomerSatisfaction.logging import logger
from PredictCustomerSatisfaction.components.modelEval import MSE, RMSE, R2Score

import pandas as pd
from typing import Tuple
from typing_extensions import Annotated
from sklearn.base import RegressorMixin
from zenml import step


@step
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
        RMSEscore = RMSEscoreFn.calculate_score(y_test, prediction)
        R2score = R2scoreFn.calculate_score(y_test, prediction)

        logger.info(f"Evaluation scores calculated successfully")

        return RMSEscore, R2score
    except Exception as e:
        logger.error(f"Error in model evaluation: {e}")
        raise e
