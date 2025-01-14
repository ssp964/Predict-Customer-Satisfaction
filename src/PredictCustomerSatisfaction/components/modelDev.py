from PredictCustomerSatisfaction.logging import logger
from abc import ABC, abstractmethod

from sklearn.linear_model import LinearRegression


class Model(ABC):
    """
    An abstract base class for models.
    """

    @abstractmethod
    def train(self, X_train, y_train):
        """
        Trainsthe model
        Args:
        X_train: Training data
        y_train: Training labels

        Returns: None
        """
        pass


class LinearRegressionModel(Model):
    """
    Linear Regression model implementation.
    """

    def train(self, X_train, y_train, **kwargs):
        """
        Train the Linear Regression model
        """

        try:
            logger.info("Linear Regression model training started")
            reg = LinearRegression(**kwargs)
            reg.fit(X_train, y_train)
            logger.info("Linear Regression model training completed")
            return reg
        except Exception as e:
            logger.error(f"Error in Linear Regression model training: {e}")
            raise e
