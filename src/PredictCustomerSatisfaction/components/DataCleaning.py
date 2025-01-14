from PredictCustomerSatisfaction.logging import logger
from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class DataStrategy(ABC):
    """
    An abstract base class for data strategies.
    """

    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        """
        Abstract method to handle the data.
        """
        pass


class DatePreProcessStrategy(DataStrategy):
    """
    Strategy to preprocess data with date columns.
    """

    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess date
        """
        try:
            data = data.drop(
                [
                    "order_approved_at",
                    "order_delivered_carrier_date",
                    "order_delivered_customer_date",
                    "order_estimated_delivery_date",
                    "order_purchase_timestamp",
                ],
                axis=1,
            )
            data["product_weight_g"].fillna(
                data["product_weight_g"].median(), inplace=True
            )
            data["product_length_cm"].fillna(
                data["product_length_cm"].median(), inplace=True
            )
            data["product_height_cm"].fillna(
                data["product_height_cm"].median(), inplace=True
            )
            data["product_width_cm"].fillna(
                data["product_width_cm"].median(), inplace=True
            )
            # writes "No review" in review_comment_message column for null values
            data["review_comment_message"].fillna("No review", inplace=True)

            data = data.select_dtypes(
                include=[np.number]
            )  # filters and keep only the numerical columns.
            cols_to_drop = ["customer_zip_code_prefix", "order_item_id"]
            data = data.drop(cols_to_drop, axis=1)
            logger.info("Data pre-processing finished successfully")
            return data
        except Exception as e:
            logger.error("Error in data pre-processing: ", e)
            raise e


class DataDivideStrategy(DataStrategy):
    """
    Data dividing strategy which divides the data into train and test data.
    """

    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        """
        Divides the data into train and test data.
        """
        try:
            X = data.drop("review_score", axis=1)
            y = data["review_score"]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            logger.info("Dataset divided into train and test data successfully")
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logger.error("Error in Data Division: ", e)
            raise e


class DataCleaning:
    """
    Data cleaning class which preprocesses the data and divides it into train and test data.
    """

    def __init__(self, data: pd.DataFrame, strategy: DataStrategy) -> None:
        """Initializes the DataCleaning class with a specific strategy."""
        self.df = data
        self.strategy = strategy

    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        """Handle data based on the provided strategy"""
        return self.strategy.handle_data(self.df)
