import pandas as pd
from zenml import step
from PredictCustomerSatisfaction.logging import logger


class IngestData:
    """
    Ingesting the data from the datapath and returning the data as a Pandas DataFrame.
    """

    def __init__(self, data_path: str):
        """
        Args:
            data_path (str): The path to the CSV file.
        """
        self.data_path = data_path

    def get_data(self):
        """
        Ingest the data from the datapath and return the data as a Pandas DataFrame
        """
        logger.info("Ingesting data from {self.data_path}")
        return pd.read_csv(self.data_path)


@step
def ingestData(data_path: str) -> pd.DataFrame:
    """
    Ingest data from a CSV file and return a Pandas DataFrame.

    Args:
        data_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: The ingested data as a Pandas DataFrame.
    """
    try:
        ingest_data = IngestData(data_path)
        df = ingest_data.get_data()
        return df
    except Exception as e:
        logger.error(f"Error ingesting data: {e}")
        raise e
