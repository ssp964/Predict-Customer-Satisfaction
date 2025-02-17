from PredictCustomerSatisfaction.logging import logger
from PredictCustomerSatisfaction.pipeline.training_pipleine import trainPipeline

if __name__ == "__main__":
    # Run the pipeline
    try:
        logger.info("Running the training pipeline.")
        trainPipeline(
            data_path="./data/olist_customers_dataset.csv",
            model_name="LinearRegression",
        )
        logger.info("Training completed successfully.")
    except Exception as e:
        logger.error(f"Error: {e}")
        raise e
