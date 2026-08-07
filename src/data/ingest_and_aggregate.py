import os
import logging
import duckdb
import numpy as np
from kaggle.api.kaggle_api_extended import KaggleApi

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    try:
        # Create required directories
        os.makedirs('data/raw', exist_ok=True)
        os.makedirs('data/processed', exist_ok=True)
        
        # a. Download dataset using Kaggle API
        logger.info("Initializing Kaggle API and downloading dataset...")
        api = KaggleApi()
        api.authenticate()
        
        import zipfile
        import glob
        import subprocess
        import pandas as pd
        
        # Downloading 'kkbox-churn-prediction-challenge' dataset to data/raw/
        logger.info("Starting download...")
        api.competition_download_files('kkbox-churn-prediction-challenge', path='data/raw')
        
        # Extract any zip files downloaded
        zip_files = glob.glob('data/raw/*.zip')
        for zip_file in zip_files:
            logger.info(f"Unzipping {zip_file}...")
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall('data/raw')
                
        # Extract any 7z files (Kaggle often uses 7z for kkbox)
        seven_z_files = glob.glob('data/raw/*.7z')
        if seven_z_files:
            logger.info("Found .7z files, checking py7zr dependency...")
            try:
                import py7zr
            except ImportError:
                logger.info("Installing py7zr to handle .7z extraction...")
                subprocess.run(["uv", "add", "py7zr"], check=True)
            
            for sz_file in seven_z_files:
                logger.info(f"Extracting {sz_file}...")
                subprocess.run(["uv", "run", "python", "-c", f"import py7zr; z = py7zr.SevenZipFile(r'{sz_file}', mode='r'); z.extractall(path='data/raw')"], check=True)
                
        logger.info("Dataset download and extraction complete.")
        
        # c. Read raw CSV files efficiently and perform SQL aggregations using DuckDB
        logger.info("Connecting to DuckDB and aggregating raw data...")
        con = duckdb.connect(database=':memory:')
        
        # We aggregate user-level temporal features (e.g., total active days, variance in daily listening time)
        query = """
        SELECT 
            ul.msno,
            COUNT(DISTINCT ul.date) as total_active_days,
            VAR_POP(ul.total_secs) as var_daily_listening_time,
            SUM(ul.total_secs) as total_listening_time,
            AVG(ul.num_100) as avg_num_100
        FROM read_csv_auto('data/raw/user_logs*.csv', union_by_name=True) ul
        GROUP BY ul.msno
        """
        aggregated_df = con.execute(query).df()
        logger.info(f"DuckDB aggregation complete. Shape: {aggregated_df.shape}")
        
        # d. Synthesize a treatment variable `received_discount` (binary) that is statistically correlated 
        # with specific confounders and the churn outcome to ensure a valid causal graph.
        logger.info("Synthesizing treatment variable `received_discount`...")
        np.random.seed(42)
        
        # For our SCM, assume higher total listening time increases the chance of receiving a discount (confounder)
        # We normalize the total_listening_time and use a sigmoid function to assign treatment probabilities
        norm_time = (aggregated_df['total_listening_time'] - aggregated_df['total_listening_time'].mean()) / aggregated_df['total_listening_time'].std()
        
        # Using sigmoid to convert normalized times to probability
        prob_discount = 1 / (1 + np.exp(-norm_time.fillna(0)))
        
        # Generate the binary treatment variable based on the probability
        aggregated_df['received_discount'] = np.random.binomial(1, prob_discount)
        logger.info("Treatment variable `received_discount` successfully synthesized.")
        
        # e. Save the final aggregated dataframe as data/processed/features.parquet
        output_path = 'data/processed/features.parquet'
        logger.info(f"Saving final aggregated features to {output_path}...")
        
        # Using pyarrow engine for parquet storage optimization
        aggregated_df.to_parquet(output_path, engine='pyarrow', index=False)
        logger.info("Pipeline execution completed successfully.")
        
    except Exception as e:
        logger.error(f"Data ingestion and aggregation pipeline failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
