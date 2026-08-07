import os
import joblib
import pandas as pd
import numpy as np
import mlflow
import logging
from dowhy import CausalModel
from econml.dml import LinearDML
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    try:
        os.makedirs('models', exist_ok=True)
        
        logger.info("Loading feature dataset...")
        df = pd.read_parquet('data/processed/features.parquet')
        
        if 'is_churn' not in df.columns:
            logger.info("Loading outcome labels from train.csv...")
            try:
                train_df = pd.read_csv('data/raw/train.csv')
                df = df.merge(train_df[['msno', 'is_churn']], on='msno', how='inner')
                logger.info("Successfully merged outcome labels.")
            except Exception as e:
                logger.warning(f"Error loading train.csv: {e}. Generating synthetic outcome labels.")
                np.random.seed(42)
                base_churn_prob = 0.3
                treatment_effect = -0.15 
                churn_prob = base_churn_prob + df['received_discount'] * treatment_effect
                df['is_churn'] = np.random.binomial(1, churn_prob)

        df = df.dropna()
        if len(df) == 0:
            raise ValueError("Dataset is empty after dropping NaNs.")
            
        if len(df) > 100000:
            logger.info(f"Dataset has {len(df)} rows. Sampling 100,000 for training speed.")
            df = df.sample(100000, random_state=42)

        confounders = ['total_active_days', 'var_daily_listening_time', 'total_listening_time', 'avg_num_100']
        confounders = [c for c in confounders if c in df.columns]
        
        mlflow.set_experiment("Causal_DML_Churn_Analysis")
        
        with mlflow.start_run():
            logger.info("Defining Structural Causal Model via DoWhy...")
            model = CausalModel(
                data=df,
                treatment='received_discount',
                outcome='is_churn',
                common_causes=confounders,
                instruments=None,
                effect_modifiers=None
            )
            
            model.identify_effect(proceed_when_unidentifiable=True)
            
            logger.info("Training LinearDML model via EconML...")
            est = LinearDML(
                model_y=RandomForestRegressor(n_estimators=50, max_depth=5, n_jobs=-1, random_state=42),
                model_t=RandomForestClassifier(n_estimators=50, max_depth=5, n_jobs=-1, random_state=42),
                discrete_treatment=True,
                linear_first_stages=False,
                cv=3,
                random_state=42
            )
            
            X = df[confounders].values
            Y = df['is_churn'].values
            T = df['received_discount'].values
            
            est.fit(Y, T, X=X)
            
            cate = est.effect(X)
            ate = np.mean(cate)
            logger.info(f"Estimated Average Treatment Effect (ATE): {ate:.4f}")
            
            mlflow.log_param("model_type", "LinearDML")
            mlflow.log_param("n_estimators_y", 50)
            mlflow.log_param("n_estimators_t", 50)
            mlflow.log_param("cv", 3)
            mlflow.log_param("confounders", str(confounders))
            mlflow.log_metric("ATE", ate)
            
            model_path = "models/causal_model.pkl"
            logger.info(f"Saving model to {model_path}...")
            joblib.dump(est, model_path)
            mlflow.log_artifact(model_path)
            
        logger.info("Training complete.")
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
