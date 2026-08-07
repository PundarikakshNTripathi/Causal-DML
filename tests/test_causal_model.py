import pytest
import numpy as np
import joblib
import os

def test_cate_prediction():
    model_path = 'models/causal_model.pkl'
    assert os.path.exists(model_path), f"Model file {model_path} does not exist."
    
    model = joblib.load(model_path)
    
    # Synthetic input tensor representing 4 confounders
    # Features: total_active_days, var_daily_listening_time, total_listening_time, avg_num_100
    X_synthetic = np.array([
        [10.0, 500.0, 10000.0, 5.0],
        [20.0, 200.0, 50000.0, 15.0],
        [30.0, 100.0, 80000.0, 25.0],
        [5.0,  800.0, 2000.0,  2.0],
        [15.0, 300.0, 30000.0, 10.0]
    ])
    
    cate = model.effect(X_synthetic)
    
    assert cate is not None
    assert len(cate) == 5
    assert isinstance(cate, np.ndarray)
    assert not np.isnan(cate).any()
