import os
import joblib
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Cache the loaded model in memory to prevent disk reads on every API request.
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the trained causal model during startup.
    model_path = "models/causal_model.pkl"
    if not os.path.exists(model_path):
        raise RuntimeError(f"Model file not found at {model_path}. Ensure Phase 2 is completed.")
    
    ml_models["causal_model"] = joblib.load(model_path)
    yield
    # Clean up state on shutdown.
    ml_models.clear()

app = FastAPI(
    title="Causal-DML Inference API",
    description="FastAPI backend for predicting Conditional Average Treatment Effect (CATE)",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS middleware to permit frontend communication.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class UserFeatures(BaseModel):
    total_active_days: float = Field(..., description="Total number of active days")
    var_daily_listening_time: float = Field(..., description="Variance in daily listening time")
    total_listening_time: float = Field(..., description="Total listening time in seconds")
    avg_num_100: float = Field(..., description="Average number of full tracks listened to")

class CATEResponse(BaseModel):
    cate: float
    message: str

@app.post("/predict_cate", response_model=CATEResponse)
async def predict_cate(features: UserFeatures):
    try:
        model = ml_models.get("causal_model")
        if not model:
            raise HTTPException(status_code=500, detail="Model is not loaded.")
        
        # Format input tensor for EconML prediction
        X_input = np.array([[
            features.total_active_days,
            features.var_daily_listening_time,
            features.total_listening_time,
            features.avg_num_100
        ]])
        
        # Predict CATE
        cate_array = model.effect(X_input)
        cate_value = float(cate_array[0])
        
        return CATEResponse(
            cate=cate_value,
            message="Successfully predicted Conditional Average Treatment Effect."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
