# Causal-DML: Causal Inference Engine for OTT Churn

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![DuckDB](https://img.shields.io/badge/Data-DuckDB-yellow)
![DoWhy](https://img.shields.io/badge/Causal-DoWhy-green)
![EconML](https://img.shields.io/badge/ML-EconML-orange)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)

Causal-DML is a causal inference pipeline that calculates the Conditional Average Treatment Effect (CATE) of retention interventions on OTT streaming platforms. It uses Structural Causal Models (SCMs) and Double Machine Learning (DML) to isolate confounding variables in observational user data, determining the causal impact of targeted actions (e.g., push notifications, discounts) on user churn.

## Core Architecture
*   **Data Processing**: Uses DuckDB and Apache Arrow to process 30GB+ of raw observational streaming logs (KKBox dataset) out-of-core.
*   **Causal Modeling**: Uses DoWhy to define the structural causal graph and EconML (Double Machine Learning) to estimate the CATE across user segments.
*   **MLOps Infrastructure**: Tracks datasets via DVC and logs model artifacts, parameters, and Average Treatment Effect (ATE) metrics via MLflow.
*   **Deployment**: Serves inference via a FastAPI REST backend, queried by a Streamlit dashboard for counterfactual simulations.

## Quick Start

This project uses `uv` for dependency management and adheres strictly to Git Flow.

### Prerequisites
*   `uv` installed.
*   Kaggle API access token configured (for dataset retrieval).

### Initialization
```bash
git clone https://github.com/YourUsername/Causal-DML.git
cd Causal-DML

uv sync

dvc pull

./scripts/run_api.sh

./scripts/run_dashboard.sh
```

## Pipeline Status
*   **Phase 1 (Data Pipeline):** Complete. Implemented out-of-core DuckDB aggregation and Kaggle dataset ingestion. Dataset tracked securely via DVC.
*   **Phase 2 (Causal Modeling):** Complete. Deployed DoWhy Structural Causal Model and EconML LinearDML CATE estimator. Model artifacts and hyperparameters tracked via MLflow and DVC.
*   **Phase 3 (FastAPI Backend):** Pending.
*   **Phase 4 (Streamlit Dashboard):** Pending.

## License
MIT License.
