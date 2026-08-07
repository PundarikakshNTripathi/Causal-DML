#!/bin/bash
echo "Starting Causal-DML Streamlit Dashboard..."
uv run streamlit run src/app/dashboard.py --server.port 8501 --server.headless true
