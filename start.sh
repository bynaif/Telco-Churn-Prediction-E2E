#!/bin/bash
uvicorn src.backend.main:app --host 0.0.0.0 --port 8001 &
streamlit run src/frontend/app.py --server.port 8000 --server.address 0.0.0.0
