#!/bin/bash

# Activate the virtual environment
source ../lrrd-venv/bin/activate

# Run streamlit with settings for remote access
streamlit run app.py --server.enableCORS=false --server.enableXsrfProtection=false --server.port=8501
