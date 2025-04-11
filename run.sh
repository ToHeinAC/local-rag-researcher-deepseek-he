#!/bin/bash

# Activate the virtual environment
source ../lrrd-venv/bin/activate

# Run streamlit with settings for remote access and disable file watcher to prevent PyTorch errors
streamlit run app.py --server.enableCORS=false --server.enableXsrfProtection=false --server.port=8501 --server.fileWatcherType none
