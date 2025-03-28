#!/bin/bash

# Run the Streamlit app with remote access configuration
streamlit run embedding_retrieval_app.py --server.enableCORS=false --server.enableXsrfProtection=false --server.headless=true --browser.serverAddress="0.0.0.0" --server.port=8501
