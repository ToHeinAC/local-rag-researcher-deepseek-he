#!/bin/bash

# Change to the project root directory
cd "$(dirname "$0")/.." || exit

# Run the Streamlit app
streamlit run utilities/chroma_db_inspector_app.py --server.port=8502 --server.address=0.0.0.0
