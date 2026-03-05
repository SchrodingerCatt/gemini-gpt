#!/bin/bash
# 1. vectoruli bazis momzadeba
echo "Step 1: Ingesting data"
python3.11 ingest_gemini.py

# 2. serveris gashveba
echo "Step 2: Starting API server"
python3.11 main.py
"
