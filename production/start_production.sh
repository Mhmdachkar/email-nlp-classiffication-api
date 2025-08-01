#!/bin/bash
echo "Starting Production Email NLP API..."
cd "$(dirname "$0")"
python3 production_api.py
