#!/bin/bash

# Set up virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r api/requirements.txt
pip install -r webapp/requirements.txt

# Run initial training (if needed)
python model/train.py
