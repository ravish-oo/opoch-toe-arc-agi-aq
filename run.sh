#!/bin/bash
# Wrapper script to run pipeline with venv
source venv/bin/activate
python run.py "$@"
