#!/bin/bash

#################################################
# Script to run all the experiments in sequence #
#################################################


# Function to kill all Python processes for a specific user
function cleanup {
    echo "Stopping all Python processes for user $(whoami)..."
    pkill -u $(whoami) python
}

# Trap SIGINT (Ctrl+C) signal
trap cleanup SIGINT

# set env
source .venv/bin/activate

# set pythonpath
export PYTHONPATH=$(pwd)



python src/experiments/beltrami/run_lossplots.py  -m random_seed=42 optimization=ls

