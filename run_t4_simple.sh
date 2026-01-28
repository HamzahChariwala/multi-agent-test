#!/bin/bash
# Launch T4 cluster with simple architecture (HTTP only on rank 0)

cd "$(dirname "$0")"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

python3 serving/t4_cluster/simple_launcher.py

