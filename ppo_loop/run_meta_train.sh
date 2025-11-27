#!/bin/bash
# Run script for meta_train.py with correct Python path setup
# This script ensures the rl4co package can be imported from the parent directory

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Add the project root to PYTHONPATH so rl4co can be imported
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Run meta_train.py with all arguments passed to this script
cd "$SCRIPT_DIR"
python3 meta_train.py "$@"
