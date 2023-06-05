#!/bin/bash

# Initialize variables
run_path=""

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --run_path=*) run_path="${1#*=}" ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Check if run_path is provided
if [[ -z "$run_path" ]]; then
    echo "The --run_path parameter is required."
    exit 1
fi


echo "Connecting to wandb ..."


# Run Python code
python << END
# Python code goes here
import wandb
import pandas as pd

run_path = '$run_path'
export_path = run_path.replace('/', '_')
run = wandb.Api(timeout=30).run(run_path)
pd.DataFrame(run.history()).to_csv(f'{export_path}.csv')
print(f'Log exported to {export_path}.csv')
END

# Check the exit status of the Python code execution
if [[ $? -eq 0 ]]; then
    echo "Log exported successfully"
else
    echo "An error has occurred. The logs were not downloaded"
fi