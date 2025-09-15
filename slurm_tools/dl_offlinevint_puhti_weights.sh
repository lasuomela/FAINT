# bin/bash

# Get the run id from arguments
run_id=$1
echo "Downloading weights for run id: ${run_id}"

# Create the destination directory if it doesn't exist
mkdir -p faint/deployment/src/faint_deployment/model_weights/${run_id}/
scp suomelal@puhti.csc.fi:/scratch/project_2010179/Theiavint_Offline/train/logs/DepthGoals/${run_id}/*.pt faint/deployment/src/faint_deployment/model_weights/${run_id}/