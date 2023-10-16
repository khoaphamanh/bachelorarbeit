#!/bin/bash

# Submit the first job
job_ids=($(sbatch --parsable gramian.sh ))
echo "First job ID: ${job_ids[0]}"

# Loop through the remaining jobs
for i in {1..99}
do
    # Submit the job with dependency on the previous job, and store the job ID in the array
    job_ids+=($(sbatch --parsable --dependency=afterok:${job_ids[$i-1]} gramian.sh ))
    echo "Job ${i} ID: ${job_ids[$i]}"
done