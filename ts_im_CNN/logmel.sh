#!/bin/bash

#SBATCH --mail-user=phamanh@tnt.uni-hannover.de
#SBATCH --mail-type=ALL             
#SBATCH --job-name=cv1d_lms     
#SBATCH --output=result/lms/LMS-%j.txt   
#SBATCH --time=1-0             
#SBATCH --partition=gpu_normal_stud                                     
#SBATCH --cpus-per-task=16        
#SBATCH --mem=120G                  
#SBATCH --gres=gpu:1
# #SBATCH --array=0-0

source /home/phamanh/anaconda3/bin/activate /home/phamanh/anaconda3/envs/khoa

start=$(date +%s.%N)
echo "Start Time : $(date -d @$start)"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Partition: ${SLURM_JOB_PARTITION}"
echo "CPU cores: ${SLURM_JOB_CPUS_PER_NODE}"
echo "Memory: ${SLURM_MEM_PER_NODE}" 
echo "GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "Python environment: $(conda info --envs | grep '*' | sed -e 's/^[ \t*]*//')"
echo "Current work directory: ${PWD}"
echo 

python3 lms_cv.py "$@"

echo

end=$(date +%s.%N)
duration=$(echo "$end - $start" | bc)

echo "End Time   : $(date -d @$end)"
echo "Duration   : $duration seconds"
