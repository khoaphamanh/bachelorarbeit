#!/bin/bash

#SBATCH --mail-user=phamanh@tnt.uni-hannover.de # only <UserName>@tnt.uni-hannover.de is allowed as mail address
#SBATCH --mail-type=ALL             # Eine Mail wird bei Job-Start/Ende versendet
#SBATCH --job-name=fine_cv      # Name unter dem der Job in der Job-History gespeichert wird
#SBATCH --output=result/pann/PANN-%j.txt   # Logdatei für den merged STDOUT/STDERR output (%j wird durch slurm job-ID ersetzt)

#SBATCH --time=1-0             # Maximale Laufzeit des Jobs, bis Slurm diesen abbricht (HH:MM:SS)
#SBATCH --partition=gpu_normal_stud   # Partition auf der gerechnet werden soll. Ohne Angabe des Parameters wird auf der
                                    #   Default-Partition gerechnet. Es können mehrere angegeben werden, mit Komma getrennt.
#SBATCH --cpus-per-task=16       # Reservierung von 4 CPUs pro Rechenknoten
#SBATCH --mem=32G                   # Reservierung von 10GB RAM
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

python3 pann.py 

echo

end=$(date +%s.%N)
duration=$(echo "$end - $start" | bc)

echo "End Time   : $(date -d @$end)"
echo "Duration   : $duration seconds"
