#!/bin/bash
#SBATCH --job-name=deepPSMA
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --mem=32G

# Modules en env
module load GCCcore/10.3.0
module load Python/3.9.5-GCCcore-10.3.0
module load CUDA/11.4.1
pip install --upgrade pip
pip install torchinfo

echo "[INFO] Modules geladen en packages geïnstalleerd."

# Dataset en cache klaarmaken op TMPDIR
mkdir -p $TMPDIR/cache_dir
cp -r /scratch/p321870/deepPSMAchallenge/code/cache_dir_3mm/* $TMPDIR/cache_dir/

echo "[INFO] Dataset en cache gekopieerd naar TMPDIR."

# Start training
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" python /home3/p321870/jobs/DeepPSMA/main.py

echo "[INFO] Training voltooid."
