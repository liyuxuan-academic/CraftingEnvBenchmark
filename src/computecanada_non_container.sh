#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=16GB
#SBATCH --array=0-32

# Virtual Env
source ~/crafting_env/bin/activate
wandb offline
python ~/CraftingEnvBenchmark/src/craftbench/maskable_ppo.py
deactivate

