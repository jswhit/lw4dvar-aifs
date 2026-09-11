#!/bin/bash
#SBATCH -J long_window_4dvar_aifsv2
#SBATCH -o long_window_4dvar_aifsv2_latent_sstreset.out
#SBATCH -e long_window_4dvar_aifsv2_latent_sstreset.err
#SBATCH --account=gpu-ai4wp
#SBATCH -t 08:00:00
#SBATCH --partition=u1-h100
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=96g
#SBATCH --qos=gpu
module load cuda
module load rdhpcs-conda
conda activate /scratch4/BMC/gsienkf/whitaker/conda/envs/aifs2
#ln -sfnT /scratch3/NCEPDEV/da/Jeffrey.Whitaker/psobs psobs
#ln -sfnT config.yml.template config.yml
# ERA5 initial-condition/verification fetches (aifs_ic.py) need internet
# access, which this H100 node does not have -- the ic_cache/ directory must
# already be warmed from a login node before this job runs (see CLAUDE.md).
python -u long_window_4dvar.py
