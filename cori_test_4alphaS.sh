#!/bin/bash
#SBATCH -A <account>
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 1:00:00
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 10
#SBATCH --gpus-per-task=1

srun python3 app-build summer_data_4alphaS/fits.npz --input_h5 summer_data_4alphaS/4alphas_50runs_inputdata.h5 --order 3 --computecov \
--fit_obs 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
srun python3 app-tune summer_data_4alphaS/fits.npz summer_data_4alphaS/HEPData-ins1736531-v1-csv.h5 \
--target_bins 0 1 8 9 16 17 24 25 32 33 40 41 48 49 56 57 64 65 72 73 80 81 88 89 96 97 104 105 112 113 
srun python3 app-tune summer_data_4alphaS/fits.npz summer_data_4alphaS/HEPData-ins1736531-v1-csv.h5 --computecov \
--target_bins 0 1 8 9 16 17 24 25 32 33 40 41 48 49 56 57 64 65 72 73 80 81 88 89 96 97 104 105 112 113 
