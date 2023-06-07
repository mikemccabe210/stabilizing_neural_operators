#!/bin/bash -l
#SBATCH --time=02:00:00
#SBATCH -C gpu
#SBATCH --dependency=singleton
#SBATCH -q regular
#SBATCH --account=m4134_g
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH -J Exp94_MakeFNOWork
#SBATCH --image=nersc/pytorch:ngc-22.02-v0
#SBATCH -o /pscratch/sd/m/mmccabe/run_logs/Exp94_MakeFNOWork.out
#SBATCH --open-mode=append
config_file=./config/AFNO_merged.yaml
config='afno_backbone'
run_num='Exp94_MakeFNOWork'
export HDF5_USE_FILE_LOCKING=FALSE
export NCCL_NET_GDR_LEVEL=PHB
export MASTER_ADDR=$(hostname)
set -x
git commit -a -m 'Commit before the head gets stolen!'
mkdir /pscratch/sd/m/mmccabe/tmp/Exp94_MakeFNOWork/
cp -r . /pscratch/sd/m/mmccabe/tmp/Exp94_MakeFNOWork/
cd /pscratch/sd/m/mmccabe/tmp/Exp94_MakeFNOWork/
git checkout Exp94_MakeFNOWork
srun -u --mpi=pmi2 shifter --env PYTHONUSERBASE=$HOME/.local/perlmutter/nersc-pytorch-22.02-v0/\
    bash -c "
    source export_DDP_vars.sh
    python train.py --enable_amp --yaml_config=$config_file --config=$config --run_num=$run_num
    "
rm -Rf /pscratch/sd/m/mmccabe/tmp/Exp94_MakeFNOWork/
