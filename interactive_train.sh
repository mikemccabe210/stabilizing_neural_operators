#!/bin/bash
#shifter --image=nersc/pytorch:ngc-22.02-v0 --env PYTHONUSERBASE=$HOME/.local/perlmutter/nersc-pytorch-22.02-v0/ python \
#  train.py --enable_amp --config pretrained_two_step_afno_20ch_bs_64_lr1em4_blk_8_patch_8_cosine_sched --run_num test0 
export MASTER_ADDR=$(hostname)
image=nersc/pytorch:ngc-22.02-v0
ngpu=4
config_file=./config/AFNO_interactive.yaml
config="afno_backbone"
run_num="check_era5stillworksonswebranch"
cmd="python train.py --enable_amp --yaml_config=$config_file --config=$config --run_num=$run_num"
srun -n $ngpu --cpus-per-task=32 --gpus-per-node $ngpu shifter --image=${image} --env PYTHONUSERBASE=$HOME/.local/perlmutter/nersc-pytorch-22.02-v0/ bash -c "source export_DDP_vars.sh && $cmd"
