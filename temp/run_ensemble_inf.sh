#!/bin/bash
args="${@}"
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID
export WORLD_SIZE=$SLURM_NTASKS
export MASTER_ADDR=$SLURM_SUBMIT_HOST
export MASTER_PORT=29500 # default from torch launcher

# ensemble forecasting:
python inference/inference_ensemble.py --local_rank=$LOCAL_RANK --config=pretrained_two_step_afno_20ch_bs_64_lr1em4_blk_8_patch_8_cosine_sched_inf --run_num=0 ${args}
#python inference/inference_ensemble_precip.py --local_rank=$LOCAL_RANK --config=precip_inf --run_num=0 ${args}

