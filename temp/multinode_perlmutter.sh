#!/bin/bash -l
#SBATCH --time=00:20:00
#SBATCH -C gpu

#SBATCH --account=m4134_g
##SBATCH -q early_science

#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=40
#SBATCH -J test_fcn_code
#SBATCH -o %x-%j.out

nproc_per_node=4
config='afno_20ch_bs_64_lr5em4_blk_8_patch_8_cosine_sched'
run_num='test'
#image="nvcr.io/nvidia/pytorch:20.10-py3"
image=nersc/pytorch:ngc-21.08-v1

nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST) # Getting the node names
nodes_array=( $nodes )
master_node=${nodes_array[0]}
#master_addr=$(srun --nodes=1 --ntasks=1 -w $master_node bash -c 'echo $SLURM_LAUNCH_NODE_IPADDR')
master_addr=$(hostname)
worker_num=$(($SLURM_JOB_NUM_NODES))

# Loop over nodes and submit training tasks
for ((  node_rank=0; node_rank<$worker_num; node_rank++ ))
do
  node=${nodes_array[$node_rank]}
  launch="python -m torch.distributed.launch \
    --nproc_per_node=$nproc_per_node --nnodes=$SLURM_JOB_NUM_NODES \
    --node_rank=$node_rank --master_addr=$master_addr \
    train.py --enable_amp --config=$config --run_num=$run_num"
  echo "Submitting node # $node_rank, $node, with cmd $launch"

  # Launch one SLURM task per node, and use torch distributed launch utility
  # to spawn training worker processes; one per GPU
  export NCCL_IB_DISABLE=1
  srun -u -N 1 -n 1 -w $node /pscratch/home/jpathak/dummy # silly fix for CUDA unknown errors
  srun -u -N 1 -n 1 -w $node \
    shifter --image=$image --module=gpu \
    --env PYTHONUSERBASE=/pscratch/home/jpathak/perlmutter/ngc-21.08-v1\
    bash -c "$launch" &
  pids[${node_rank}]=$!
  echo $pids
done

# Wait for completion
for pid in ${pids[*]}; do
    wait $pid
done

