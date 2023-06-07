#!/bin/bash -l


branch_name="$(git symbolic-ref HEAD 2>/dev/null)" ||
branch_name="(unnamed branch)"     # detached HEAD

branch_name=${branch_name##refs/heads/}

if [ $# -eq 0 ]
  then
    name=${branch_name}
  else
    name=${branch_name}_${1}
fi

cp_dir=/pscratch/sd/m/mmccabe/tmp/${name}/

echo "#!/bin/bash -l"
echo "#SBATCH --time=01:00:00"
echo "#SBATCH -C gpu"
echo "#SBATCH --dependency=singleton"
echo "#SBATCH -q regular"
echo "#SBATCH --account=m4134_g"
echo "#SBATCH --nodes=1"
echo "#SBATCH --ntasks-per-node=4"
echo "#SBATCH --gpus-per-node=4"
echo "#SBATCH --cpus-per-task=32"
echo "#SBATCH -J ${name}"
echo "#SBATCH --image=nersc/pytorch:ngc-22.02-v0"
echo "#SBATCH -o /pscratch/sd/m/mmccabe/run_logs/${name}.out"
echo "#SBATCH --open-mode=append"

echo "config_file=./config/AFNO_merged.yaml"
echo "config='afno_swe'"
echo "run_num='${name}'"

echo "export HDF5_USE_FILE_LOCKING=FALSE"
echo "export NCCL_NET_GDR_LEVEL=PHB"

echo "export MASTER_ADDR=\$(hostname)"

echo "set -x"

echo "git commit -a -m 'Commit before the head gets stolen!'"
echo  "mkdir ${cp_dir}"
echo  "cp -r . ${cp_dir}"
echo   "cd ${cp_dir}"
echo   "git checkout ${branch_name}"

echo 'srun -u --mpi=pmi2 shifter --env PYTHONUSERBASE=$HOME/.local/perlmutter/nersc-pytorch-22.02-v0/\
    bash -c "
    source export_DDP_vars.sh
    python train.py --enable_amp --yaml_config=$config_file --config=$config --run_num=$run_num
    "'
echo "rm -Rf ${cp_dir}"
