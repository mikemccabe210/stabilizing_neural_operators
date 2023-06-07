import os
import glob
import subprocess


    
def generate_batch(ic_file, output_dir):
    return """#!/bin/bash -l

#SBATCH --time=01:00:00
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH --account=m4134
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH -J afno_swe_gen
#SBATCH --image=nersc/pytorch:ngc-22.02-v0
#SBATCH -o /pscratch/sd/m/mmccabe/run_logs/swe_gen.out
#SBATCH --open-mode=append

export HDF5_USE_FILE_LOCKING=FALSE
export NCCL_NET_GDR_LEVEL=PHB
export MASTER_ADDR=$(hostname)

set -x

srun -u --mpi=pmi2 shifter --env PYTHONUSERBASE=$HOME/.local/perlmutter/nersc-pytorch-22.02-v0/\
    bash -c "
    source activate dedalus3
    source export_DDP_vars.sh
    python data_process/gen_SWE_from_ic_file.py --ic_file=%s --output_dir=%s
    "
""" % (ic_file, output_dir)


if __name__ == '__main__':
    base_out_dir = '/pscratch/sd/m/mmccabe/data/swe/train'
    base_in_dir = '/pscratch/sd/m/mmccabe/data/swe/inits'
    files_paths = glob.glob(base_in_dir + "/*.npy")
    files_paths.sort()
    for i, file in enumerate(files_paths):
        out_path = base_out_dir + '/snapshots%02d' % i
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        print(file, out_path)
        sbatch = generate_batch(file, out_path)
        with open('tmp%s.sh'%i, 'w') as f:
            f.write(sbatch)
        p = subprocess.Popen( ['sbatch', './tmp%s.sh'%i], stdout=subprocess.PIPE )
