# Towards Stability of Autoregressive Neural Operators

Repository for paper "Towards Stability of Autoregressive Neural Operators". 

This is currently a work in progress and the included branch reflects the shallow water experiments.


## Training:

The config options (for all network types AFNO, UNet, FNO, S2CNN, DeepSphere) can be set in the config file. Example SWE and ERA5 configs can be bound in the config directory. Note
that the branch that built this repository was configured for SWE. ERA5 and Navier-Stokes compatibility will be added shortly.

example training launch script for a single gpu job:
```
python train.py --enable_amp --yaml_config=$config_file --config=$config --run_num=$run_num
```


## Inference:

example inference launch script for a single gpu job:
```
python inference/inference.py --config full_field --run_num 0
```
The inference script will use the last saved checkpoint corresponding to the config and run number to generate trajectories from multiple initial conditions in the validation dataset

The number of initial conditions and the length of the trajectories to be generated can be specified in the config file

## Data

The rotating Shallow Water Equation system can be generated through the following process: Note that the full dataset once generated is ~300 GB. 

```
python data_process/gen_SWE_from_ic_file.py --ic_file=$ic_file --output_dir=$output_dir
```
Note that this was configured for running multiple processes in parallel through a third-party controller. The data must then be interpolated onto the correct grid:
```
python data_process/fix_swe_data.py --data_root=$data_root
```





