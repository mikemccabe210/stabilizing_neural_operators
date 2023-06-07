import h5py
import numpy as np

with h5py.File('/pscratch/sd/s/shas1693/data/era5/static/orography.h5','a') as f:
    
    orog = f['orog'][:]
    omean = np.mean(orog)
    print(omean)
    ostd = np.std(orog)
    print(ostd)

    orog -= omean
    orog /= ostd

    f['orog'][...] = orog

    f.flush()


    
