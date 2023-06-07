import torch
import numpy as np
import h5py


years = [1979, 1989, 1999, 2004, 2010]

global_means = np.zeros((1,21,1,1))
global_stds = np.zeros((1,21,1,1))
mins = np.zeros((1,21,1,1))
maxs = np.zeros((1,21,1,1))
time_means = np.zeros((1,21,721, 1440))

for ii, year in enumerate(years):
    
    with h5py.File('/pscratch/sd/s/shas1693/data/era5/train/'+ str(year) + '.h5', 'r') as f:

        rnd_idx = np.random.randint(0, 1460-500)
        print(rnd_idx)

        global_means += np.mean(f['fields'][rnd_idx:rnd_idx+500], keepdims=True, axis = (0,2,3))
        global_stds += np.var(f['fields'][rnd_idx:rnd_idx+500], keepdims=True, axis = (0,2,3))
        maxs += np.max(f['fields'][rnd_idx:rnd_idx+500], keepdims=True, axis = (0,2,3))
        mins += np.min(f['fields'][rnd_idx:rnd_idx+500], keepdims=True, axis = (0,2,3))
        time_means += np.mean(f['fields'][rnd_idx:rnd_idx+500], keepdims=True, axis = (0))
        print(global_means.shape)
        print(global_stds.shape)
        print(maxs.shape)
        print(mins.shape)
        print(time_means.shape)

global_means = global_means/len(years)
global_stds = np.sqrt(global_stds/len(years))
time_means = time_means/len(years)
mins = mins/len(years)
maxs = maxs/len(years)

np.save('/pscratch/sd/s/shas1693/data/era5/global_means.npy', global_means)
np.save('/pscratch/sd/s/shas1693/data/era5/global_stds.npy', global_stds)
np.save('/pscratch/sd/s/shas1693/data/era5/mins.npy', mins)
np.save('/pscratch/sd/s/shas1693/data/era5/maxs.npy', maxs)
np.save('/pscratch/sd/s/shas1693/data/era5/time_means.npy', time_means)

print("means: ", global_means)
print("stds: ", global_stds)







