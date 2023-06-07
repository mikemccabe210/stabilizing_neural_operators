import h5py
import numpy as np
import dedalus.public as d3
import os
import glob
import multiprocessing as mp





def process_files(filename):
    print(filename)
    meter = 1 / 6.37122e6
    hour = 1
    second = hour / 3600
    g = 9.80616 * meter / second**2

    H = 5960 * meter

    Nphi = 256
    Ntheta = 128
    dtype = np.float64
    coords = d3.S2Coordinates('phi', 'theta')
    dist = d3.Distributor(coords, dtype=dtype)
    basis = d3.SphereBasis(coords, (Nphi, Ntheta), radius=1, dealias=1, dtype=dtype)
    h3 = dist.Field(name='h', bases=basis)
    #u3 = dist.Field(name='u', bases=basis)
    #v3 = dist.Field(name='v', bases=basis)
    u3 = dist.VectorField(coords, name='u', bases=basis)

    #filecount += 1
    with h5py.File(filename, mode='r+') as file:
        try:
            del(file['fields'])
        except:
            pass
            #print(filename, 'done in last batch')
            #return filename
        u = file['tasks']['u'][:]
        #u = file['tasks']['u'][:][:, 0]
        #v = file['tasks']['u'][:][:, 1]
        h = file['tasks']['h'][:]
        nphi = h.shape[1]
        ntheta = h.shape[2]
        out_fields = np.zeros((u.shape[0], 3, ntheta, nphi))
        delta = np.pi/(ntheta+1)
        for j in range(u.shape[0]):
            if j % 50 == 0:
                print('row', j)
            #u3['g'] = u[j]
            #v3['g'] = v[j]
            u3['g'] = u[j]
            h3['g'] = h[j]

            for i, pt in enumerate(np.linspace(np.pi-delta/2, delta/2, ntheta)):
                u_interp = np.swapaxes(d3.Interpolate(u3, 'theta', pt).evaluate()['g'], 1, 2)
                #u_interp = np.swapaxes(d3.Interpolate(u3, 'theta', pt).evaluate()['g'], 0, 1)
                #v_interp = np.swapaxes(d3.Interpolate(v3, 'theta', pt).evaluate()['g'], 0, 1)
                h_interp = np.swapaxes(d3.Interpolate(h3, 'theta', pt).evaluate()['g'], 0, 1)
                out_fields[j, :2, i:i+1] = u_interp * second / meter
                #out_fields[j, 1, i:i+1] = v_interp * second / meter
                out_fields[j, 2, i:i+1] = h_interp / meter

        file.create_dataset('fields', data=out_fields)
    print(filename, 'complete')
    return filename

if __name__ == '__main__':
    train_path = '/pscratch/sd/m/mmccabe/data/swe/train' 
    valid_path = '/pscratch/sd/m/mmccabe/data/swe/test'
    test_path = '/pscratch/sd/m/mmccabe/data/swe/out_of_sample'
    files = glob.glob(train_path + '/*.h5') + glob.glob(valid_path + '/*.h5') + glob.glob(test_path + '/*.h5')
    cpus = mp.cpu_count()
    print('CPUs', cpus)
    filecount = 0

    pool = mp.Pool(45)
    returns = pool.map(process_files, files)
