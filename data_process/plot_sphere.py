"""
Plot sphere outputs.
Usage:
    plot_sphere.py <files>... [--output=<dir>]
Options:
    --output=<dir>  Output directory [default: ./frames]
"""

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['NUMEXPR_MAX_THREADS'] = '1'

def build_s2_coord_vertices(phi, theta):
    phi = phi.ravel()
    phi_vert = np.concatenate([phi, [2*np.pi]])
    phi_vert -= phi_vert[1] / 2
    theta = theta.ravel()
    theta_mid = (theta[:-1] + theta[1:]) / 2
    theta_vert = np.concatenate([[np.pi], theta_mid, [0]])
    return np.meshgrid(phi_vert, theta_vert, indexing='ij')


def main(filename, start, count, output, clim, task):
    """Save plot of specified tasks for given range of analysis writes."""
    # Plot settings
    # task = 'vorticity'
    cmap = plt.cm.RdBu_r
    dpi = 100
    figsize = (8, 8)
    savename_func = lambda write: 'write_{:06}.png'.format(write)
    # Create figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1], projection='3d')
    # Plot writes
    with h5py.File(filename, mode='r') as file:
        dset = file['tasks'][task]
        if len(dset.dims) == 3:
            phi = dset.dims[1][0][:].ravel()
            theta = dset.dims[2][0][:].ravel()
        else:
            phi = dset.dims[2][0][:].ravel()
            theta = dset.dims[3][0][:].ravel()
        phi_vert, theta_vert = build_s2_coord_vertices(phi, theta)
        x = np.sin(theta_vert) * np.cos(phi_vert)
        y = np.sin(theta_vert) * np.sin(phi_vert)
        z = np.cos(theta_vert)

        dset = dset[:]
        norm = matplotlib.colors.Normalize(-clim, clim)
        fc_set = cmap(norm(dset))
        for index in range(start, start+count):
            data_slices = (index,) + tuple([slice(None) for _ in range(len(dset.shape)-1)])
            fc = fc_set[data_slices]
            if len(data_slices) == 4:
                fc = fc[0]
            # clim = np.percentile(np.abs(data), 99.5)

            #fc[:, theta.size//2, :] = [0,0,0,1]  # black equator
            if index == start:
                surf = ax.plot_surface(x, y, z, facecolors=fc, cstride=1, rstride=1, linewidth=0, antialiased=False, shade=False, zorder=5)
                ax.set_box_aspect((1,1,1))
                ax.set_xlim(-0.7, 0.7)
                ax.set_ylim(-0.7, 0.7)
                ax.set_zlim(-0.7, 0.7)
                ax.axis('off')
            else:
                surf.set_facecolors(fc.reshape(fc.size//4, 4))
            # Save figure
            savename = savename_func(file['scales/write_number'][index])
            savepath = output.joinpath(savename)
            fig.savefig(str(savepath), dpi=dpi)
    plt.close(fig)


if __name__ == "__main__":

    import pathlib
    from docopt import docopt
    from dedalus.tools import logging
    from dedalus.tools import post
    from dedalus.tools.parallel import Sync
    import os
    output_path = pathlib.Path('./frames').absolute()
    # # Create output directory if needed
    with Sync() as sync:
        if sync.comm.rank == 0:
            if not output_path.exists():
                output_path.mkdir()
    files = ['snapshots/' + p for p in os.listdir('snapshots')]
    task = 'vorticity'
    with h5py.File(files[-1], mode='r') as file:
        dset = file['tasks'][task][:]
    clim = np.percentile(np.abs(dset), 99.5)

    post.visit_writes(files, main, output=output_path, task=task, clim=clim)

    import cv2
    import numpy as np
    import glob

    frameSize = (800, 800)

    out = cv2.VideoWriter('output_video2.avi', cv2.VideoWriter_fourcc(*'DIVX'), 60, frameSize)

    for filename in sorted(glob.glob('{}/{}'.format(output_path, '/*.png'))):
        img = cv2.imread(filename)
        # print(img.shape)
        out.write(img)

    out.release()