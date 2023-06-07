import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import h5py
import math

def mean2(x):
    y = np.sum(x) / np.size(x)
    return y

def corr2(a,b):
    a = a - mean2(a)
    b = b - mean2(b)

    r = (a*b).sum() / math.sqrt((a*a).sum() * (b*b).sum())
    return r


def plot(dirs, savename):
    '''take in list of directories of hdf5 file, produce/save associated plots'''
    data = []
    gt = 0
    pred = []
    rmse = []
    for i, dir in enumerate(dirs):
        data.append(h5py.File(os.path.join(dir, 'autoregressive_predictions.h5'),'r'))
        gt = data[i]["ground_truth"]
        pred.append(data[i]["predicted"])
        rmse.append(data[i]["rmse"])


    fig, axs = plt.subplots(len(dirs) + 1, 5)
    #time mean
    mean = np.mean(gt,  axis = 0)

    #plot ground truth
    for i in range(5):
        axs[0, i].contourf(gt[5*i +1][0] -mean[0], vmin = -3, vmax = 3)
        axs[0,i].get_xaxis().set_visible(False)
        axs[0,i].get_yaxis().set_visible(False)


        #plot predicted
        for j in range(len(dirs)):
            axs[j+1, i].contourf(pred[j][5*i+1][0] - mean[0], vmin=-3, vmax = 3)
            axs[j+1,i].get_xaxis().set_visible(False)
            axs[j+1,i].get_yaxis().set_visible(False)
    plt.title("Turbulent Component of Fields")
    plt.savefig("/global/u2/s/sanjeevr/MLPDE-share/figures/turbulent_comp_" + savename + ".jpg")

    #plot ACC
    corr = np.zeros((len(dirs), 25))
    for dir in range(len(dirs)):
        for i in range(25):
            corr[dir, i] = corr2((gt[i,0]-mean[0]).flatten(), (pred[dir][i, 0]-mean[0]).flatten())
        plt.plot(corr, label = dirs[dir])

    plt.legend()
    plt.title("ACC")
    plt.xlabel("Timestep")
    plt.ylabel("ACC")
    plt.savefig("/global/u2/s/sanjeevr/MLPDE-share/figures/ACC_" + savename + ".jpg")

    #plot RMSE
    for dir in range(len(dirs)):
        plt.plot(rmse[dir][0:25], label = dirs[dir]) 
    plt.title("RMSE")
    plt.xlabel("Timestep")
    plt.ylabel("RMSE")
    plt.legend()
    plt.savefig("/global/u2/s/sanjeevr/MLPDE-share/figures/RMSE_" + savename + ".jpg")

    #plot time-average fields
    mean = []
    mean.append(np.mean(gt, axis = 0))
    for p in pred:
        mean.append(np.mean(p)) 
    fig, axs = plt.subplots(len(dirs) + 1)
    for i, m in enumerate(mean):
        axs[i].contourf(m) 
        axs[i].get_xaxis().set_visible(False) 
        axs[i].get_yaxis().set_visible(False) 
        
    plt.title("Time Averaged Fields")
    plt.savefig("/global/u2/s/sanjeevr/MLPDE-share/figures/time_average_" + savename + ".jpg")


