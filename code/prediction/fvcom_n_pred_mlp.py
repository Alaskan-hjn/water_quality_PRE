#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 10:11:32 2023

@author: jyj
"""


import numpy as np
from matplotlib import *
from datetime import datetime,timedelta
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import netCDF4 as nc
import sys
from torch.autograd import Variable
from matplotlib.pyplot import *
import scipy.io as sio 
import torch.utils.data as DD #定义数据集
import h5py 
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import random
import hdf5storage as hdf5
from params import *

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class MLP(nn.Module):
    def __init__(self, neurons):
        super(MLP, self).__init__()
        self.fc  = nn.Sequential(
            nn.Linear(6, neurons),
            nn.BatchNorm1d(neurons),
            nn.ReLU(),
            nn.Linear(neurons,neurons),
            nn.BatchNorm1d(neurons),
            nn.ReLU(),
            nn.Linear(neurons,neurons),
            nn.BatchNorm1d(neurons),
            nn.ReLU(),
            nn.Linear(neurons,neurons),
            nn.BatchNorm1d(neurons),
            nn.ReLU(),
            nn.Linear(neurons,neurons),
            nn.BatchNorm1d(neurons),
            nn.ReLU(),
            nn.Linear(neurons,neurons),
            nn.BatchNorm1d(neurons),
            nn.ReLU(),
            nn.Linear(neurons,neurons),
            nn.BatchNorm1d(neurons),
            nn.ReLU(),
            nn.Linear(neurons,neurons),
            nn.BatchNorm1d(neurons),
            nn.ReLU(),
            nn.Linear(neurons,neurons),
            nn.BatchNorm1d(neurons),
            nn.ReLU(),
            nn.Linear(neurons,neurons),
            nn.BatchNorm1d(neurons),
            nn.ReLU(),
            nn.Linear(neurons,1),
        )

    def forward(self, X):
        Y = self.fc(X)
        return Y
def rmse(v1,v2):
    v1=v1.reshape(1,-1)
    v2=v2.reshape(1,-1)
    rmse=np.sqrt(np.nanmean((v1-v2)**2))
    return rmse

def findSmallest(arr):
    smallest = arr[0]
    smallest_index = 0
    for i in range(1,len(arr)):
        if arr[i] < smallest:
            smallest = arr[i]
            smallest_index = i
    return smallest_index

def selectionSort(arr):
    newArr = []
    for i in range(len(arr)):
        smallest = findSmallest(arr)
        newArr.append(arr.pop(smallest))
    return newArr

def normalize(pt):
	normalized_v = (pt-np.nanmean(pt)) / np.nanstd(pt)
	return normalized_v


def get_clim(var):
    v0=var
    for im in range(1,13):
        var[im-1::12]=var[im-1::12]-np.nanmean(var[im-1::12],axis=0)
        
    vc=v0-var
    return var,vc

def mean_x(var):
    v_mean=np.zeros((101,97,12))
    for im in range(0,12):
        v_mean[:,:,im]=np.nanmean(var[:,:,im::12],axis=2)
    return v_mean
#%%
starttime=datetime.now()
params = set_params_mlp()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
fdir='../../data/'
file_sal=hdf5.loadmat(fdir+'SAL_SOM_HMM_15_12_4_7_rcp4.5_0325_19-25.mat')
file_flux=hdf5.loadmat(fdir+'flux_interp_to_roms_month_2019-2025.mat')
time=file_sal['time_hat_1'][:]
t1=datetime(2019, 1, 1)
t2=datetime(2025, 12, 31)
t1=t1.toordinal()+366
t2=t2.toordinal()+366
inx_t=np.where((time>=t1) & (time<=t2))[0]
sal=file_sal['sal_hat_1'][inx_t]
lon=file_sal['LON_FVCOM_1'][inx_t]
lat=file_sal['LAT_FVCOM_1'][inx_t]
time=file_sal['time_hat_1'][inx_t]
depth=file_sal['DEPTH_FVCOM_1'][inx_t]

time1=file_flux['TIME_FLUX_1']
t1=datetime(2016, 1, 1)
t2=datetime(2025, 12, 31)
t1=t1.toordinal()+366
t2=t2.toordinal()+366
inx_t=np.where((time1>=t1) & (time1<=t2))[0]
n_flow_flux=np.floor(file_flux['N_FLOW_FLUX_MONTH_1'][inx_t])
t1=datetime(2016, 1, 1)
t2=datetime(2025, 12, 31)
t1=t1.toordinal()+366
t2=t2.toordinal()+366
inx_t=np.where((time1>=t1) & (time1<=t2))[0]
n_sea_flux=np.floor(file_flux['N_SEA_FLUX_MONTH_1'][inx_t])

sal=sal.reshape(1,-1)
lat=lat.reshape(1,-1)
lon=lon.reshape(1,-1)
depth=depth.reshape(1,-1)
time=time.reshape(1,-1)
n_flow_flux=n_flow_flux.reshape(1,-1)
n_sea_flux=n_sea_flux.reshape(1,-1)

inx_x=~np.isnan(sal)
sal=sal[inx_x]
lat=lat[inx_x]
lon=lon[inx_x]
depth=depth[inx_x]
time=time[inx_x]
n_flow_flux=n_flow_flux[inx_x]
n_sea_flux=n_sea_flux[inx_x]

sal=sal[:,np.newaxis]
lat=lat[:,np.newaxis]
lon=lon[:,np.newaxis]
depth=depth[:,np.newaxis]
time=time[:,np.newaxis]
n_flow_flux=n_flow_flux[:,np.newaxis]
n_sea_flux=n_sea_flux[:,np.newaxis]


X=np.concatenate((sal,depth,lat,lon,n_flow_flux,n_sea_flux),axis=1)
inx=np.where(X[:,0]>0)[0]
X=X[inx,:]
time=time[inx,:]
sal=sal[inx,:]
lat=lat[inx,:]
lon=lon[inx,:]

# normalization
normaliztation=3
if normaliztation==1:
    # normalization 1
    file_norm=hdf5.loadmat(fdir+'/flux_mean&std.mat')
    p_flow_flux_mean=file_norm['p_flow_flux_mean'][:]
    p_sea_flux_mean=file_norm['p_sea_flux_mean'][:]
    p_flow_flux_std=file_norm['p_flow_flux_std'][:]
    p_sea_flux_std=file_norm['p_sea_flux_std'][:]
    X_mean=np.nanmean(X,0)
    X_mean[4]=p_flow_flux_mean
    X_mean[5]=p_sea_flux_mean
    X_std=np.nanstd(X,0)
    X_std[4]=p_flow_flux_std
    X_std[5]=p_sea_flux_std
    X_mean=X_mean[np.newaxis,:]
    X_std=X_std[np.newaxis,:]
    X=(X-X_mean)/X_std
    # get Y_mean  and  Y_std
    file=hdf5.loadmat(fdir+'ROMS_quality_n&p_mean&std.mat')
    Y_std=file['N_std'][:]
    Y_mean=file['N_mean'][:]
if normaliztation==2:
    # normalization 2
    file_norm=hdf5.loadmat(fdir+'mean&std.mat')
    X_mean=file_norm['X_N_MEAN'][:]
    Y_mean=file_norm['Y_N_MEAN'][:]
    X_std=file_norm['X_N_STD'][:]
    Y_std=file_norm['Y_N_STD'][:]
    X=(X-X_mean)/X_std
if normaliztation==3:
    # process flow_flux
    file_norm_flux=hdf5.loadmat(fdir+'log_X&log_Y.mat')
    X_log=file_norm_flux['X_N_log'][:,4]
    flux_log=np.log10(X[:,4])/X_log

    # process other data
    file_norm=hdf5.loadmat(fdir+'mean&std.mat')
    X_mean=file_norm['X_N_MEAN'][:]
    Y_mean=file_norm['Y_N_MEAN'][:]
    X_std=file_norm['X_N_STD'][:]
    Y_std=file_norm['Y_N_STD'][:]
    X=(X-X_mean)/X_std
    X[:,4]=flux_log
sio.savemat('X_N.mat',{'X':X})
X=torch.from_numpy(X).float()

all_set = DD.TensorDataset(X)
batch=1000000
gpu0_bsz=8
acc_grad=1
train_loader = torch.utils.data.DataLoader(all_set, batch_size=batch,num_workers=4,pin_memory=False)
hat_q=[]
tt=[]
# %%

net, params = torch.load('mlp_model_79799.pth')
loss0 = nn.MSELoss(reduction='mean').to(device)
net.to(device)
net.eval()
with torch.no_grad():
    for step, batch_X in enumerate(train_loader):
        batch_X=torch.squeeze(torch.stack(batch_X),0)
        epoch_test=0
        output = net(batch_X.to(device))
        hat_q.append(output)
hat_q=torch.cat(hat_q,dim=0)
quality_hat=hat_q.cpu()
quality_hat=np.array(quality_hat)
sio.savemat('Y_N.mat', {'Y':quality_hat})
if normaliztation==4:
    quality_hat = 10**(quality_hat*Y_log)
else:
    quality_hat = quality_hat*Y_std+Y_mean
n_hat=quality_hat

endtime=datetime.now()
print (endtime-starttime)

print(tt)
sio.savemat(('mlp_fvcom_pred_'+'721-4.5-2019-2025'+'_0507(16-21).mat'), {'n_hat':n_hat,'lat':lat,'lon':lon,'depth':depth,'sal':sal,'time':time})
