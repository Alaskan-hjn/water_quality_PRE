#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 10:44:54 2023

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
import torch.utils.data as DD
import h5py 
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import random
import hdf5storage as hdf5
from params import *

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

setup_seed(1)

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
            nn.Linear(neurons,1)
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
file=hdf5.loadmat(fdir+'water_quality_2016-2021_idw_602_282_filtered.mat')
time=file['TIME'][:]
t1=datetime(2019, 1, 1)
t2=datetime(2021, 12, 31)
t1=t1.toordinal()+366
t2=t2.toordinal()+366
inx_t=np.where((time>=t1) & (time<=t2))[0]
sal=file['SAL'][inx_t]
lon=file['LON'][inx_t]
lat=file['LAT'][inx_t]
depth=file['DEPTH'][inx_t]
n_flow_flux=np.floor(file['N_FLOW_FLUX'][inx_t])
n_sea_flux=np.floor(file['N_SEA_FLUX'][inx_t])
po4=file['P'][inx_t]
n=file['N'][inx_t]

X=np.concatenate((sal,depth,lat,lon,n_flow_flux,n_sea_flux),axis=1)
Y=n
inx=np.where(X[:,0]>0)[0]
X=X[inx,:]
Y=Y[inx,:]

normaliztation=3
if normaliztation==1:
    # normalization 1
    file_norm=hdf5.loadmat(fdir+'flux_mean&std.mat')
    n_flow_flux_mean=file_norm['n_flow_flux_mean'][:]
    n_sea_flux_mean=file_norm['n_sea_flux_mean'][:]
    n_flow_flux_std=file_norm['n_flow_flux_std'][:]
    n_sea_flux_std=file_norm['n_sea_flux_std'][:]
    X_mean=np.nanmean(X,0)
    X_mean[4]=n_flow_flux_mean
    X_mean[5]=n_sea_flux_mean
    X_std=np.nanstd(X,0)
    X_std[4]=n_flow_flux_std
    X_std[5]=n_sea_flux_std
    X_mean=X_mean[np.newaxis,:]
    X_std=X_std[np.newaxis,:]
    X=(X-X_mean)/X_std
    Y_mean=np.nanmean(Y,0)
    Y_std=np.nanstd(Y,0)
    Y_mean=Y_mean[np.newaxis,:]
    Y_std=Y_std[np.newaxis,:]
    Y=(Y-Y_mean)/Y_std
if normaliztation==2:
    # normalization 2
    file_norm=hdf5.loadmat(fdir+'mean&std.mat')
    X_mean=file_norm['X_N_MEAN'][:]
    Y_mean=file_norm['Y_N_MEAN'][:]
    X_std=file_norm['X_N_STD'][:]
    Y_std=file_norm['Y_N_STD'][:]
    X=(X-X_mean)/X_std
    Y=(Y-Y_mean)/Y_std
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
    Y=(Y-Y_mean)/Y_std
    X[:,4]=flux_log

X=torch.from_numpy(X).float()
Y=torch.from_numpy(Y).float()

smp = Y.shape[0]
indices = list(range(smp))
split = int(np.floor(0.2 * smp))
np.random.shuffle(indices)  
train_indices, val_indices = indices[split:], indices[:split]
all_set = DD.TensorDataset(X,Y)
batch=100000
gpu0_bsz=100
acc_grad=1
train_sampler = DD.SubsetRandomSampler(train_indices)
valid_sampler = DD.SubsetRandomSampler(val_indices)
train_loader = torch.utils.data.DataLoader(all_set, batch_size=batch,sampler=train_sampler,num_workers=4,pin_memory=False)
validation_loader = torch.utils.data.DataLoader(all_set, batch_size=batch,sampler=valid_sampler,num_workers=4,pin_memory=False)
net, params = torch.load('mlp_model_79.pth')
if torch.cuda.device_count()>3:
        print("let's use ",torch.cuda.device_count()," gpus!")
        net=BalancedDataParallel(gpu0_bsz // acc_grad, net, dim=0)
        net.to(device)
else:
    net.to(device)
mm=[]
tt=[]
neuron_number=400
epoch=100
global_model_id = 79799
loss0 = nn.MSELoss(reduction='mean').to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001, weight_decay=0.01)

for name,param in enumerate(net.parameters()):
    if name <15:
        param.requires_grad = False
#%%
for t in range (epoch):
    net.train()
    epoch_loss=0
    print('epoch: ',t)
    for step, (batch_X,batch_Y) in enumerate(train_loader):
        optimizer.zero_grad()
        batch_t   = batch_Y.to(device)
        hat_t = net(batch_X.to(device))
        
        # delete 0 value
        mask=batch_t!=-999
        tmp1   = torch.masked_select(batch_t,mask)
        tmp2   = torch.masked_select(hat_t,mask)
        loss = loss0(tmp1,tmp2)

        epoch_loss += loss.item()/len(train_loader)
        loss.backward()         
        optimizer.step()
    rmse = np.sqrt(epoch_loss)
    print(rmse)
    mm.append(rmse)
    
    if t%5==0:
        net.eval()
        with torch.no_grad():
            epoch_test = 0
            for step, (test_X,test_Y) in enumerate(validation_loader):    
                output = net(test_X.to(device))
                test_t   = test_Y.to(device)
                hatTest_t = output.reshape(test_t.shape)
                mask_t=test_t!=-999
                tmp1_t   = torch.masked_select(test_t,mask_t)
                tmp2_t   = torch.masked_select(hatTest_t,mask_t)
                t_loss = loss0(tmp1_t, tmp2_t)
                all_loss=t_loss
                epoch_test += all_loss/len(validation_loader)
        test_rmse = torch.sqrt(epoch_test).item()
        tt.append(test_rmse)

params['train_RMSE']=mm
params['test_RMSE']=tt

endtime=datetime.now()
print(endtime-starttime)

print('train_RMSE')
print(params['train_RMSE'])
print('test_RMSE')
print(params['test_RMSE'])
print('Saving to'+' mlp_model_'+str(global_model_id)+'.pth'+' file')
torch.save((net,params),('mlp_model_'+str(global_model_id)+'.pth'))
print('Saving to rmse file')
sio.savemat(('mlp_rmse_'+str(global_model_id)+'.mat'), {'train_rmse':mm,'test_rmse':tt,'epoch':epoch,'neuron_number':neuron_number,'lr':params['learning_rate'],'decay':params['decay']})
