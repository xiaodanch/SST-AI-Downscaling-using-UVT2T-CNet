import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
from pathlib import Path
import logging
from typing import Dict, List
import tqdm
import numpy as np
import xarray as xr
#from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from scipy import stats


def recursive_mkdir(path):
    split_dir = path.split("/")
    for k in range(len(split_dir)):
        d = "/".join(split_dir[:(k+1)])
        if (d != '') and (not os.path.exists(d)):
            os.mkdir(d)


#def rmse(x,y):
#    mse = torch.mean((x - y) ** 2,dim=[2,3])
#    rmse = torch.sqrt(mse) # scaler 
#    rmse = torch.mean(rmse)
#    return rmse
def rmse(x,y):
    x_np = x.detach().cpu().numpy().flatten()
    y_np = y.detach().cpu().numpy().flatten()
    rmse = np.sqrt(np.mean((x_np - y_np) ** 2))
    return rmse

def plot_figs(y1, y2, 
        plot_dir, var_name):
    """plot loss"""
    plt.figure(figsize=(10, 6))
    plt.plot(y1, label='Training')
    plt.plot(y2, label='Validation')
    plt.title(f'Training History')
    plt.xlabel('Epochs')
    plt.ylabel(var_name)
    plt.legend()
    #plt.grid(True)
    plt.savefig(os.path.join(plot_dir,f'{var_name}.png'))
    plt.close()

def save_model(epoch, model, optimizer, save_dir, lr_scheduler=None):
    path = os.path.join(save_dir, f"epoch_{epoch}_model.pth")  
    save_dict = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        #'scheduler_state_dict': lr_scheduler.state_dict(),
    }
    torch.save(save_dict, path)
    print(f'save {epoch} to {path}')

def save_best_model(epoch, model, optimizer, save_dir, lr_scheduler=None):
    path = os.path.join(save_dir, f"best_{epoch}_model.pth") 
    save_dict = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        #'scheduler_state_dict': lr_scheduler.state_dict(),
    }
    torch.save(save_dict, path)
    print(f'save best {epoch} to {path}')

def save_model_state(model, save_dir): 
    path = os.path.join(save_dir, "latest_model_state.pth")
    torch.save(model.state_dict(), path)
    print('saved')

#def norm_loss(x,y):
#    k = 9
#    alpha = 0.7
#    x_np = x.cpu().detach().numpy
#    y_np = y.cpu().detach().numpy
#    _, _, H, W = x_np.shape
#
#    mse_losses = []
#    r_losses = []
#    for i in range(0, H, k):
#        for j in range(0, W, k):
#            x_block = x_np[:, :, i:i+k, j:j+k].flatten()
#            y_block = y_np[:, :, i:i+k, j:j+k].flatten()
#
#            mse_ij = torch.mean((x_block-y_block) ** 2)
#            #mae_ij = np.mean(np.abs(x_block-y_block))
#            r_ij, _ = stats.pearsonr(x_block,y_block)
#
#            mse_losses.append(mse_ij)
#            r_losses.append(1-r_ij)
#    
#    mse_losses = np.array(mse_losses).reshape(-1,1)
#    r_losses = np.array(r_losses).reshape(-1,1)
#    scaler = MinMaxScaler()
#    scaler.fit(mse_losses)
#    mse_norm = scaler.transform(mse_losses) # z must n*1, 1*n is Error
#    scaler.fit(r_losses)
#    r_norm = scaler.transform(r_losses)
#
#    loss1 = np.mean(mse_norm)
#    loss2 = np.mean(r_norm)
#    loss = alpha * loss1 + (1-alpha) * loss2
#
#    return loss1, loss2, torch.tensor(loss, device=x.device, requires_grad=True)

def norm_loss(x, y):
    loss_fn = nn.MSELoss()
    if loss_fn(x,y) > 0.1:
        return 0, 0, loss_fn(x,y)
    else:
        k = 9
        alpha = 0.3
        _, _, H, W = x.shape

        mse_losses = []
        r_losses = []

        for i in range(0, H, k):
            for j in range(0, W, k):
                x_block = x[:, :, i:i+k, j:j+k].flatten()
                y_block = y[:, :, i:i+k, j:j+k].flatten()

                mse_ij = torch.mean((x_block - y_block) ** 2)

                x_mean = torch.mean(x_block)
                y_mean = torch.mean(y_block)
                cov = torch.mean((x_block - x_mean) * (y_block - y_mean))
                x_std = torch.std(x_block)
                y_std = torch.std(y_block)
                r_ij = cov / (x_std * y_std + 1e-8)  

                mse_losses.append(mse_ij)
                r_losses.append(1 - r_ij)

        # 
        mse_losses = torch.stack(mse_losses)
        r_losses = torch.stack(r_losses)

        mse_norm = (mse_losses - torch.min(mse_losses)) / (torch.max(mse_losses) - torch.min(mse_losses) + 1e-8)
        r_norm = (r_losses - torch.min(r_losses)) / (torch.max(r_losses) - torch.min(r_losses) + 1e-8)

        # 
        loss1 = torch.mean(mse_norm)
        loss2 = torch.mean(r_norm)
        loss = alpha * loss1 + (1 - alpha) * loss2

    return loss1, loss2, loss

def pearson_r(x,y):
    x_np = x.detach().cpu().numpy().flatten()
    y_np = y.detach().cpu().numpy().flatten()
    r, _ = stats.pearsonr(x_np, y_np)
    return r