import os
import time
from tqdm import tqdm
import argparse
import configparser
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from functools import partial
import numpy as np

import utils
sys.path.append('../models')
from cbam_unet import net
from dataset import get_processed_dataset, get_dataloader
from min_norm_solvers import MinNormSolver, gradient_normalizers
import srloss
import csv
import random

# seed
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if multi GPUs
torch.backends.cudnn.deterministic = True  #False if fast train
torch.backends.cudnn.benchmark = False     #True if fast train

# device & config file
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"==={device}===")
# configure
#----------modify different cases
output_path = "result/work1+uv/test1_cbam_uvt_bs=16"
BATCH_SIZE = 16
NUM_WORKERS = 8
LEARNINGRATE = 1e-4
EPOCHS = 3600
SAVE_EPOCH = 25
utils.recursive_mkdir(output_path)
print(f"save to {output_path}")
#----------
var_name = 'temp'
PROCESSED_DATA_DIR = "data/work1+uv_tuvnorm"

with open(os.path.join(output_path,"training_log.csv"), "w", newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["step", "scale1", "loss1", "scale2", "loss2", "loss"]) 

def check_invalid_values(tensor):
    return torch.isnan(tensor).any() or torch.isinf(tensor).any()

to_torch = partial(torch.tensor, dtype=torch.float16, device=device)
# train
def train_epoch(train_loader, model, optimizer, criterion1, criterion2, scaler):
    total_loss = 0
    total_rmse = 0
    total_pear = 0
    model.train()
    for step, (lr_data, label_data) in enumerate(train_loader):
        lr_data = lr_data
        label_data = label_data[:,[0],:,:]
        lr_data = lr_data.to(device)
        label_data = label_data.to(device)
        interp_data = F.interpolate(lr_data, size=label_data.shape[2:], mode='bilinear', align_corners=False)

        optimizer.zero_grad()
        
        output = model(interp_data)
        loss1 = criterion1(output, label_data)
        loss2 = criterion2(output, label_data)

        grads = {'loss1': [], 'loss2': []}
        optimizer.zero_grad()
        loss1.backward(retain_graph=True)
        for param in model.parameters():
            if param.grad is not None:
                grads['loss1'].append(param.grad.clone())
        optimizer.zero_grad()
        loss2.backward(retain_graph=True)
        for param in model.parameters():
            if param.grad is not None:
                grads['loss2'].append(param.grad.clone())
        losses = {'loss1': loss1.item(), 'loss2': loss2.item()}
        # grad normalize
        gn = gradient_normalizers(grads, losses, 'loss+')
        for task in grads:
            for gr_i in range(len(grads[task])):
                grads[task][gr_i] = grads[task][gr_i] / gn[task]
        sol, _ = MinNormSolver.find_min_norm_element([grads['loss1'], grads['loss2']])
        scale1, scale2 = sol[0], sol[1]  #scale[t] = float(sol[i])
        optimizer.zero_grad()
        loss = scale1 * loss1 + scale2 * loss2
        if (step) % 1000 == 0 or scale2 > scale1 or scale2 > 0.001 + 1e-6 or step == len(train_loader)-1:
            with open(os.path.join(output_path,"training_log.csv"), "a", newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow([step, scale1, loss1.item(), scale2, loss2.item(), loss.item()])

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        rmse = utils.rmse(output,label_data)
        total_rmse += rmse
        total_pear += utils.pearson_r(output,label_data)

    return total_loss / len(train_loader), total_rmse / len(train_loader), total_pear / len(train_loader)

def validate(valid_loader, model, criterion):
    """valid"""
    total_loss = 0
    total_rmse = 0
    total_pear = 0
    model.eval()
    
    with torch.no_grad(), torch.amp.autocast(device_type=device.type):
        for step, (lr_data, label_data) in enumerate(valid_loader):
            lr_data = lr_data
            label_data = label_data[:,[0],:,:]
            lr_data = lr_data.to(device)
            label_data = label_data.to(device)
            interp_data = F.interpolate(lr_data, size=label_data.shape[2:], mode='bilinear', align_corners=False)
            #print(f"valid of step {step}, {input_data.shape}, {label_data.shape}")
            output = model(interp_data)
            loss = criterion(output, label_data)
            #loss = criterion(output, label_data)
            total_loss += loss.item()
            rmse = utils.rmse(output,label_data)
            total_rmse += rmse
            total_pear += utils.pearson_r(output,label_data)

    return total_loss / len(valid_loader), total_rmse / len(valid_loader), total_pear / len(valid_loader)

def main():
    print(f"=== train dataset ===")
    train_data_dir = os.path.join(PROCESSED_DATA_DIR, 'train')
    train_ds = get_processed_dataset(train_data_dir)
    train_loader = get_dataloader(train_ds,BATCH_SIZE,NUM_WORKERS,True)

    print(f"=== valid dataset ===")
    valid_data_dir = os.path.join(PROCESSED_DATA_DIR, 'valid')
    valid_ds = get_processed_dataset(valid_data_dir)
    valid_loader = get_dataloader(valid_ds,1,NUM_WORKERS,False)

    model = net(use_checkpoint=True).to(device)
   # model = nn.DataParallel(model)
    optimizer = optim.Adam(model.parameters(), LEARNINGRATE)
    #criterion = nn.MSELoss()
    criterion1 = nn.L1Loss()
    criterion2 = srloss.RLoss()
    scaler = torch.amp.GradScaler(device.type)

    train_losses = []
    val_losses = []
    train_rmses = []
    val_rmses = []
    train_pears = []
    val_pears = []
    best_rmse = 1
    best_train_rmse = 1
    patience = 30
    counter = 0

    time0 = time.time()
    for epoch in range(EPOCHS):
        #print(f"Epoch {epoch+1}")
        start_time = time.time()
        train_loss, train_rmse, train_pear = train_epoch(train_loader, model, optimizer, criterion1, criterion2, scaler)
        val_loss, val_rmse, val_pear = validate(valid_loader, model, criterion1)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_rmses.append(train_rmse)
        val_rmses.append(val_rmse)
        #scheduler.step(val_loss)
        train_pears.append(train_pear)
        val_pears.append(val_pear)

        # plot
        utils.plot_figs(train_losses,val_losses,output_path,'loss')
        utils.plot_figs(train_rmses,val_rmses,  output_path,'rmse')
        utils.plot_figs(train_pears,val_pears,  output_path,'r')

        # save
        utils.save_model_state(model,output_path)

        # save and print
        if (epoch + 1) % SAVE_EPOCH == 0:
            utils.save_model(epoch + 1, model, optimizer,output_path)

        # early stop
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_train_rmse = train_rmse # record train rmse
            counter = 0 
           # save best model
            utils.save_best_model(epoch + 1, model, optimizer, output_path)
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered!")
                break

        torch.cuda.empty_cache()
        end_time = time.time()
        epoch_time = end_time - start_time
        print(f"Epoch {epoch+1}/{EPOCHS}: train_rmse={train_rmse:.2f}, valid_rmse={val_rmse:.2f}, took {epoch_time/60:.2f} mins")

    timee = time.time()
    print(f"====START{time0}, END{timee}===")
if __name__ == '__main__':
    main() 
