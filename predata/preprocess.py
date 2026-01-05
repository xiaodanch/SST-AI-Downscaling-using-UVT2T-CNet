import torch
import glob
import xarray as xr
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from pathlib import Path

# Function to get files from input directories based on years
def get_files(input_path,years):
    input_files = []
    for input_dir in input_path:
        for year in years:
            files_of_year = glob.glob(os.path.join(input_dir,f"*{year}.nc"))
            input_files.extend(files_of_year)
    input_files = sorted(input_files)
    return input_files

# Function to read and calculate normalization data
def read_norm(var_name, time_name, input_file, clm_file, box=None):
    # get climatology data
    clm = np.load(clm_file)
    clm_mean = clm['clm_mean']
    clm_std = clm['clm_std']
    
    # read time and data
    with xr.open_dataset(input_file) as ds:
        times = ds[time_name].values
        input_data = ds[var_name].values.squeeze()

    if box is not None:
        Llon, Rlon, Blat, Ulat = box
        input_data = input_data[:, Blat:Ulat+1, Llon:Rlon+1]

    # time -> day of year (366) <=> index of climatology data
    time_pd = pd.to_datetime(times)
    time_false = time_pd.map(lambda x: x.replace(year=2012))
    fake_doy = time_false.dayofyear 

    # normalization
    norm_data = (input_data - clm_mean[fake_doy-1,0][:, None, None]) / clm_std[fake_doy-1,0][:, None, None]

    return norm_data


def preprocess_save(lr_path, hr_path, years, save_path):
    # clm file
    clm_t_lr = os.path.join('clm5km_temp_r54.npz')
    clm_t_hr = os.path.join('clm500m_temp_r486.npz')
    # u
    clm_u_lr = os.path.join('clm5km_u_r54.npz')
    clm_u_hr = os.path.join('clm500m_u_r486.npz')
    # v
    clm_v_lr = os.path.join('clm5km_v_r54.npz')
    clm_v_hr = os.path.join('clm500m_v_r486.npz')

    # gather input files
    lr_files = get_files(lr_path, years)
    hr_files = get_files(hr_path, years)
    print(f"data files of {lr_paths} from {years[0]} to {years[-1]}: {len(lr_files)} files")
    assert len(lr_files) == len(hr_files), f"NOT MATCH between lr and hr"

    # read input file and calculate normalization data
    for f_idx in tqdm(range(len(lr_files)), desc="Processing files"):
        # read and normalize data
        # temp
        t_lr = read_norm('temp', 'ocean_time', lr_files[f_idx], clm_t_lr, box=[81,134,41,94])
        t_hr = read_norm('temp', 'ocean_time', hr_files[f_idx], clm_t_hr, box=[31,516,37,522])
        # u
        u_lr = read_norm('u', 'ocean_time', lr_files[f_idx], clm_u_lr, box=[80,133,41,94])
        u_hr = read_norm('u', 'ocean_time', hr_files[f_idx], clm_u_hr, box=[30,515,37,522])
        # v
        v_lr = read_norm('v', 'ocean_time', lr_files[f_idx], clm_v_lr, box=[81,134,40,93])
        v_hr = read_norm('v', 'ocean_time', hr_files[f_idx], clm_v_hr, box=[31,516,36,521])

        with xr.open_dataset(lr_files[f_idx]) as ds:
            time = ds['ocean_time'].values

        for t_idx in range(len(time)):
            # numpy -> tensor
            # temp
            t_lr_tensor = torch.tensor(t_lr[np.newaxis, t_idx, :, :], dtype=torch.float32) #[1, 54, 54]
            t_hr_tensor = torch.tensor(t_hr[np.newaxis, t_idx, :, :], dtype=torch.float32)
            # u
            u_lr_tensor = torch.tensor(u_lr[np.newaxis, t_idx, :, :], dtype=torch.float32) #[1, 54, 54]
            u_hr_tensor = torch.tensor(u_hr[np.newaxis, t_idx, :, :], dtype=torch.float32)
            # v
            v_lr_tensor = torch.tensor(v_lr[np.newaxis, t_idx, :, :], dtype=torch.float32) #[1, 54, 54]
            v_hr_tensor = torch.tensor(v_hr[np.newaxis, t_idx, :, :], dtype=torch.float32)
            # cat
            lr_tensor = torch.cat((t_lr_tensor,u_lr_tensor,v_lr_tensor), dim=0) #[1, 54, 54]
            hr_tensor = torch.cat((t_hr_tensor,u_hr_tensor,v_hr_tensor), dim=0)
            # save as pt
            save_file = os.path.join(save_path, f"sample_-{np.array(time[t_idx], dtype='datetime64[D]')}.pt")
            torch.save({
                'lr': lr_tensor,
                'hr': hr_tensor
            }, save_file)
            # check shape
            if f_idx == 0 and t_idx ==0:
                print(f"check tensor shape: {lr_tensor.shape}, {hr_tensor.shape}")
        
        print(f"finish {Path(lr_files[f_idx]).name} with {t_idx+1} samples")


if __name__ == "__main__":
    # Define parameters-------------------------------------------------
    lr_paths = ["/data/roms/5km"]
    hr_paths = ["/data/roms/500m"]
    # Define years for training and validation
    train_years = range(1993, 2016+1)
    valid_years = range(2017, 2019+1)
    test_years = range(2020, 2022+1)
    # save path
    save_dir = "/data/tuv_norm"
    #-------------------------------------------------------------------

    # save directory for train and valid
    train_dir = os.path.join(save_dir, 'train')
    valid_dir = os.path.join(save_dir, 'valid')
    test_dir = os.path.join(save_dir, 'test')
    Path(train_dir).mkdir(parents=True, exist_ok=True)
    Path(valid_dir).mkdir(parents=True, exist_ok=True)
    Path(test_dir).mkdir(parents=True, exist_ok=True)

    # train set
    #preprocess_save(var_name, lr_paths, mr_paths, hr_paths, train_years, train_dir)
    # valid set 
    preprocess_save(lr_paths, hr_paths, train_years, train_dir)
    preprocess_save(lr_paths, hr_paths, valid_years, valid_dir)
    preprocess_save(lr_paths, hr_paths, test_years, test_dir)

