import os
import glob
import xarray as xr
import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import uniform_filter1d
    
# calculate clm
def clm(path, var_name, prefix, box=None):
    time = []
    var = []
    for year in range(1993,2016+1): # train set
        print(year)
        # read data
        var_path = os.path.join(path, f'{prefix}{year}.nc')
        with xr.open_dataset(var_path) as ds:
            current_time = ds['ocean_time'].values
            current_var = ds[var_name].values.squeeze()
        
        # for uv -> select box match with temp
        if box is not None:
            Llon, Rlon, Blat, Ulat = box
            current_var = current_var[:, Blat:Ulat+1, Llon:Rlon+1]

        if year == 1993:
            print(f"shape: {current_var.shape}")

        # append to time and u
        time.append(current_time)
        var.append(current_var)

    time = np.concatenate(time,axis=0)
    var = np.concatenate(var,axis=0)

    # fake date to get doy (366 days in each year)
    time_pd = pd.to_datetime(time)
    time_false = time_pd.map(lambda x: x.replace(year=2012))
    fake_doy = time_false.dayofyear 

    # configure parameters and initial
    vWindowHalfWidth = 5 # 11 days, 5+1+5 as 1
    movwindow = 31 # smooth 31 days
    mclim = np.full((366,1),np.nan) # climate mean
    sclim = np.full((366,1),np.nan) # climate std

    # calculate
    index = range(len(fake_doy))
    for i in range(1,366+1):
        #print(f"day {i:03d}")
        if i != 60: #1.1->1, 1.31->31, 2.1->32, 2.28->59, 2.29->60, 3.1->61
            index_day = np.where(fake_doy==i)[0]
            lower_bound = index_day[:, None] - vWindowHalfWidth  # lower bound "1-5"
            upper_bound = index_day[:, None] + vWindowHalfWidth  # upper bound "1+5"
            condition = (index >= lower_bound) & (index <= upper_bound)
            mask = condition.any(axis=0) 

            mclim[i-1] = np.nanmean(var[mask,:,:])
            sclim[i-1] = np.nanstd(var[mask,:,:])
            #print(f"{i:03d}, {u[mask,:,:].shape}")
    # get 2.29
    mclim[60-1] = np.mean(mclim[[59-1,61-1]]) #2.29 using mean of 2.28 & 3.1
    sclim[60-1] = np.mean(sclim[[59-1,61-1]])

    # smooth with 31 days
    mclim_smooth = uniform_filter1d(mclim, size=movwindow, mode='wrap')
    sclim_smooth = uniform_filter1d(sclim, size=movwindow, mode='wrap')
    return mclim_smooth, sclim_smooth

#mclm_lr_u, sclm_lr_u = clm(path='/home/chenxiaodan/data/roms/5km', var_name='u', prefix='suf_5km_', box=[80, 133, 41, 94])
#print(mclm_lr_u.shape, sclm_lr_u.shape)
#np.savez("./data/clm5km_u_r54.npz", clm_mean=mclm_lr_u, clm_std=sclm_lr_u)

#mclm_lr_v, sclm_lr_v = clm(path='/home/chenxiaodan/data/roms/5km', var_name='v', prefix='suf_5km_', box=[81, 134, 40, 93])
#print(mclm_lr_v.shape, sclm_lr_v.shape)
#np.savez("./data/clm5km_v_r54.npz", clm_mean=mclm_lr_v, clm_std=sclm_lr_v)

mclm_hr_u, sclm_hr_u = clm(path='/home/chenxiaodan/data/roms/500m', var_name='u', prefix='suf_500m_', box=[30, 515, 37, 522])
print(mclm_hr_u.shape, sclm_hr_u.shape)
np.savez("./data/clm500m_u_r486.npz", clm_mean=mclm_hr_u, clm_std=sclm_hr_u)

mclm_hr_v, sclm_hr_v = clm(path='/home/chenxiaodan/data/roms/500m', var_name='v', prefix='suf_500m_', box=[31, 516, 36, 521])
print(mclm_hr_v.shape, sclm_hr_v.shape)
np.savez("./data/clm500m_v_r486.npz", clm_mean=mclm_hr_v, clm_std=sclm_hr_v)
