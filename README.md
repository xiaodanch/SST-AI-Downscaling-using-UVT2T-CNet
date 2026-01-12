# SST-AI-Downscaling-using-UVT2T-CNet

This repository contains the source code and implementation for the paper:

**"High-resolution regional SST AI downscaling based on multi-mode inputs from nested ROMS simulations"**.

## Abstract

This study proposes a multi-mode AI downscaling approach (UVT2T-CNet) to reconstruct high-resolution (HR) sea surface temperature (SST) fields in coastal regions. The model is trained on realistic LR-HR data pairs from nested ROMS simulations to bridge the "simulated-to-real" gap. By incorporating sea surface currents (U, V) as auxiliary physical inputs, the model effectively learns the underlying dynamical processes, such as temperature advection.

## Repository Content
```Plaintext
SST-AI-Downscaling-using-UVT2T-CNet/
├── models/
│   ├── __init__.py
│   ├── cbam.py
│   ├── unet.py
│   ├── cbam_unet.py
│   ├── cbam_unet.py
│   └── swinir.py      
├── GeoLAM/
│   └── lam_unet.ipynb        
├── predata/
│   ├── predata_clm.py
│   └── preprocess.py       
├── src/
│   ├── train.py
│   ├── min_norm_solvers.py
│   ├── srloss.py                      
│   └── utils.py                         
├── README.md               
```

## Key Results

* **Performance**: Reduced RMSE by 21.93% compared to bilinear interpolation and achieved a spatial correlation (R) of 0.86.


* **Physical Consistency**: GeoLAM analysis confirms that the model correctly learns temperature advection transport in shaping SST patterns.


* **Efficiency**: Achieving orders-of-magnitude faster speed than nested numerical modeling.

## Citation

If you find this code or research helpful, please cite our paper:

```bibtex
@article{Chen_2026,
author = {Chen, Xiaodan and Zheng, Fei and Xia, Jiangjiang and Zhu, Jiang and Shu, Yeqiang and Liu, Danian},
title = {High-resolution regional SST AI downscaling based on multi-mode inputs from nested ROMS simulations},
journal = {Machine Learning: Science and Technology},
doi = {10.1088/2632-2153/ae3054},
url = {https://doi.org/10.1088/2632-2153/ae3054},
year = {2026},
month = {jan},
publisher = {IOP Publishing},
volume = {7},
number = {1},
pages = {015003},
}
```

## Acknowledgments

```
This project builds upon the following open-source repositories:
* https://github.com/Jongchan/attention-module.git
* https://github.com/X-Lowlevel-Vision/LAM_Demo.git
* https://github.com/JingyunLiang/SwinIR.git
* https://github.com/isl-org/MultiObjectiveOptimization.git
```







