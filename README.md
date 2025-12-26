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
│   └── cbam_unet.py      
├── GeoLAM/
│   ├── __init__.py
│   └── geolam.ipynb        
├── predata.py           
├── src/
│   ├── train.py             
│   └── utils.py                         
├── README.md             
└── requirements.txt      
```

## Key Results

* 
**Performance**: Reduced RMSE by 21.93% compared to bilinear interpolation and achieved a spatial correlation (R) of 0.86.


* 
**Physical Consistency**: GeoLAM analysis confirms that the model correctly learns temperature advection transport in shaping SST patterns.


* **Efficiency**: Achieving orders-of-magnitude faster speed than nested numerical modeling.

## Installation & Usage

1. Clone the repository:
```bash
git clone https://github.com/xiaodanch/SST-AI-Downscaling-using-UVT2T-CNet.git
```


2. Prepare the data using `predata.py`.
3. Start training:
```bash
python train.py
```

## Citation

If you find this code or research helpful, please cite our paper:

```bibtex
@article{Chen2025SST,
  title={High-resolution regional SST AI downscaling based on multi-mode inputs from nested ROMS simulations},
  author={Xiaodan Chen and Fei Zheng and Jiangjiang Xia and Jiang Zhu and Yeqiang Shu and Danian Liu},
  journal={Machine Learning: Science and Technology},
  year={2025},
  doi={10.1088/2632-2153/ae3054}
}
```

