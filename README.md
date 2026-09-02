<p align="center">
  <h1 align="center">SSFCNet: Stage-Specific Feature Coordination for Oriented Object Detection in Remote Sensing Images</h1>
</p>


# This is the code space for SSFCNet.

# Installation:
MMRotate depends on PyTorch, MMCV and MMDetection. Below are quick steps for installation. Please refer to Install Guide for more detailed instruction.

---

```text
conda create --name openmmlab python=3.9.0 -y
conda activate openmmlab
conda install pytorch==2.11 torchvision==0.26.0 cudatoolkit=12.8 -c pytorch 
pip install -U openmim
mim install mmcv-full
mim install mmdet
```

# SSCFNet Architecture

```
mmrotate/
  ├── api
  ├── datasets
  ├── models
  ├── utils
  ├── __init__.py
  ├── version.py     
  └── core         
tests/
  ├── data
  ├── test_data
  ├── test_models
  └── test_utils    
configs/
  └── ssfcnet     
tools/
  ├── analysis_tools
  ├── data     
  └── train.py            
```

# Training
Run the following command from the repository root directory:
```text
python tools/train.py
```

# Validation
Run the following command from the repository root directory:
```text
python tools/test.py
```
# Notes on Reproducibility
Minor performance fluctuations may occur due to differences in hardware, CUDA/cuDNN versions, PyTorch versions, random initialization, and data preprocessing details.
