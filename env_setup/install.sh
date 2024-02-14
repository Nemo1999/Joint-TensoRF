#!/bin/bash
# This script shows how to install dependency for Bundle-Adjusting TensoRF

# conda initialize
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"

# Create a conda environment named 'BAT'
conda  env create  --file "env_setup/requirements.yaml"
# Activate the environment
conda activate Bundle_Adjusting_TensoRF
# Install Pytorch and torchvision
conda install pytorch==1.13.1 torchvision pytorch-cuda=11.6 -c pytorch -c nvi
dia -y

pip install lpips pymcubes

# # fix PILLOW_VERSION not found error for old version of torchvision https://discuss.pytorch.org/t/cannot-import-name-pillow-version-from-pil/66096/5
# conda install pillow=6.2.1 -y

# # fix module 'distutils has no attribute "version"' error https://github.com/pytorch/pytorch/issues/69894
# pip install setuptools==59.5.0

pip install icecream
pip install plotly  wandb

# install sympy
conda install mpmath -y
conda install sympy -y

# dependency for TensoRF
pip install configargparse lpips imageio-ffmpeg kornia opencv-python plyfile scikit-image

# dependency for L2G
pip install roma
