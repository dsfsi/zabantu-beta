#!/bin/bash
# -----------------------------------------------------------
# NVIDIA Setup Script
#
# This script installs the necessary drivers and CUDA toolkit
# for NVIDIA GPUs on an Ubuntu server.
#
# Author: Ndamulelo Nemakhavhani
# Date: 2024-03-31
# -----------------------------------------------------------
set -e
echo "
    _   _ _   _ _ _ _   
 | \ | | | | (_) (_)  
 |  \| | | | |_| |_ _ __  
 | . \` | | | | | | '_ \ 
 | |\  | |_| | | | | | | |
 |_| \_|\__,_|_|_|_| |_| |
"

# Update the system to ensure we have the latest packages
sudo apt update
sudo apt install -y ubuntu-drivers-common
if ! nvidia-smi &> /dev/null; then
  echo ">> Installing NVIDIA drivers..."
  sudo ubuntu-drivers install
  echo ">> Rebooting the system to apply the driver changes.."
  sudo reboot
  #   After rebooting, check if the drivers were installed successfully by running `nvidia-smi`
else
  echo ">> NVIDIA drivers already installed."
  nvidia-smi
fi

# -----------------------------------------------------------
# AT THIS POINT, YOU SHOULD HAVE NVIDIA DRIVERS INSTALLED
# - You can safely proceed with the CUDA toolkit installation below
# - Or use the `conda` to install both pytorch and CUDA toolkit at the same time(Recommended)
# - Example: conda install pytorch cudatoolkit=12.1 -c pytorch
# -----------------------------------------------------------

# You can now proceed to install the CUDA toolkit and SDKs as needed.
## Install GPU drivers and CUDA SDKs
UBUNTU_VERSION="ubuntu2004/x86_64"  # NB: Modify this if you're using a different version of Ubuntu
wget https://developer.download.nvidia.com/compute/cuda/repos/${UBUNTU_VERSION}/cuda-keyring_1.1-1_all.deb
sudo apt install -y ./cuda-keyring_1.1-1_all.deb
rm -v cuda-keyring_1.1-1_all.deb
sudo apt update
# This should install latest versions of both the SDK and the
## Disabling this by default because it installs the latest version of the toolkit and drivers, which usually more
## recent and more expensive GPUs. Uncomment if you want to install the latest versions.
#sudo apt -y install cuda

## If you want specific versions of the toolkit:
# Make sure the version you install is compatible with pytorch
# Refer to: https://pytorch.org/get-started/locally/
sudo apt-get install cuda-toolkit-12-1 --yes
## You can also install the drivers separately, but this usually results in broken dependencies
# So we don't recommend it unless you are one who lives to break things
# Use: sudo ubuntu-drivers list --gpgpu to list the available drivers for your GPU
# sudo apt-get install nvidia-driver-535


# Update environment variables to tell the system where to find the CUDA libraries
echo 'export PATH="/usr/local/cuda/bin/${PATH:+:${PATH}}"' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH="/usr/local/cuda/lib64/${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"' >> ~/.bashrc
. ~/.bashrc

## Verify the installation of CUDA
# Do the export again, because source ~/.bashrc doesn't work sometimes - Not sure why??
export PATH="/usr/local/cuda/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
nvcc --version
## If pytorch is already installed, you can check if it was built with CUDA support
# python -c "import torch; print(torch.cuda.is_available())"

# -----------------------------------------------------------
## ONE LAST STEP:
# 1. RUN source ~/.bashrc to apply the changes on your interactive shell
# 2. Alternatively, instead of running this script with `bash nvidia_setup.sh`, you can run it with `. nvidia_setup.sh

## Related
# https://www.nvidia.com/download/index.aspx
# https://cloud.google.com/compute/docs/gpus/install-drivers-gpu#verify-driver-install
# https://ubuntu.com/server/docs/nvidia-drivers-installation
# https://techcommunity.microsoft.com/t5/azure-high-performance-computing/getting-started-with-the-ncsv3-series-and-ncas-t4-v3-series/ba-p/3568874
# https://learn.microsoft.com/en-us/azure/virtual-machines/linux/n-series-driver-setup#ubuntu
# https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#meta-packages
# https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local
