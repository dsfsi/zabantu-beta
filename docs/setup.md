> **NB:** We have only tested the project on Ubuntu 20.04, but it should work on other Linux distributions as well with
> minimal changes.

# Creating a Python Environment

## Option 1 - Using Conda(Recommended)

* Pre-requisite: Make sure you have [Conda](https://docs.anaconda.com/free/miniconda/#quick-command-line-install) installed

```bash
conda env create -f environment.yml
```

* Alternatively you can also leverage the `env` recipe defined in the `Makefile` to create the environment

```bash
make env
```

# Poetry Package Manager

* We recommend using [Poetry](https://python-poetry.org/docs/) to manage your Python package dependencies.
* You can install Poetry by running the following command:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```
* Refer to the [Poetry documentation](https://python-poetry.org/docs/) for more information on how to use Poetry.


# GPU Drivers & CUDA

* If you plan to use a GPU for training, you will need to install the appropriate GPU drivers and CUDA toolkit.
* Refer to the [PyTorch documentation](https://pytorch.org/get-started/locally/) for more information on how to install the appropriate GPU drivers and CUDA toolkit.
* To check which version of CUDA is supported by your GPU, refer to the [CUDA GPUs](https://developer.nvidia.com/cuda-gpus) page.
* A typical setup would go as follows:

```bash
# Update package lists
sudo apt update && sudo apt install -y ubuntu-drivers-common
sudo ubuntu-drivers install

## Install GPU drivers and CUDA SDKs
UBUNTU_VERSION="ubuntu2204/x86_64"  # NB: Modify this if you're using a different version of Ubuntu
wget https://developer.download.nvidia.com/compute/cuda/repos/${UBUNTU_VERSION}/cuda-keyring_1.1-1_all.deb
sudo apt install -y ./cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt -y install cuda  # This should install both the SDK and the NVIDIA drivers required to train on a GPU

# (Optional) Install the NVIDIA Container Toolkit for running GPU-accelerated Docker containers
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# Verify the installations
nvidia-smi
python3 -c "import torch; print(torch.cuda.is_available())"
```