#!/bin/bash
set -e

# Script: server_setup.sh
# Description: Performs essential setup for a data science server on Ubuntu. Youw will typically only ever run this script once.
# Author: Ndamulelo Nemakhavhani


. ~/.bashrc
touch .env
echo "Updating package lists..."
sudo apt update

# Install essential dependencies
echo "Installing essential dependencies..."
sudo apt install -y wget git curl build-essential

# Install Miniconda
echo "Installing Miniconda..."
if [ -f "$HOME/miniconda3/bin/conda" ]; then
  echo "Miniconda already installed."
  conda --version
else
  echo "Installing Miniconda..."
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash Miniconda3-latest-Linux-x86_64.sh -b
  rm Miniconda3-latest-Linux-x86_64.sh
  export PATH="$HOME/miniconda3/bin:$PATH"
  echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.bashrc
  . ~/.bashrc
  conda --version
fi

# Install Poetry
echo "Installing Poetry..."
if [ -f "$HOME/.local/bin/poetry" ]; then
  echo "Poetry already installed."
  poetry --version
else
  echo "Installing Poetry..."
  curl -sSL https://install.python-poetry.org | python3 -
  export PATH="$HOME/.local/bin:$PATH"
  echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
  . ~/.bashrc
  poetry --version
fi

# Install GPU dependencies (if you have an NVIDIA GPU)
if ! nvidia-smi &> /dev/null; then
  echo "Installing NVIDIA drivers..."
  # Note this will cause the server to reboot, so you must re-run this script after the reboot
  bash ./scripts/nvidia_setup.sh
else
  echo "NVIDIA drivers already installed."
  nvidia-smi
  nvcc --version
fi

# Create and activate a conda environment (if environment.yml exists)
if [ -f "environment.yml" ]; then
  name=$(head -n 1 environment.yml | cut -d' ' -f2)
  echo "Creating and activating conda environment name=$name..."
  conda env create -f environment.yml --json
  eval "$(conda shell.bash hook)"
  conda activate $name
else
  echo "environment.yml not found. Skipping conda environment creation."
  exit 1
fi

# Install other Python dependencies (if pyproject.toml exists)
if [ -f "pyproject.toml" ]; then
  echo "Installing Python dependencies..."
  poetry install
  pip install accelerate -U
else
  echo "pyproject.toml not found. Skipping Python dependency installation."
fi

# Verify that Pytorch is ready to train with CUDA
python3 -c "import torch; print('Pytorch+CUDA ready to go?', torch.cuda.is_available())"


if [ "$?" -eq "0" ]; then
  echo "Setup completed ok. Please restart your terminal to apply the changes. or run `source ~/.bashrc`"
else
  echo "Setup completed with errors."
fi

# -----------------------------------------------------------
## ONE LAST STEP:
# 1. RUN source ~/.bashrc to apply the changes on your interactive shell
# 2. Alternatively, instead of running this script with `bash nvidia_setup.sh`, you can run it with `. nvidia_setup.sh
