<h1> ZaBantu <sup>Beta</sup></h1>

## Training Cross-Lingual Language Models for South African Bantu Languages 

* By: [Ndamulelo Nemakhavhani](https://linkedin.com/in/ndamulelonemakhavhani)


## Overview

ZaBantu is a project that aims to train cross-lingual language models for South African languages using the XLM-RoBERTa architecture. The project is inspired by the [AfriBERTa](https://arxiv.org/abs/2106.06118) and [XLM-RoBERTa](https://arxiv.org/abs/1911.02116) models. The project is currently in the beta phase and is being activily developed to ensure we have sufficient data and resources to benchmark the models.


## Documentation structure

You can navigate the documentation using the links below:

1. [Machine setup](./setup.md) - Instructions on how to setup your machine to run the code in this repository on a CUDA GPU
2. [Get the data](./get-data.md) - Instructions on how to get the data used in this project
3. [Training the model](./xlmr-masked-taining.md) - Instructions on how to train the model on our data or your own custom dataset
4. [Experiment tracking](./Tracking.md) - Instructions on how to track your experiments using Comet.ml. This is optional but recommended


## Getting Started

* You can quickly get started with training a light-weight model to see how everything works by following these instructions:
* WE ASSUME, that you already have access to a machine running `Ubuntu 20.04` with  `1 x NVIDIA Tesla T4` GPU. Any other version of Ubuntu or GPU should work similarly, but we have only tested on this configuration.

 - 1. Clone this repository to your local machine.
 ```bash
 git clone https://github.com/ndamulelonemakh/zabantu-beta.git
 cd zabantu-beta
 ```

 - 2. Install NVIDIA drivers (if not already installed)
 ```bash
    bash scripts/nvidia_setup.sh

    # This script will cause your machine to reboot
    # Wait for the machine to reboot, then run it again(next step)
 ```
 - 3. Wait for the machine to reboot, then install the CUDA toolkit
 ```bash
    bash scripts/nvidia_setup.sh
 ```

 - 4. Install Python Dependencies
 ```bash
 bash scripts/server-setup.sh

 # If any steps fail, try running the individual commands manually
 ```

> **Optional** If you intend to use comet.ml and other optional tools, copy the `env.template` file to `.env` and fill in the required fields

 - 5. Verify that your Pytorch installation is aware of your CUDA installation

 ```bash
 # Optional:
 python -c "import torch; print(torch.cuda.is_available())"
 ```

 - 6. Run the sample training pipeline by running the following command:

 ```bash
 make train_lite
 
 # If you are using comet.ml, you should be able to see the training progress on the comet.ml dashboard
 ```


## 



## Contributing

* Refer to the [CONTRIBUTING.md](./CONTRIBUTING.md) file for instructions on how to contribute to this project