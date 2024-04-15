<h1> ZaBantu <sup>Beta</sup></h1>

## Training Lite Cross-Lingual Language Models for South African Bantu Languages - Preview

* This repository aims to provide a QuickStart template(s) for training a polyglot(i.e. multilingual) Large Language Models (LLMs)
for `low-resource` settings with a specific focus on [Bantu languages](https://en.wikipedia.org/wiki/Bantu_languages).


* You can use this repo as a starting point for:
  * Masked Language Modeling (MLM) - see `train_masked` folder
  * Fine-tuning on semantic downstream tasks - see `notebooks` folder
    * For example Named Entity Recognition (NER), Sentiment Analysis, Fake News/Misinformation Detection, Text Generatio etc.


* Refer to the [docs](./docs/index.md) folder for more details or visit the [project website](https://zabantu-beta.github.io)
* You can also try out some trained models on [Huggingface](https://huggingface.co/dsfsi/zabantu-xlm-roberta)

## Pre-requisites

- [Ubuntu 20.04](https://ubuntu.com/download/desktop) - The project is guaranteed to work on Ubuntu 20.04, but should work on other Linux distributions as well.
- [NVIDIA GPU](https://www.nvidia.com/en-gb/geforce/graphics-cards/) - for training the Large Language Model (LLM) on a GPU
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) - for GPU acceleration. If you are training on a cloud Data Science VM, this should be pre-installed.

## (Recommended) Cloud Data Science VMs

* **SKIP THIS STEP** if you are using a VM provided by [DSFSI](https://dsfsi.github.io/)  or your own custom VM with atleast 1 NVIDIA CUDA-compatible GPU.
* Refer to the [Infrastructure Guide](./docs/infrastructure.md) for more details on how to deploy a GPU powered VM on AWS, GCP or Azure.
* Alternatively, you can check the `notebooks` folder to run the example code for free on [Google Colab](https://colab.research.google.com/) or [AWS SageMaker Studio Labs](https://studiolab.sagemaker.aws/)
* Other cheap GPU compute options include [Paperspace](https://www.paperspace.com/), [Run Pod](https://runpod.io/), [Jarvis Labs](https://jarvislabs.ai/) or [Lambda GPU Cloud](https://lambdalabs.com/service/gpu-cloud)

## QuickStart

### 1. Clone the repository

```bash
git clone https://github.com/ndamulelonemakh/zabantu-beta.git
cd zabantu-beta
```

<br/>

### 2. Install dependencies

- 2.1. **NVIDIA Drivers and CUDA Toolkit**
  * If you have opted not to use a [Managed Data Science VM](https://azure.microsoft.com/en-us/products/virtual-machines/data-science-virtual-machines), you will need to manually 
  install NVIDIA drivers and CUDA Toolkit using our utility scripts as follows:
  * **SKIP THIS STEP** if you are using a VM provided by [DSFSI](https://dsfsi.github.io/)  as the drivers are pre-installed.

```bash
bash scripts/nvidia_setup.sh
# On the first run, the script will reboot your machine to load the NVIDIA drivers
# After rebooting, run the script again to install the CUDA Toolkit
bash scripts/nvidia_setup.sh

# reload the .bashrc file to make sure the CUDA Toolkit is in your PATH
source ~/.bashrc
```

- 2.2. **Python Dependencies**
  * Once your NVIDIA depenedencies are in order, you can proceed to install Python related dependencies using the following commands:

```bash
bash scripts/server_setup.sh

# reload the .bashrc file to make sure conda and poetry are in your PATH
source ~/.bashrc
```

> **Optional** If you intend to use comet.ml and other optional tools, copy the `env.template` file to `.env` and fill in the required fields


<br/>

### 3. Pre-Train a sample Large Language Model (LLM)

```bash
make train_lite

# This will run a sample training session on a sample dataset under `demos/data
# Depending on how powerful your GPU is, training can take anywhere from a few minutes to a few hours
```

* If you wish to `reproduce` all the experiments, you can use `dvc repro` as follows:
* This will run all the experiments in the `dvc.yaml` file
  * Start by *downloading* the full training set by following the instructions in the [Get Data](./docs/get-data.md)
  * Then run the following command from the project root directory:
```bash
dvc repro
```

<br/>

### 4. Fine-tune the Pre-Trained LLM on a downstream task

```bash
make fake_news_detection  # TODO:
```

<br/>

### 5. Evaluate & Visualize the results

```bash
make evaluate_fake_news_detection  #TODO:
```

<br/>


* <span style="color: green; font-size:16px"> **Hint**: Refer to the `Makefile` for details on the commands used in the QuickStart guide. You can easily modify the commands to suit your specific use case.
</span>


<br/>

Project Structure
------------

    .
    ├── LICENSE              <- File that specifies the license under which the project is distributed.
    ├── README.md            <- This file
    ├── data/
    │   ├── raw/             <- The original, immutable data dump.
    │   ├── interim/         <- Intermediate data that has been transformed.
    │   └── processed/       <- Clean datasets that can be used out of the box without further processing.

    ├── demos/               <- Contains demonstration scripts or examples related to the project.
    ├── notebooks/           <- Contains Jupyter notebooks for running interactive analysis or experiments.

    ├── docs/                <- Contains detailed documentation files for loading data, provisioning servers, installing dependencies, etc.
    ├── infra/               <- Quick start Infrastructure as Code (IaC) scripts for deploying a GPU powered VMs in the cloud.
    
    ├── configs/             <- YAML configuration files for training different model variants.
    ├── scripts/             <- Contains utility bash scripts used to consolidate multiple python commands into a single bash command.
    
    ├── train_masked/        <- Directory related to training with masked data.
    ├── tokenize/            <- Directory related to tokenization tasks or scripts.

    ├── .dvc/                <- Hidden directory used by DVC for storing metadata and cache.
    ├── .git/                <- Hidden directory used by Git for version control.
    
    ├── .dvcignore           <- Specifies files and directories that should be ignored by Data Version Control (DVC).
    ├── dvc.lock             <- Lock file generated by DVC to ensure reproducibility of the pipeline.
    ├── params.yaml          <- DVC YAML parameters and hyperparameters used for training the models. Defining parameters here makes it easy to track how they affect the model's performance.
    ├── dvc.yaml             <- DVC pipeline configuration file that defines the stages and dependencies of the training pipeline

    ├── .gitignore           <- Specifies files and directories that should be ignored by Git version control.
    ├── pyproject.toml       <- Configuration file for Python projects that specifies dependencies and build settings.
    ├── poetry.lock          <- Lock file generated by Poetry package manager to ensure deterministic dependencies.

    ├── environment.yml      <- Conda environment file that specifies the project's dependencies.
    ├── Makefile             <- File that contains automation rules and commands for building and managing the project.
    ├── requirements.txt     <- File that lists the project's Python dependencies synced from the `pyproject.toml` file.

    ├── env.template         <- A template file for creating a new `.env` file for storing environment secrets
    └── .env                 <- User-specific secrets based on the `env.template` file. DO NOT COMMIT TO GIT!!!


# Train on your own data

* Once you have successfully trained the model on the sample dataset, you can proceed to train the model on your own dataset by following these steps:
* Download your dataset, which is expected to be a list of text files in a folder. Each text file should contain a single sentence per line.
* The accepted naming convention for the file is `somefile.whatever.<language-code>.txt` where `<language-code>` is the ISO 639-3 code for the language of the text file.
* You can optionally include you own custom configs under `configs` folder or use the defaults provided.
* Once you are ready, you can train the model on your own dataset by running a command similar to the one below:

```bash
# first, train your sentencepiece tokenizer
## remember to change any of the parameters to suit your specific use case
/bin/bash scripts/train_sp_tokenizer.sh --input-texts-path somefolder/mydocument.ven.txt \ 
                                      --sampled-texts-path data/temp/stagingdir/0 \
                                      --seed 47 \
                                      --alpha 0.5 \
                                      --tokenizer-model-type unigram \
                                      --vocab-size 70000 \
                                      --tokenizer-output-dir data/tokenizers/my-awesome-tokenizer-70k
	
# then, train your model
/bin/bash scripts/train_masked_xlm.sh --config configs/my-custom-or-existing.yml \
                                    --training_data demos/data \
                                    --experiment_name myfirst-xlmr-experiment \
                                    --tokenizer_path data/tokenizers/my-awesome-tokenizer-70k \
                                    --epochs 5

```

# Documentation

* There are two ways to access the documentation in this repository:
  * Visit the [project website](https://mungana-ai.github.io/train-a-polyglot-mlm/)
  * Run the following command in your terminal to build the documentation site locally:

    ```bash
    make docs
    ```
* If you encounter any issues or have any questions, please feel free to open an issue on the [GitHub repository](https://github.com/ndamulelonemakh/zabantu-beta)


# Contributing

* We welcome contributions to this project. Please refer to the [Contributing Guide](./docs/CONTRIBUTING.md) for more details on how to contribute.


# Citation

```tex
@misc{nemakhavhani-2024-ZabantuBeta,
  title   = {Training Lite Cross-Lingual Language Models for South African Bantu Languages - Preview},
  author  = {Ndamulelo Nemakhavhani, Vukosi Marivate, Jocelyn Mazarura},
  year    = {2024},
  url= {https://github.com/ndamulelonemakh/zabantu-beta},
  keywords = {NLP, BERT, Low-resource, XLM-R, Bantu}
}
```

--------

# Troubleshooting


* **nvcc: command not found..**
  - This indicates that the CUDA Toolkit is not installed or not in your PATH. You can install the CUDA Toolkit manually by following the instructions provided
  in the `scripts/nvidia_setup.sh` script.


* **CondaError: Run 'conda init' before 'conda activate'**
  - This error occurs when you have not initialized conda in your shell. You can fix this by running the following command:

  ```bash
  conda init bash
  ```


  * **scripts/server_setup.sh: script did not complete successfully**
  - Although this is rare, try running the script again to see if it completes successfully. If the error persists, please open an issue on the [GitHub repository](https://github.com/ndamulelonemakh/zabantu-beta)
  - Another possible solution, is to run the commands in the script manually in your terminal.


  * **Unable to push/pull to DVC Google Drive remote** - file not found
    - This is usually just a permission error
    - Ensure that the service account you are using has the necessary permissions to access the Google Drive folder, i.e. Share the folder with the service account email address
    as you would with any other Google Drive user.
    - You can also run `dvc pull` or `dvc push` with the `--verbose` flag to get more details on the error.

<hr/>

<br/>

<p align="center">Made somewhere in 🌍 by <a href="https://www.linkedin.com/in/ndamulelonemakhavhani/">N Nemakhavhani</a>❤️</p>
