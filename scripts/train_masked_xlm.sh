#!/bin/bash
set -e

: '
# ╭━━━╮╱╱╱╱╱╱╱╱╱╱╱╱╭━━━╮
# ┃╭━╮┃╱╱╱╱╱╱╱╱╱╱╱╱┃╭━╮┃
# ┃╰━╯┣━━┳━┳━━┳━━┳━╋╯╭╯┃
# ┃╭━━┫╭╮┃╭┫╭╮┃┃━┫╭╋╮╰╮┃
# ┃┃╱╱┃╰╯┃┃┃╰╯┃┃━┫┃┃╰━╯┃
# ╰╯╱╱╰━━┻╯╰━╮┣━━┻╯╰━━━╯
# ╱╱╱╱╱╱╱╱╱╭━╯┃
# ╱╱╱╱╱╱╱╱╱╰━━╯

===============================================================================
A helper script to manually train a cross-linguagl(XLM) model on a SINGLE CUDA-enabled GPU.
===============================================================================

It perfoms the following steps:
- Preprocesses the input data by optionally normalising the text before doing a train/test split.
- Converts sentencepiece.model and sentencepiece.vocab files into a Huffingface Fast Tokenizer format.

===============================================================================
'

function usage {
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "╭━━━╮"
    echo "┃╭━╮┃"
    echo "┃╰━╯┣━━┳━┳━━┳━━┳━╮╭━━┳━╮"
    echo "┃╭━━┫╭╮┃╭┫╭╮┃┃━┫╭╯┃┃━┫┏┛"
    echo "┃┃╱╱┃╰╯┃┃┃╰╯┃┃━┫┃╰┫┃━┫┃"
    echo "╰╯╱╱╰━━┻╯╰━╮┣━━┻╯╱╰━━┻┛"
    echo "╱╱╱╱╱╱╱╱╭━╯┃"
    echo "╱╱╱╱╱╱╱╱╰━━╯"
    echo
    echo "Options:"
    echo "  --config <path>               Path to the yaml config file defining the training parameters (Required)"
    echo "  --training_data <path>        Path to directory containing raw text files to train with. Expected naming format is bantu.<language-code>.txt (Required)"
    echo "  --experiment_name <name>      Name of the experiment. Used to create the output directory. If not provided, the model_name from the config file is used."
    echo "  --tokenizer_path <path>       Path to the tokenizer directory. Can be a Huffingface tokenizer or a folder containing sentencepiece.model and sentencepiece.vocab files"
    echo "  --epochs <num>                Number of training epochs (default: 10)"
    echo "  --help                        Display this help message"
    echo
}


# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --config)
            config="$2"
            shift 2
            ;;
        --experiment_name)
            experiment_name="$2"
            shift 2
            ;;
        --training_data)
            training_data="$2"
            shift 2
            ;;
        --tokenizer_path)
            tokenizer_path="$2"
            shift 2
            ;;
        --epochs)
            epochs="$2"
            shift 2
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

echo "$(date) - Running xlm training pipeline from WorkingDir=$PWD. With user args: "
echo "$@"

source .env || echo "WARNING - No .env file found. This might cause errors"

# Set default values
epochs=${epochs:-10}
PYTHON_INTERPRETER=python3
TRAINING_LIB_PATH="$PWD/train_masked"
EXPERIMENTS_ROOT="$PWD/data/models/pretrained"
# Choose 1 gpu to avoid ncc2 errors
# Reference: https://stackoverflow.com/a/60158779/5232639
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES=0

# Alternative: Select least used gpu
# https://stackoverflow.com/a/69904259/5232639
# export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | sort -k2 -n | head -n1 | cut -d, -f1)


echo "STAGE 1: Preprocessing text at: $training_data"
${PYTHON_INTERPRETER} "$TRAINING_LIB_PATH/preprocess.py" \
    --input_dir "$training_data" \
    --tokenizer_path "$tokenizer_path" \
    --config_file "$config" \
    --log_level INFO


echo "STAGE 2: Training XLM model: $experiment_name"
${PYTHON_INTERPRETER} "$TRAINING_LIB_PATH/pre_train.py" \
  --experiment_name "$experiment_name" \
  --config_path "$config" \
  --tokenizer_path "$tokenizer_path" \
  --epochs "$epochs"


echo "$(date) - XLM training pipeline finished - see logs for details"
ls -al "$EXPERIMENTS_ROOT/$experiment_name" || echo "No model found at $EXPERIMENTS_ROOT/$experiment_name"


echo "STAGE 3: (Optional) Trying to upload training artifacts to remote storage"
dvc push data/models/pretrained || true
