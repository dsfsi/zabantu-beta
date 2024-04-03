#!/bin/bash
set -e

# ===========================================================================
# Sentencepiece Tokenizer
# ===========================================================================
#
# This script builds a BPE or Unigram tokenizer using sentencepiece.
# It supports building either a Byte Pair Encoding (BPE) or Unigram tokenizer based on
# your desired configuration.
#
# Usage:
#  ./train_tokenizer.sh [OPTIONS]
#
# Options:
#   --input-texts-path (REQUIRED)  - Path to the directory containing your
#                                     input text data.
#   --sampled-texts-path (REQUIRED) - Path to the directory where sampled
#                                     texts for training will be stored.
#   --seed (REQUIRED)               - Random seed for sampling (for reproducibility).
#   --alpha (OPTIONAL)             - Smoothing factor for multinomial sampling, higher value favors dominant languages
#                                     (default: 0.3).
#   --tokenizer-model-type (REQUIRED) - Type of tokenizer to train ("bpe" or
#                                     "unigram").
#   --vocab-size (REQUIRED)         - Desired vocabulary size for the tokenizer.
#   --tokenizer-output-dir (REQUIRED) - Path to the directory where the
#                                     trained tokenizer model will be saved.
#   --help                         - Display this help message.
#
# ===========================================================================

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --input-texts-path)
      inputTextsPath="$2"
      shift
      shift
      ;;
    --sampled-texts-path)
      sampledTextsPath="$2"
      shift
      shift
      ;;
    --seed)
      seed="$2"
      shift
      shift
      ;;
    --tokenizer-model-type)
      tokenizerModelType="$2"
      shift
      shift
      ;;
    --vocab-size)
      vocabSize="$2"
      shift
      shift
      ;;
    --tokenizer-output-dir)
      tokenizerOutputDir="$2"
      shift
      shift
      ;;
    --alpha)
      alpha="$2"
      shift
      shift
      ;;
    --help)
      echo "Usage: $0 --input-texts-path <path> --sampled-texts-path <path> --seed <seed> --tokenizer-model-type <type> --vocab-size <size> --tokenizer-output-dir <path> --alpha <alpha>"
      echo
      echo "Options:"
      echo "  --input-texts-path       Path to the input texts for sentencepiece training. Can either be a single .txt file or a directory containing .txt files."
      echo "  --sampled-texts-path     Path to temporarily store the sampled texts for sentencepiece training"
      echo "  --seed                   Random seed for sampling"
      echo "  --tokenizer-model-type   Type of tokenizer model (BPE or Unigram)"
      echo "  --vocab-size             Vocabulary size for the tokenizer"
      echo "  --tokenizer-output-dir   Output directory to store the trained tokenizer"
      echo "  --alpha                  Alpha value for multinomial sampling (default: 0.3)"
      echo "  --help                   Show this help message and exit"
      exit 0
      ;;
    *)
      echo "Unknown option: $key"
      exit 1
      ;;
  esac
done

source .env || true
TOKENIZER_LIB_PATH="$PWD/tokenize"
echo "$(date) - Running sentencepiece tokenization pipeline. WorkingDirectory=$(pwd)"

echo "Stage 1: Sampling the input texts for sentencepiece training"
python "$TOKENIZER_LIB_PATH/multinomial_sampler.py" \
  --datasets_path "$inputTextsPath" \
  --output_path "$sampledTextsPath" \
  --seed "$seed" \
  --alpha "$alpha"

# Overwrite the tokenizerInputs variable with the sampled texts
# This ensures that we are giving the tokenizer an inclusive set of texts for multi-lingual tokenization
tokenizerInputs=$(find "$sampledTextsPath" -type f -name "sampled.*" -print | paste -sd "," -)
echo "Sampling done. Listing directory=$sampledTextsPath >> $tokenizerInputs"

echo "Stage 2: Training sentencepiece tokenizer"
python "$TOKENIZER_LIB_PATH/train_sp_tokenizer.py" \
  --input_data_paths "$tokenizerInputs" \
  --model_type "$tokenizerModelType" \
  --vocab_size "$vocabSize" \
  --output_dir "$tokenizerOutputDir" \
  --normalize \
  --overwrite \
  --eval_text "Ndi matsheloni avhudi! Vho tanganedzwa fhano Tshakhuma."

echo "$(date) - Finished running sentencepiece tokenization pipeline"