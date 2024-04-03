"""
Train a SentencePiece tokenizer using Byte Pair Encoding (BPE) or Unigram model.

Pre-requisites:
    - sudo apt-get install cmake build-essential pkg-config libgoogle-perftools-dev
    - pip install sentencepiece tokenizers

Usage:
    python train_native_sp_tokenizer.py \
        --input_data_paths /path/to/input/data1.txt,/path/to/input/data2.txt \
        --model_type bpe \
        --name sentencepiece.bpe \
        --vocab_size 10000 \
        --output_dir /path/to/output/dir \
        --normalize

More usage options are available at https://github.com/google/sentencepiece

BSD 3-Clause License
For full license text, refer to the LICENSE file included with this source code.

Author: Ndamulelo Nemakhavhani <GitHub: @ndamulelonemakh>
"""

import argparse
import glob
import logging
import os
import shutil
import tempfile
import time
import traceback
from pathlib import Path

import sentencepiece as sp
from tokenizers.normalizers import BertNormalizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_drive_tokenizer(tokenizer: sp.SentencePieceProcessor,
                         text: str = "muthu ufanela u guda machine learning 😁!") -> None:
    """
    Test drives the trained SentencePiece tokenizer by encoding and decoding a sample text.

    Args:
        tokenizer (sp.SentencePieceProcessor): Trained SentencePiece tokenizer.
        text (str, optional): Sample text to test the tokenizer. Defaults to a predefined string.
    """
    print("*" * 10, "Test driving tokenizer", "*" * 10)
    encoded_text_tokens = tokenizer.encode(text, out_type=str)
    encoded_text_ids = tokenizer.encode(text, out_type=int)
    print("Original text:", text)
    print("Tokenized text:", encoded_text_tokens)
    print("Vectorized text:", encoded_text_ids)
    print("-" * 20)
    print("Decoded text:", tokenizer.decode(encoded_text_ids))
    print("*" * 10, "Test done!", "*" * 10)


def _normalize_text_file(input_data_path: str, **kwargs) -> str:
    """
    Preprocess text using: huggingface.co/docs/tokenizers/en/api/normalizers#tokenizers.normalizers.BertNormalizer.

    Args:
        input_data_path (str): Path to the input text file.
        **kwargs: Additional keyword arguments to pass to the normalizer.

    Returns:
        str: Path to the normalized text file.
    """
    logger.info(f"Applying BertNormalizer on {input_data_path=}\n{kwargs=}")
    normalizer = BertNormalizer(**kwargs)

    name = Path(input_data_path).stem
    with tempfile.NamedTemporaryFile(mode="w", prefix=name + "_", suffix=".txt", delete=False) as fo:
        with open(input_data_path, "r") as f:
            for line in f:
                if line.strip():
                    normalized_text = normalizer.normalize_str(line.strip())
                    fo.write(normalized_text + "\n")
        logger.info(f"Normalized text file: {fo.name}")
    return fo.name


def _save_artifacts_to_outdir(name: str, output_dir: str) -> None:
    """
    Moves the SentencePiece model and vocabulary files to the specified output directory.

    Args:
        name (str): Base name of the model and vocabulary files.
        output_dir (str): Target directory to store the output files.
    """
    if not os.getcwd() == output_dir:
        logger.info(f"Moving model and vocab files to user-defined output directory: {output_dir}")
        try:
            os.makedirs(output_dir, exist_ok=True)
            for file in os.listdir(output_dir):
                if file.startswith(name):
                    os.remove(os.path.join(output_dir, file))
            shutil.move(src=f"{name}.model", dst=output_dir)
            shutil.move(src=f"{name}.vocab", dst=output_dir)
            logger.info(f"Moved model and vocab files to: {output_dir}")
        except Exception as e:
            logger.error(f"Error moving model and vocab files: {e}")
            traceback.print_exc()
            raise


def train_sentencepiece_tokenizer(input_data_paths: list[str],
                                  model_type: str = "bpe",
                                  name: str = None,
                                  vocab_size: int = 10000,
                                  output_dir: str = None,
                                  **kwargs) -> sp.SentencePieceProcessor:
    """
    Trains a BPE or Unigram tokenizer using [SentencePiece](https://github.com/google/sentencepiece)

    Args:
        input_data_paths (list[str]): Paths to the input data files.
        model_type (str, optional): Type of the model to train. Choose from ['bpe', 'unigram']. Defaults to 'bpe'.
        name (str, optional): Name of the tokenizer model.
        vocab_size (int, optional): Size of the vocabulary.
        output_dir (str, optional): Directory to store the trained model and vocabulary.
        **kwargs: Additional keyword arguments.

    Returns:
        sp.SentencePieceProcessor: Trained SentencePiece tokenizer.
    """
    logger.info(f"Training SentencePiece {model_type.upper()} tokenizer...")
    output_dir = output_dir or os.getcwd()
    name = name or "sentencepiece." + model_type
    model_path = Path(output_dir) / f"{name}.model"
    overwrite_existing = kwargs.pop("overwrite", False)
    apply_normalization = kwargs.pop("normalize", False)

    if model_path.exists() and overwrite_existing is False:
        logger.info(f"Model file already exists: {model_path} and {overwrite_existing=}. Returning existing model.")
        tokenizer = sp.SentencePieceProcessor(model_file=model_path.as_posix())
        return tokenizer

    if apply_normalization:
        input_data_paths = [_normalize_text_file(path) for path in input_data_paths]

    start = time.perf_counter()
    sp.SentencePieceTrainer.train(input=",".join(input_data_paths),
                                  model_prefix=name,
                                  model_type=model_type,
                                  vocab_size=vocab_size,
                                  **kwargs)
    elapsed = time.perf_counter() - start
    logger.info(f"Training completed after {elapsed:.2f}s")

    _save_artifacts_to_outdir(name, output_dir)
    tokenizer = sp.SentencePieceProcessor(model_file=model_path.as_posix())
    return tokenizer


def main(input_data_paths: str,
         output_dir: str,
         vocab_size: int = 10000,
         name: str = None,
         model_type: str = "bpe",
         eval_text: str = None,
         normalize: bool = False,
         overwrite: bool = False,
         log_level: str = "INFO") -> None:
    logger.setLevel(log_level.upper())

    # support wildcard expansion e.g. some/path/*.txt
    if "*" in input_data_paths:
        expanded_paths = ','.join(glob.glob(input_data_paths))
        logger.info(f"Expanded input data paths: {expanded_paths}")
        input_data_paths = expanded_paths

    logger.info(f"Training SentencePiece tokenizer with the following options:\n{vars()}")
    start = time.perf_counter()
    name = name or f"sentencepiece.{model_type}"
    tokenizer = train_sentencepiece_tokenizer(input_data_paths=input_data_paths.split(","),
                                              model_type=model_type,
                                              name=name,
                                              vocab_size=vocab_size,
                                              output_dir=output_dir,
                                              normalize=normalize,
                                              overwrite=overwrite)

    if eval_text:
        test_drive_tokenizer(tokenizer, eval_text)

    elapsed = time.perf_counter() - start
    logger.info(f"Training completed in {elapsed:.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a SentencePiece tokenizer. We do not save the model as a "
                                                 "HuggingFace tokenizer yet, because we dont know the model "
                                                 "architecture  at this stage.")
    parser.add_argument("--input_data_paths",
                        '-i',
                        type=str,
                        required=True,
                        help="Comma-separated string containing input text file paths. Note if any of the sentences "
                             "are longer than 4096 tokens, they will be SKIPPED.")
    parser.add_argument("--name",
                        '-n',
                        type=str,
                        required=False,
                        help="Name of the output model. Should start with 'sentencepiece'.")
    parser.add_argument("--output_dir",
                        '-o',
                        type=str,
                        required=True,
                        help="Output directory for the model and vocab files.")
    parser.add_argument("--vocab_size",
                        '-v',
                        type=int,
                        default=10000,
                        help="The maximum vocabulary size. Make sure this can be achieved based on the corpus size.")
    parser.add_argument("--model_type",
                        '-t',
                        type=str,
                        default="bpe",
                        choices=["bpe", "unigram"],
                        help="Model type to train.")
    parser.add_argument("--eval_text",
                        type=str,
                        required=False,
                        help="Text to evaluate the tokenizer with.")
    parser.add_argument("--normalize",
                        action="store_true",
                        help="Whether to apply text normalization to remove accents, etc.")
    parser.add_argument("--overwrite",
                        action="store_true",
                        help="Overwrite existing model files.")
    parser.add_argument("--log_level",
                        type=str,
                        default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Logging level.")
    args = parser.parse_args()
    print(args)
    main(**vars(args))
