"""
Utility script to sample sentences from monolingual text corpora using a multinomial distribution.

Pre-requisites:
- Python 3.10 or newer
- Required Python packages: `tqdm`

Usage:
    python multinomial_sampler.py -d /path/to/datasets -o /path/to/output -a 0.3 -s 47

Arguments:
- `-d`/`--datasets_path`: Path to the directory containing monolingual text corpora for different languages.
- `-o`/`--output_path`: Path to the directory where the script will store the sampled sentences.
- `-a`/`--alpha`: Multinomial alpha for sampling distribution. Lower value favours language with smaller corpus size.
- `-s`/`--seed`: Seed for random number generator. This is essential for reproducibility.

BSD 3-Clause License
For full license text, refer to the LICENSE file included with this source code.

Author: Ndamulelo Nemakhavhani <GitHub: @ndamulelonemakh>
"""

import logging
import os
import random
import time
from pathlib import Path

from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_sampling_probabilities(corpus_size_by_language: dict[str, int], alpha: float) -> dict[str, float]:
    """Calculates the sampling probabilities for each language using multinomial distribution.

    Args:
        corpus_size_by_language: A dictionary mapping language codes to their corpus sizes (number of sentences).
        alpha: The multinomial distribution smoothing factor (higher alpha favors larger corpora).

    Returns:
        A dictionary mapping language codes to their corresponding sampling probabilities.
    """

    total_sentences = sum(corpus_size_by_language.values())
    lang_probs = {lang: count / total_sentences for lang, count in corpus_size_by_language.items()}

    total_distr = sum(prob ** alpha for prob in lang_probs.values())
    sampling_probs = {lang: (prob ** alpha / total_distr) for lang, prob in lang_probs.items()}

    return sampling_probs


def sample_sentences(corpus_by_language: dict[str, list[str]],
                     sampling_probs: dict[str, float],
                     output_dir: Path) -> Path:
    """Samples sentences from each language corpus based on the calculated probabilities.

    Args:
        corpus_by_language: A dictionary mapping language codes to their corresponding sentence lists.
        sampling_probs: A dictionary mapping language codes to their sampling probabilities.
        output_dir: The directory path to temporarily store the sampled sentences.

    Returns:
        The directory path where the sampled sentences are stored.
    """

    output_dir.mkdir(exist_ok=True)

    for language, prob in tqdm(sampling_probs.items(), desc="Sampling Sentences", total=len(sampling_probs)):
        num_samples = int(round(prob * len(corpus_by_language[language])))
        sampled_sentences = random.sample(corpus_by_language[language], num_samples)

        output_file = output_dir / f"sampled.{language}"
        with open(output_file, "w", encoding="utf-8") as outfile:
            outfile.writelines(sampled_sentences)

    return output_dir


def _print_usage_with_examples() -> str:
    return '''python multinomial_sampler.py -d <DATASETS_PATH> -o <OUTPUT_PATH> [-a <ALPHA>] [-s <SEED>]
    
    Example 1: 
        python multinomial_sampler.py -d /path/to/datasets -o /path/to/output -a 0.3 -s 47
    
    '''


def main(input_data_path: str, output_path: str, alpha: float = 0.3, seed: int = 47):
    random.seed(seed)

    corpus_content_by_language = {}
    corpus_size_by_language = {}
    start_time = time.perf_counter()
    input_data_path = Path(input_data_path)

    if input_data_path.is_file():
        logger.info(f"Loading training file from path={input_data_path}")
        languages = input_data_path.name.split(".")[-1]
        with open(input_data_path, "r", encoding="utf-8") as infile:
            corpus_content_by_language[languages] = infile.readlines()
            corpus_size_by_language[languages] = len(corpus_content_by_language[languages])
    elif input_data_path.is_dir():
        logger.info(f"Loading training files from directory={input_data_path}")
        for file in Path(input_data_path).iterdir():
            if file.is_file():
                lang = file.name.split(".")[-1]
                with open(file, "r", encoding="utf-8") as infile:
                    corpus_content_by_language[lang] = infile.readlines()
                    corpus_size_by_language[lang] = len(corpus_content_by_language[lang])
    else:
        raise ValueError(f"Invalid {input_data_path}. Please provide a valid file or directory path.")

    logger.info(f"Number of training files found: {len(corpus_content_by_language)}")
    logger.info(f"Corpus size/language: {corpus_size_by_language}")
    sampling_probabilities = calculate_sampling_probabilities(corpus_size_by_language, alpha)
    os.makedirs(output_path, exist_ok=True)
    output_dir = sample_sentences(corpus_content_by_language, sampling_probabilities, Path(output_path))

    # Verify that the are the same number of files in the output directory as the number of languages
    # and that none of the files are empty
    if not len(list(output_dir.iterdir())) == len(corpus_content_by_language):
        expected_files = [f"sampled.{lang}" for lang in corpus_content_by_language.keys()]
        actual_files = [file.name for file in output_dir.iterdir()]
        raise ValueError("Not all languages were sampled.\n"
                         f"Expected({len(expected_files)}): {expected_files}\n"
                         f"Actual({len(actual_files)}): {actual_files}")

    if any([os.path.getsize(file) <= 0 for file in output_dir.iterdir()]):
        empty_files = [file.name for file in output_dir.iterdir() if os.path.getsize(file) <= 0]
        raise ValueError(F"Some of the sampled files are empty.\nEmpty files: {empty_files}")

    elapsed_time = time.perf_counter() - start_time
    logger.info(f"Sampling completed in {elapsed_time:.2f}s")


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Sample Sentences for SentencePiece Training",
                                     usage=_print_usage_with_examples(),
                                     prog="multinomial_sampler",
                                     add_help=True,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--datasets_path",
        '-d',
        type=Path,
        required=True,
        help="Path to the directory containing monolingual text corpora for different languages. Passing file paths is "
             "not supported.",
    )
    parser.add_argument(
        "--output_path",
        '-o',
        type=Path,
        required=True,
        help="Path to the directory where the script will store the sampled sentences",
    )
    parser.add_argument(
        "--alpha",
        "-a",
        type=float,
        default=0.3,
        help="Multinomial alpha for sampling distribution. "
             "Lower value favours language with smaller corpus size.\nThis also determines how versatile your tokenizer"
             " will be, the more the diversity, the better the versatility. Default is 0.3")
    parser.add_argument(
        "--seed",
        '-s',
        type=int,
        default=47,
        help="Seed for random number generator. This is essential for reproducibility. Default is 47.")

    args = parser.parse_args()
    main(input_data_path=args.datasets_path, output_path=args.output_path, alpha=args.alpha, seed=args.seed)
    sys.exit(0)
