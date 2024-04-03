"""
Text Pre-processing utilities for cross-lingual language modelling using XLM-R.

Pre-requisites:
- Python 3.10 or newer
- Required Python packages: `transformers`, 'PyYAML', 'sentencepiece'`

Usage:
    python preprocess .py -i inputdir -c myconfig.yml

Arguments:
- `-i`/`--input_dir`: Directory containing input texts to be split.
- `-c`/`--config_file`: YAML configuration file specifying train/eval split parameters, normalization options. This is
expected to be the same configuration file used for training the target XLM-R model.

BSD 3-Clause License
For full license text, refer to the LICENSE file included with this source code.

Author: Ndamulelo Nemakhavhani <GitHub: @ndamulelonemakh>
"""

import logging
import os
import random
import shutil
import tempfile
import time
import json
from pathlib import Path
from typing import Union, TextIO

import yaml
from tokenizers.normalizers import BertNormalizer
from transformers import XLMRobertaTokenizerFast, XLMRobertaTokenizer

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def _prepare_tokenizer(tokenizer_path: str, max_length: int = 512) -> XLMRobertaTokenizerFast:
    """Converts a trained SentencePiece tokenizer model to XLMRobertaTokenizerFast format.

    Args:
        tokenizer_path: Path to the directory containing the tokenizer files.
        max_length: The maximum length of the sequences for the tokenizer.

    Side effects:
        Saves the converted tokenizer in the specified path. This will include the tokenizer configuration file,
        vocabulary file, and special tokens mapping.
    """
    logger.info(f"Converting SentencePiece tokenizer in {tokenizer_path=} to XLMRobertaTokenizerFast format")
    
    logger.info(f"Checking for SentencePiece model files in {tokenizer_path}")
    tokenizer_files = list(Path(tokenizer_path).glob("*.model"))
    
    # When we call save_pretrained, Huggingface generates an extra sentencepiece.bpe.model file
    # even though we already have a unigram model. Have not idea why this happens????
    if len(tokenizer_files) > 1:
        logger.warning(f"Multiple model files found in {tokenizer_path}. Using the type in tokenizer.json")
        with open(os.path.join(tokenizer_path, "tokenizer.json"), "r") as f:
            tokenizer_config = json.load(f)
            model_type = tokenizer_config["model"]["type"].lower()
            logger.info(f"Type from config is: {model_type=}")
            model_file = [
                pth
                for pth in tokenizer_files
                if pth.name.split(".")[1] == model_type
            ][0]
    else:
        model_file = tokenizer_files[0]
        model_type = model_file.name.split(".")[1]    
    
    tokenizer = XLMRobertaTokenizer(vocab_file=model_file, sp_model_kwargs={"model_file": model_file.as_posix()})
    tokenizer.save_pretrained(tokenizer_path)
    tokenizer.save_vocabulary(tokenizer_path)
    logger.info("Succesfully converted sentencepiece model files into XLM-R tokenizer format.")
    
    tokenizer = XLMRobertaTokenizerFast.from_pretrained(pretrained_model_name_or_path=tokenizer_path,
                                                        model_max_length=max_length,
                                                        local_files_only=os.path.exists(tokenizer_path),
                                                        use_fast=True)
    tokenizer.save_pretrained(tokenizer_path)
    tokenizer.save_vocabulary(tokenizer_path)
    dir_contents = os.listdir(tokenizer_path)
    logger.info(f"Converted tokenizer saved in {tokenizer_path}.\nContents: {dir_contents}")
    return tokenizer


def _dir_not_empty(dir_name: Union[str, Path]) -> bool:
    return os.path.exists(dir_name) and any(os.scandir(dir_name))


def _count_lines(f: TextIO) -> int:
    i = 0
    for i, _ in enumerate(f):
        pass
    return i + 1


def _read_training_config(config_path: str) -> dict[str, any]:
    """Reads the training configuration parameters from a YAML file.

    Args:
        config_path: Path to the YAML configuration file. This is the same file used for training the XLM-R model.

    Returns:
        A dictionary containing the parsed configuration parameters.
    """
    assert Path(config_path).exists(), f"Config file {config_path} does not exist."
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
        return config_dict


def _directory_maker(dirs: list[Union[str, Path]], clear: bool = False) -> bool:
    """
    Creates directories specified in `dirs`. If `clear` is True, existing directories will be emptied before creation.

    Args:
        dirs (list[Union[str, Path]]): A list of directory paths to create.
        clear (bool): Whether to clear the directories if they already exist. Defaults to False.

    Returns:
        True if all directories were successfully created or cleared; False if an error occurred.
    """
    logger.info(f"Creating {len(dirs)} directories: {dirs}")
    if clear:
        logger.warning("All contents of directories will be deleted!!")

    all_created = True
    for d in dirs:
        directory = Path(d)
        if clear and directory.exists():
            try:
                shutil.rmtree(directory)
            except Exception as e:
                logger.error(f"Error removing directory {directory}: {e}")
                all_created = False
                break
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Error creating directory {directory}: {e}")
            all_created = False
            break
    return all_created


def _normalize_text_file(input_data_path: str | Path, **kwargs) -> str:
    """
    Preprocess text using: huggingface.co/docs/tokenizers/en/api/normalizers#tokenizers.normalizers.BertNormalizer.

    Args:
        input_data_path (str | Path): Path to the input text file.
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


def _save_selected_lines_only(original_file: Union[str, Path],
                              destination_file: Union[str, Path],
                              indexes: list[int],
                              other_destination: Union[str, Path]):
    """
    Saves selected lines from an original file to a destination file and the rest to another file based on
     specified line indexes.

    Args:
        original_file (Union[str, Path]): Path to the original file.
        destination_file (Union[str, Path]): Path to the destination file where selected lines will be saved.
        indexes (list[int]): List of line indexes to be saved from the original file.
        other_destination (Union[str, Path]): Path to the destination file where the rest of the lines will be saved.
        When doing train/test split, this file will usually be the test file, allowing us to write both train and test
        files in a single pass.

    Raises:
        Exception: If an error occurs while saving the selected lines.
    """
    logger.info(f"ALT: Saving {len(indexes)} lines from {original_file} to {destination_file}")
    start = time.perf_counter()
    index_lookup = {i: True for i in indexes}
    with open(original_file, "r") as fin:
        with open(destination_file, "w") as fout:
            with open(other_destination, "w") as fother:
                for i, line in enumerate(fin):
                    if i in index_lookup:
                        fout.write(line)
                    else:
                        fother.write(line)
                    if i % 10000 == 0:
                        logger.info(f"Processed {i} lines so far")
    logger.info(
        f"Saved {len(indexes)} lines to {destination_file} in {time.perf_counter() - start:.2f} seconds.")
    return destination_file


def _verify_run(traindir: Union[str, Path],
                evaldir: Union[str, Path],
                config: dict[str, any]) -> None:
    logger.info(f"\n{'*' * 10}Running train/eval split sanity checks{'*' * 10}")
    train_files = list(Path(traindir).glob(config['data']['train_file_pattern']))
    assert len(train_files) >= 1, f"Expected at least 1 training file in {traindir} but found {train_files}."
    logger.info(f"Found {len(train_files)} training files. =>\n{train_files}")
    os.system(f"wc -l {traindir}/train*")

    eval_files = list(Path(evaldir).glob(config['data']['eval_file_pattern']))
    assert len(eval_files) >= 1, f"Expected at least 1 eval file in {evaldir} but found {eval_files}."
    logger.info(f"Found {len(eval_files)} eval files. =>\n{eval_files}")
    os.system(f"wc -l {evaldir}/eval*")
    os.system(f"wc -l {evaldir}/full_eval.txt")

    logger.info(f"{'*' * 10}Done running train/eval santity checks{'*' * 10}")


def do_train_eval_split(input_dir: Path,
                        data_config: dict[str, any],
                        force_overwrite: bool = False) -> tuple[Path, Path]:
    """Randomly select a subset of lines from each input text file to form the eval set, and save the rest
    to the train set. The split ratio is determined by the `train_eval_split_ratio` parameter in the configuration.

    Args:
        input_dir: The directory containing input texts to process.
        data_config: A dictionary containing configuration parameters for performing the train/eval split.
        force_overwrite: If True, existing train/eval directories will be overwritten.

    Returns:
        A tuple containing paths to the train and eval directories.
    """
    logger.info("RUnning train/eval split..")

    logger.debug(f"STAGE 1 - Searching input .txt files in => {input_dir}")
    input_files = list(input_dir.glob('*.txt'))
    logger.debug(f"Found {len(input_files)} input files.")
    if len(input_files) == 0:
        raise FileNotFoundError(f"No input files found in {input_dir}")

    train_dir = Path(data_config['train'])
    eval_dir = Path(data_config['eval_for_lang'])
    full_eval_filepath = eval_dir.joinpath('full_eval.txt').as_posix()

    logger.debug(f"STAGE 2 - Creating train and eval directories: {train_dir}, {eval_dir}")
    dirs_created = _directory_maker([train_dir, eval_dir], clear=force_overwrite)
    if not dirs_created:
        raise Exception(f"Unexpected error enocuntred while creating directories: {train_dir=}, {eval_dir=}")

    if not force_overwrite and _dir_not_empty(train_dir) and _dir_not_empty(eval_dir):
        logger.info(f"{os.listdir(train_dir)}\n\n{os.listdir(eval_dir)}")
        logger.warning(f"Train directory {train_dir} is not empty. "
                       f"And force_overwrite is {force_overwrite}. Returning existing directories.")
        return train_dir, eval_dir

    eval_ratio = data_config.get('train_eval_split_ratio', 0.2)
    logger.info(f"Using evaluation SPLIT_RATIO={eval_ratio}")

    language_filters = data_config.get('filter_langs', "all").split("|")
    use_all_langs = language_filters[0] == "all"
    logger.debug(f"Applied Language filters => {language_filters}")

    logger.debug(f"STAGE 3 - Splitting files into train and eval sets.")
    for f in input_files:
        logger.debug(f"Processing file: {f}")

        lang_code = Path(f).stem.split(".")[-1].replace(".txt", "")
        if lang_code == "sna":
            logger.warning(f"Not using shona for now")
            continue

        if not use_all_langs and lang_code not in language_filters:
            logger.warning(f"Skipping file {f} as {lang_code=} is not in {language_filters=}")
            continue

        if data_config.get('normalize_text', False) is True:
            logger.info(f"Normalizing actived. Normalising: {f}")
            f = _normalize_text_file(f)
            logger.info(f"Normalised filepath: {f}")

        with open(f, "r") as fin:
            n = _count_lines(fin)
            logger.debug(f"Found {n} lines for language {lang_code}")
            indexes = list(range(n))
            eval_indexes = random.sample(indexes, int(n * eval_ratio))
            train_indexes = list(set(indexes) - set(eval_indexes))

            logger.info(f"Finished splitting {n=}, {lang_code=} lines into {len(train_indexes)} "
                        f"train lines and {len(eval_indexes)} eval lines.")

            _save_selected_lines_only(original_file=f,
                                      destination_file=train_dir.joinpath(f"train.{lang_code}"),
                                      indexes=train_indexes,
                                      other_destination=eval_dir.joinpath(f"eval.{lang_code}"))

        # still within the loop
        logger.info(f"Appending {lang_code=} text to: {full_eval_filepath=}")
        exit_code = os.system(f"cat {eval_dir.as_posix()}/eval.{lang_code} >> {full_eval_filepath}")
        if exit_code != 0:
            raise Exception(f"Error appending {lang_code=} eval text to {full_eval_filepath=}. Aborting.")

    return train_dir, eval_dir


def main(input_dir: str,
         config_file: str,
         normalize: bool = False,
         log_level: str = "INFO",
         overwrite: bool = False,
         debug: bool = False,
         tokenizer_path: str = None) -> None:
    assert Path(input_dir).exists(), f"Input directory [{input_dir}] does not exist."
    assert Path(config_file).exists(), f"Config file [{config_file}] does not exist."

    logger.setLevel(log_level.upper())
    config = _read_training_config(config_file)
    random.seed(config['common']['random_seed'])
    logger.info(f"Done Loading config from {config_file}.\nDataConfig => {config['data']}")

    # update multi-source confis
    overwrite = overwrite or config['data'].get('overwrite_cache', False)
    normalize = normalize or config['data'].get('normalize_text', False)

    if normalize:
        logger.info("Adding normalization step to config.")
        config['data']['normalize'] = True

    traindir, evaldir = do_train_eval_split(Path(input_dir),
                                            config['data'],
                                            force_overwrite=overwrite)
    # Sanity checks
    if debug:
        _verify_run(traindir, evaldir, config=config)
    logger.info(f"Train/eval split completed ok. {traindir}, Eval dir: {evaldir=}")

    # Make sure the Fast tokenizer is available for pre-training
    _prepare_tokenizer(tokenizer_path=tokenizer_path or config['model']['tokenizer_path'],
                       max_length=config['model']['max_length'])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog="preprocess",
        description="Pre-process input texts for cross-lingual language modeling using XLM-R.")
    parser.add_argument("--input_dir",
                        "-i",
                        type=str,
                        required=True,
                        help="Directory containing input texts to be pre-processed.")
    parser.add_argument("--config_file",
                        "-c",
                        type=str,
                        required=True,
                        help="Path to config file for the experiment.")
    parser.add_argument("--normalize",
                        help="Apply text normalization to input texts before splitting.",
                        action="store_true")
    parser.add_argument("--log_level",
                        "-l",
                        type=str,
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        default="INFO")
    parser.add_argument("--overwrite",
                        help="Overwrite existing train/eval directories",
                        action="store_true")
    parser.add_argument("--debug",
                        "-d",
                        action="store_true",
                        help="Activate debug mode for diagnostic checks and logs",
                        default=False)
    parser.add_argument("--tokenizer_path",
                        default=None,
                        help="Path to the tokenizer directory. If not provided, the config file value will be used.")
    args = parser.parse_args()
    main(**vars(args))
