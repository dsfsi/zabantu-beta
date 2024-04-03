"""
This is the entry point for the Masked Language Modelling  training pipeline. Its purpose is to:
1. Parse the YAML configuration file passed from the command line.
2. Instantiate a CustomXLMRTrainer class with the parsed config.
3. Call the train method to trigger the HuggingFace Trainer to start training on a GPU powered server.

Usage:
    python train.py --experiment_name <name> --config_path <path> [--tokenizer_path <path>] [--epochs <num>]

Example:
    python train_masked/pre_train.py --experiment_name "tshivenda-xlm-r" \\
                               --config_path "configs/tshivenda-xlmr-base.yml" \\
                               --tokenizer_path "data/tokenizers/tshivenda-xlmr-base" \\
                               --epochs 3

Args:
    - Refer to the UserOptions dataclass for the full list of arguments that can be passed from the command line.

BSD 3-Clause License
For full license text, refer to the LICENSE file included with this source code.

Author: Ndamulelo Nemakhavhani <GitHub: @ndamulelonemakh>
"""

import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Optional

import yaml
from transformers import HfArgumentParser

from _pretrain import XLMRMasterConfig, XLMRTrainingOrchestrator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class UserOptions:
    config_path: str = field(
        metadata={
            "help": "Path to the YAML config file containing the training parameters. "
                    "We recommend adding most of the customization here for better tracking of experiments."
        }
    )

    experiment_name: str = field(
        default=None,
        metadata={
            "help": "Name of the experiment. This will be used to create the training directory. "
                    "If None, the last part of the model_name is used."
        }
    )

    # These fields: Are already defined the config file but can be overridden from the command line(If necessary)
    tokenizer_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to the tokenizer. Both local and HuggingFace paths are supported."})

    epochs: int = field(default=-1,
                        metadata={"help": "Number of training epochs, if different from the config file. "
                                          "-1 means use the value in the config file."})


# Aware that this is a duplicate of th
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


def main():
    logger.info("XLM-R driver script started")
    start = perf_counter()

    parser = HfArgumentParser(UserOptions)
    opts: UserOptions = parser.parse_args_into_dataclasses()[0]
    logger.info(f"Loaded UserOptions => {opts}")

    config_dict = _read_training_config(opts.config_path)
    config: XLMRMasterConfig = XLMRMasterConfig.from_dict(config_dict)

    os.environ['LOG_LEVEL'] = config.common_config.log_level

    if opts.epochs > 1:
        config.training_config.num_train_epochs = opts.epochs
        logger.info(f"Overriding training epochs with {opts.epochs=}")

    if opts.tokenizer_path is not None:
        config.set_override(key='tokenizer_path', value = opts.tokenizer_path)
        logger.info(f"Overriding the tokenizer path with => {opts.tokenizer_path}")

    if opts.experiment_name is None:
        experiment_name = config.common_config.model_name
        logger.info("No experiment name passed from comman line. using model_name from config file:", experiment_name)
    else:
        experiment_name = opts.experiment_name

    experiment_full_path = Path(config.common_config.experiment_path).joinpath(experiment_name)
    os.makedirs(experiment_full_path, exist_ok=True)
    config.common_config.experiment_path = str(experiment_full_path)
    logger.info(f"experimentDirectory={experiment_full_path} ready for training.")

    common_config_path = os.path.join(experiment_full_path, "config.yml")
    shutil.copy2(opts.config_path, common_config_path)
    logger.info(f"Copied  train config={opts.config_path} to the experimentDirectory={common_config_path}")

    logger.info("Calling training orchestrator...")
    trainer = XLMRTrainingOrchestrator(config)
    trainer.train()

    logger.info(f"XLM-R driver script exiting after: {round(perf_counter() - start, 4)}s")


if __name__ == "__main__":
    main()
