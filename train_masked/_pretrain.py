"""
This module contains custom training configurations and classes for pre-training XLM-R language models in low-resource
settings.It builds upon previous work by @KELECHI OGUEJI (see https://github.com/castorini/afriberta) and research
by @facebookresearch (see https://github.com/facebookresearch/XLM).

Usage:
    import yaml
    from _pretrain import XLMRunner

    # Load the configuration from a YAML file
    config_dict = yaml.safe_load(open("path/to/config.yaml"))
    config = ZabantuXLMConfig.from_dict(config_dict)

    # Create an instance of the XLMRunner class
    trainer = XLMRunner(config)

    # Train the multi-lingual language model
    trainer.train()

References:
    - XLM: https://github.com/facebookresearch/XLM
    - AfriBERTa: https://github.com/castorini/afriberta

BSD 3-Clause License
For full license text, refer to the LICENSE file included with this source code.

Author: Ndamulelo Nemakhavhani <GitHub: @ndamulelonemakh>
"""

import logging
import math
import os
import random
from collections import OrderedDict
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Union, Sequence
from typing import Optional

import numpy as np
import torch
import torch.utils.data as torch_data
import transformers
from torch.utils.data import DataLoader, Dataset
from transformers import MODEL_FOR_MASKED_LM_MAPPING, TrainingArguments, XLMRobertaConfig
from transformers import (Trainer,
                          DataCollatorForWholeWordMask,
                          DataCollatorForLanguageModeling,
                          PreTrainedTokenizer,
                          XLMRobertaForMaskedLM,
                          XLMRobertaTokenizerFast,
                          EarlyStoppingCallback)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

LOG_LEVEL = os.getenv("LOG_LEVEL", logging.DEBUG)


# region Util functions
def create_file_logger(log_file: str,
                       name: Optional[str] = None,
                       level: Optional[str] = None,
                       log_to_stdout: bool = False) -> logging.Logger:
    if not Path(log_file).parent.exists():
        print(f"Log file directory {Path(log_file).parent} does not exist. Trying to create it...")
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        print(f"Log file directory {Path(log_file).parent} created successfully.")
    level = level or os.getenv("LOG_LEVEL", logging.DEBUG)
    name = name or f"{__name__}_logger"
    logger = logging.getLogger(name)
    logger.setLevel(level=level)
    formatter = logging.Formatter("%(asctime)s - %(module)s [%(funcName)s] %(lineno)d: %(message)s")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level=level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if log_to_stdout:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(level=level)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    return logger


# endregion


# region Configs

@dataclass
class XLMRDatasetConfig:
    train: str = field(metadata={
        'help': 'training data directory. File patten is train.{lang} eg. train.ven'})
    eval_all: str = field(
        metadata={'help': 'path to a text file with all the languages combined for model evaluation'})
    eval_for_lang: str = field(
        metadata={'help': 'evaluation data directory. File patten is eval.{lang} eg. train.ven'})
    train_file_pattern: str = field(
        metadata={'help': 'string pattern to be passed in to glob to retrieve training data'})

    eval_file_pattern: str = field(metadata={
        'help': 'training data directory. File patten is train.{lang} eg. train.ven'})

    filter_langs: str = field(default="all",
                              metadata={
                                  'help': 'Pipe seperated ISO 639-3 language codes to train on e.g. nso|ven'})
    train_eval_split_ratio: float = field(default=0.2, metadata={'help': 'Ratio of eval to train data'})
    
    tokenization_sampling_factor: float = field(default=1.0, metadata={'help': 'Sampling factor for tokenization'})
    normalize_text: bool = field(default=True, metadata={'help': 'Normalize text before tokenization & training'})
    overwrite_cache: bool = field(default=False, metadata={'help': 'Override preprocessed files in each run'})
    rechunk: bool = field(default=False, metadata={'help': 'Rechunk the sentences to set lenghts close to max_length'})
    input_text_encoding: str = field(default='utf-8', metadata={'help': 'Encoding of the input text'})


@dataclass
class XLMRCommonConfig:
    experiment_path: Optional[str] = field(metadata={
        'help': 'path to the experiment directory where the model will be saved'})

    model_name: Optional[str] = field(metadata={'help': 'Name of the model checkpoint e.g. zulu-xlm-r-base'})

    log_level: Optional[str] = field(default="INFO",
                                     metadata={
                                         'help': 'log level to be used. One of: DEBUG, INFO, WARNING, ERROR, CRITICAL'})

    evaluate_only: Optional[bool] = field(default=False, )

    train_only: Optional[bool] = field(default=False)

    random_seed: Optional[int] = field(default=47,
                                       metadata={'help': 'random seed to be used to allow for reproducibility'})

    gpu_count: Optional[int] = field(default=1, metadata={'help': 'number of GPUs to use for training'})

    mlm_probability: Optional[float] = field(
        default=0.15,
        metadata={
            'help': 'Ratio of tokens to mask for train_masked language modeling loss'})

    default_xlm_checkpoint: Optional[str] = field(
        default="xlm-roberta-base",
        metadata={
            'help': 'default cross lingual model achitecture checkpoint to '
                    'use from hugging face'})

    use_whole_word_mask: Optional[bool] = field(default=False,
                                                metadata={
                                                    'help': ''})
    language_sampling_factor: Optional[float] = field(default=1.0,
                                                      metadata={'help': ''})

    resume_training: Optional[bool] = field(
        default=False,
        metadata={
            'help': 'Weather to forcefully resume training from the last saved checkpoint'})

    input_text_encoding: Optional[str] = field(default='utf-8')
    minimum_tokens_per_sentence: Optional[int] = field(
        default=5,
        metadata={'help': 'Ignore sentences with less than this number of tokens'})

    train_from_scratch: Optional[bool] = field(
        default=True,
        metadata={'help': 'Weather to train from scratch or resume training from the last saved checkpoint'})


@dataclass(kw_only=True, frozen=True)
class XLMRMasterConfig:
    """Convenience class to hold all the configuration parameters for the cross lingual language modelling"""
    model_config: XLMRobertaConfig
    data_config: XLMRDatasetConfig
    training_config: TrainingArguments
    common_config: XLMRCommonConfig
    overrides: dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        result = {'training': self.training_config.to_dict(),
                  'model': self.model_config.to_dict(),
                  'data': asdict(self.data_config),
                  'common': asdict(self.common_config),
                  'overrides': self.overrides}
        return result
    
    def set_override(self, key: str, value: Any):
        if self.overrides is None:
            self.overrides = {}
        self.overrides[key] = value

    @classmethod
    def from_dict(cls, config_dict: dict[str, any]):
        return cls(model_config=XLMRobertaConfig(**config_dict['model']), 
                   data_config=XLMRDatasetConfig(**config_dict['data']) , 
                   training_config=TrainingArguments(**config_dict['training']), 
                   common_config=XLMRCommonConfig(**config_dict['common']), 
                   overrides=config_dict.get('overrides', {}))


# endregion


# region Data classes

class XLMREvaluationDataset(torch_data.Dataset):

    def __init__(self, tokenizer: XLMRobertaTokenizerFast,
                 eval_file_path: str,
                 config: XLMRCommonConfig,
                 defer_load: bool = False) -> None:
        """Initialise an in-memory text collection used to evaluate a cross lingual language model.

        Args:
            tokenizer (XLMRobertaTokenizerFast): The tokenizer used to encode the text.
            eval_file_path (str): The path to the file containing the text to be evaluated.
            defer_load (bool, optional): Defer loading the data until the first call to __getitem__. Defaults to False.

        Raises:
            FileNotFoundError: If the file at eval_file_path does not exist.

        """
        assert tokenizer.vocab_size > 0, "Tokenizer must be initialised with a vocabulary."

        self.config: XLMRCommonConfig = config
        self.eval_file_path: Union[str, Path] = eval_file_path
        self.tokenizer: XLMRobertaTokenizerFast = tokenizer
        self.examples: Optional[list[dict[str, torch.Tensor]]] = None
        self._class_name = self.__class__.__name__
        self._logger = create_file_logger(
            log_file=Path(config.experiment_path).joinpath("logs",
                                                           f"{__name__}.{self._class_name}.log").as_posix())
        if not defer_load:
            corpus = self.__load_data()
            self.examples = self.__texts_to_sequences(corpus)
        else:
            self._logger.warning(
                f"Deferring loading of data until first call to __getitem__ because defer_load={defer_load}")

    def __read_text(self, text_file_path: str, return_list: bool = True) -> list[str]:
        """Read the text from the file at text_file_path.

        Args:
            text_file_path (str): The path to the file containing the text to be evaluated.

        Raises:
            FileNotFoundError: If the file at text_file_path does not exist.

        Returns:
            list[str]: The text from the file at text_file_path.
        """
        if not Path(text_file_path).exists():
            raise FileNotFoundError(f"File not found: {text_file_path}")

        with open(text_file_path, "r", encoding=self.config.input_text_encoding) as f:
            text = f.read() if not return_list else f.readlines()

        return text

    def __load_data(self) -> list[str]:
        """Load the data from the file at self.eval_file_path into memory.
        """
        self._logger.info(f"Loading eval data from {self.eval_file_path}")
        min_tokens = self.config.minimum_tokens_per_sentence
        corpus = [
            line for line in self.__read_text(self.eval_file_path, return_list=True)
            if len(line.split()) > min_tokens and not line.isspace()
        ]
        self._logger.info(f"Loaded {len(corpus)} sentences from {self.eval_file_path}")
        return corpus

    def __texts_to_sequences(self, texts: list[str]) -> Sequence[dict[str, torch.tensor]]:
        """Convert a list of texts to a list of sequences of token ids.
        For example, the text "Hello world" might be converted to the sequence [0, 123, 456].
        This would be represented as:

        examples = [
            { input_ids: [0, 123, 456]},
        ]

        """
        self._logger.info("Converting texts to sequences of token ids.")
        encoded_input = self.tokenizer(
            texts,
            max_length=self.tokenizer.model_max_length,
            add_special_tokens=True,
            truncation=True
        )
        examples = np.array(
            [
                {"input_ids": torch.tensor(ids, dtype=torch.long)}
                for ids in encoded_input["input_ids"]
            ]
        )
        self._logger.info("Converting texts to sequences of token ids done.")
        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.tensor]:
        """Obtain one example of data using its array index
        """
        return self.examples[index]


class XLMRTrainingDataset(torch_data.Dataset):
    """
    Custom torch dataset that loads all training text files from a directory and implements random sampling on each
    language to train on. Following XLM-R, we train on one language per batch.

    Note that, because of the small size of our data, we decide to hold all of it in memory - this might not be the
    best approach for large datasets.
    """

    def __init__(self,
                 tokenizer: XLMRobertaTokenizerFast,
                 train_data_dir: str,
                 train_file_pattern: str,
                 batch_size: int,

                 config: XLMRCommonConfig,
                 lang_sampling_factor: Union[int, float] = 1.0,
                 ):
        self.config: XLMRCommonConfig = config
        self._class_name = self.__class__.__name__
        self.logger = create_file_logger(
            log_file=Path(config.experiment_path).joinpath("logs",
                                                           f"{__name__}.{self._class_name}.log").as_posix())
        self.logger.propagate = False
        self.logger.info(f"Initialising training dataset..")

        self.lang_sampling_factor = lang_sampling_factor
        self.batch_size = batch_size * self.config.gpu_count
        self.data_seed = self.config.random_seed
        self.sampling_counter = 0
        self.worker_id = -1
        self.examples = {}
        self.languages: list[str] = []
        self.language_data_index_mapping = {}
        self.num_examples_per_language = OrderedDict()
        self._load_data(tokenizer,
                        train_data_dir,
                        train_file_pattern,
                        self.config.minimum_tokens_per_sentence)
        self._set_language_sampling_probabilities()

    def set_worker_id(self, worker_id: int) -> None:
        self.worker_id = worker_id

    # region Helper methods
    def _load_data(self,
                   tokenizer: XLMRobertaTokenizerFast,
                   train_data_dir: str,
                   train_file_pattern: str,
                   min_tokens: int):
        self.logger.info(f"Loading train data from {train_data_dir} using pattern {train_file_pattern}")
        file_paths: list[Path] = list(Path(train_data_dir).resolve().glob("train.*"))
        if len(file_paths) == 0:
            raise FileNotFoundError(f"No training files found in {train_data_dir} using pattern {train_file_pattern}")
        for file_path in file_paths:
            language = file_path.suffix.replace(".", "")
            self.logger.info(f"Loading training data for language {language}")
            lines = [
                line
                for line in file_path.read_text(encoding="utf-8").splitlines()
                if len(line.split()) > min_tokens and not line.isspace()
            ]

            encoding = tokenizer(
                lines,
                max_length=tokenizer.model_max_length,
                add_special_tokens=True,
                truncation=True,
            )

            inputs = np.array(
                [
                    {"input_ids": torch.tensor(ids, dtype=torch.long)}
                    for ids in encoding["input_ids"]
                ]
            )

            self.num_examples_per_language[language] = len(inputs)
            self.examples[language] = inputs
            self.languages.append(language)
        self.logger.info(f"Training data successfully loaded: {self.num_examples_per_language}")

    def _create_language_index_mapping(self) -> None:
        """Shuffle the sequence positions per langugage
        Example:
            {venda: [1, 2, 3], sepedi: [4, 5, 6]} -> {venda: [3, 1, 2], sepedi: [6, 4, 5]}
        """
        self.language_data_index_mapping = {}
        for language in self.languages:
            num_examples = len(self.examples[language])
            language_index_mapping = list(range(num_examples))
            np.random.shuffle(language_index_mapping)
            self.language_data_index_mapping[language] = language_index_mapping

    def _sample_next_language(self) -> None:
        """Pick a random language to be used to sample the next bactch
        """
        if not self.languages:  # check if all language examples have been exhausted
            self.logger.info(
                f"Worker {self.worker_id}: All language examples exhausted, recreating variables..."
            )
            self._refresh()
        sampled_language_index = np.argmax(np.random.multinomial(1, self.language_probs))
        self.batch_language: str = self.languages[sampled_language_index]

    def _set_language_sampling_probabilities(self) -> None:
        """Initialize the sampling probabilities of languages based on the number of sentences for each language.

        We use this to control the order of batch languages seen by the model. Ideally, we want to
        maintain a diverse order of batch languages as much as possible The most diverse order is
        acheived by setting the `lang_sampling_factor` to 1.0
        """
        if self.lang_sampling_factor <= 0:
            # if sampling factor is negative or set to 0, we sample following a uniform distribution
            self.language_probs = [
                1 / len(self.num_examples_per_language) for _ in self.num_examples_per_language
            ]
            return

        total_num_examples = len(self)
        probs = np.array(
            [
                (value / total_num_examples) ** self.lang_sampling_factor
                for value in self.num_examples_per_language.values()
            ]
        )
        self.language_probs = list(probs / probs.sum())
        self.logger.info(
            f"Language probs created as:\n {dict(zip(self.num_examples_per_language.keys(), self.language_probs))}"
        )

    def _get_random_index(self) -> int:
        """Inisde the current language, get a random index to use in the next batch
        """
        try:
            return self.language_data_index_mapping[self.batch_language].pop()
        except (IndexError, KeyError):
            del self.language_probs[self.languages.index(self.batch_language)]
            del self.languages[self.languages.index(self.batch_language)]
            prev_batch_lang = self.batch_language
            self._sample_next_language()
            msg = f"Worker {self.worker_id}: All data examples exhausted for language: {prev_batch_lang}." \
                  f" Newly sampled batch language set as: {self.batch_language}"
            self.logger.info(msg)
            return self._get_random_index()

    def _refresh(self) -> None:
        """Called when all examples have been seen by trainer - this will shuffle the language order and reset the
        language sampling probabilities
        """
        self.logger.info(f"Worker {self.worker_id}: Refreshing training data...")
        self.languages = list(self.num_examples_per_language.keys())
        self._set_language_sampling_probabilities()
        self._create_language_index_mapping()
        self.logger.info(f"Worker {self.worker_id}: Refreshing training data...Done")

    # endregion

    def __len__(self) -> int:
        return sum(len(input_) for input_ in self.examples.values())

    def __getitem__(self, _) -> dict[str, torch.tensor]:
        """Return a sentence from a random language encoded as a sequence of token ids.
        """
        if self.sampling_counter % self.batch_size == 0:
            self._sample_next_language()
            current_batch_no = self.sampling_counter // self.batch_size
            self.logger.debug(
                f"Worker {self.worker_id} : Language sampled for batch {current_batch_no} is {self.batch_language}"
            )
        self.sampling_counter += 1
        next_sentence_index = self._get_random_index()
        return self.examples[self.batch_language][next_sentence_index]


# endregion


# region Custom Trainer

class CustomHuggingFaceTrainer(Trainer):
    """A simple wrapper of the  huggingface Trainer class"""

    def __init__(self, **kwargs) -> None:
        super(CustomHuggingFaceTrainer, self).__init__(**kwargs)

    def get_train_dataloader(self) -> DataLoader:
        """Initialise data loader which will transform the training dataset into batches."""
        assert self.train_dataset is not None, "Trainer: training requires a train_dataset."
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,  # Retrieve the batch size from TrainingArguments object
            sampler=self._get_train_sampler(),
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,  # Whether to drop the last incomplete batch, if the
            # dataset size is not divisible by the batch size.
            num_workers=self.args.dataloader_num_workers,
            worker_init_fn=self.worker_init_fn,
        )

    @staticmethod
    def get_partitioned_data_for_worker(examples: dict[str, np.ndarray],
                                        num_workers: int,
                                        worker_id: int) -> tuple[dict[str, np.ndarray], dict[str, int]]:
        """
        Partitions the training data examples for each worker in a multi-worker training setup.

        This function is used to divide the training data examples evenly among multiple workers.
        Each worker gets a unique subset of the data examples based on its worker_id.
        The function calculates the number of examples each worker should get, and then slices the input examples
        accordingly (Note that this requires the full dataset to be loaded in memory).

        Args:
            examples (dict[str, np.ndarray]): A dictionary where each key is a language and the value is a numpy array
            of examples for that language.
            num_workers (int): The total number of workers involved in the training.
            worker_id (int): The unique identifier for the current worker. This is used to slice the examples.

        Returns:
            Tuple[dict[str, np.ndarray], dict[str, int]]: A tuple containing two dictionaries.
            - The first dictionary has the same structure as the input 'examples' but contains only the examples for the
             current worker.
            - The second dictionary contains the count of examples for each language that the current worker will
            process as well as the global total count under the key 'total'.
        """
        shard = {}
        shard_stats = {}
        # This basically ensures that each worker gets the same number of examples for each language
        # e.g. If we have 3 workers and 15 examples for zulu, each worker will get 5 examples
        for language, inputs in examples.items():
            num_examples_per_worker = math.ceil(len(inputs) / num_workers)
            begin_index, end_index = (
                num_examples_per_worker * worker_id,
                num_examples_per_worker * (worker_id + 1),
            )
            shard[language] = inputs[begin_index:end_index]
            shard_stats[language] = len(shard[language])

        shard_stats["total"] = sum(shard_stats.values())
        return shard, shard_stats

    def worker_init_fn(self, worker_id: int) -> None:
        """Assign the partitioned examples to each worker and set a unique random seed for each
        """
        # Trying to set a unique random seed for each worker
        # without this, all workers will have the same random seed wh
        np.random.seed(np.random.get_state()[1][0] + worker_id + random.randint(1, 1000))

        worker_info = torch.utils.data.get_worker_info()
        # noinspection PyUnresolvedReferences
        worker_info.dataset.set_worker_id(worker_id)

        # partition the dataset for current worker
        worker_info.dataset.examples, shard_stats = self.get_partitioned_data_for_worker(
            worker_info.dataset.examples, worker_info.num_workers, worker_id)

        print(f"Stats for shard created for worker {worker_id}: \n {shard_stats}")

        # shuffle index of training examples per language for current
        # noinspection PyUnresolvedReferences
        worker_info.dataset.create_language_index_mapping()


class XLMRTrainingOrchestrator:
    """Driver for the custom Trainer class that allows for easy training of a foundation XLM-R model."""

    __slots__ = [
        "common_cfg",
        "data_config",
        "model_config",
        "train_config",
        "_logger",
        "_tokenizer",
        "_tokenizer_path",
        "_data_collator",
        "train_dataset",
        "eval_dataset",
        "model_path",
        "model",
        "trainer"
    ]

    def __init__(self, config: XLMRMasterConfig) -> None:
        experiment_path = config.common_config.experiment_path
        self.common_cfg: XLMRCommonConfig = config.common_config

        self.data_config: XLMRDatasetConfig = config.data_config
        self.model_config: dict[str, Any] = config.model_config.to_dict()
        self.train_config: TrainingArguments = config.training_config
        self.train_config.output_dir = experiment_path

        self.model_path = experiment_path
        self.model: Optional[XLMRobertaForMaskedLM] = None
        self.trainer: Optional[CustomHuggingFaceTrainer] = None

        self._tokenizer_path = config.overrides.get('tokenizer_path') or self.model_config.pop('tokenizer_path')
        self._tokenizer: Optional[PreTrainedTokenizer] = None
        self._data_collator: Optional[DataCollatorForLanguageModeling] = None
        self.train_dataset: Optional[XLMRTrainingDataset] = None
        self.eval_dataset: Optional[XLMREvaluationDataset] = None
        self._logger = create_file_logger(os.path.join(experiment_path, "training.log"), log_to_stdout=True)

        # modifying huggingface logger to log into a file
        huggingface_logger = transformers.logging.get_logger()
        file_handler = logging.FileHandler(os.path.join(experiment_path, "huggingface.log"))
        file_handler.setLevel(level=logging.DEBUG)
        huggingface_logger.addHandler(file_handler)

        # random seeds for reproducibility
        np.random.seed(self.common_cfg.random_seed)
        torch.manual_seed(self.common_cfg.random_seed)

        self._logger.info(f"XLMRTrainingOrchestrator.init completed ok. Using config:\n{config}")

    # region Properties

    @property
    def tokenizer(self) -> XLMRobertaTokenizerFast:
        if self._tokenizer is None:
            tokenizer_checkpoint = self._tokenizer_path
            is_from_local = os.path.exists(tokenizer_checkpoint)
            self._logger.info(f"Loading pretrained tokenizer from {tokenizer_checkpoint=} (local: {is_from_local=})")

            self._tokenizer: XLMRobertaTokenizerFast = XLMRobertaTokenizerFast.from_pretrained(
                pretrained_model_name_or_path=tokenizer_checkpoint,
                model_max_length=self.model_config["max_length"],
                local_files_only=is_from_local)

            self._logger.info(f"Tokenizer instantiated with {self._tokenizer.model_max_length=},"
                              f" and {self._tokenizer.vocab_size=}")
        return self._tokenizer

    @property
    def data_collator(self) -> DataCollatorForLanguageModeling:
        """This instance will take care of batching and dynamic padding of the input sentences to make sure
        each batch has the same sequence length. It will also apply the masking strategy to the input data.

        Returns:
            DataCollatorForLanguageModeling: An instance of the data collator class.
        """

        if self._data_collator is None:
            self._logger.info("Instantiating data collator")
            if self.common_cfg.use_whole_word_mask:
                self._logger.debug("_set_data_collator_class(): Using whole word masking...")
                # Use if you want to mask whole words at a time - useful for agglutinative languages
                collator_class = DataCollatorForWholeWordMask
            else:
                self._logger.debug("_set_data_collator_class(): Using standard masking...")
                collator_class = DataCollatorForLanguageModeling

            self._data_collator = collator_class(
                tokenizer=self.tokenizer,
                mlm=True,  # Whether to use train_masked language modeling as opposed to train_causal language modeling
                mlm_probability=self.common_cfg.mlm_probability)
            self._logger.info(f"Data collator instantiated with mlm_probability: {self.common_cfg.mlm_probability}")
        return self._data_collator

    # endregion

    # region Helper methods
    def _build_datasets(self) -> None:
        self._logger.info("Building datasets running...")
        self.train_dataset = XLMRTrainingDataset(
            tokenizer=self.tokenizer,
            train_data_dir=self.data_config.train,
            batch_size=self.train_config.per_device_train_batch_size,
            lang_sampling_factor=self.common_cfg.language_sampling_factor,
            train_file_pattern=self.data_config.train_file_pattern,
            config=self.common_cfg
        )
        self._logger.info(f"No. of training sentences: {len(self.train_dataset)}")

        self.eval_dataset = XLMREvaluationDataset(tokenizer=self.tokenizer,
                                                  eval_file_path=self.data_config.eval_all,
                                                  config=self.common_cfg)
        self._logger.info(f"No. of evaluation sentences: {len(self.eval_dataset)}")
        self._logger.info("Building datasets done.")

    def __save_final_training_results(self, train_results):
        """
        Saves the final training results to a file.
        """
        self._logger.info(f"Saving training results..")
        train_results_file = os.path.join(self.train_config.output_dir, "train_results.txt")
        with open(train_results_file, "w") as writer:
            self._logger.info("***** Train results *****")
            for key, value in sorted(train_results.metrics.items()):
                self._logger.info(f"  {key} = {value}")
                writer.write(f"{key} = {value}\n")
        self._logger.info(f"Saving training results done!")

    def _evaluate_helper(self,
                         eval_dataset: Optional[Dataset] = None,
                         language: str = "all") -> None:
        """Helper function to evaluate the perplexity of the model on a given dataset.
        The results are persisted to a file in the experiment directory.
        """
        eval_output = self.trainer.evaluate(eval_dataset)
        eval_output["perplexity"] = math.exp(eval_output["eval_loss"])

        output_eval_file = os.path.join(self.train_config.output_dir, language + "_eval.txt")
        with open(output_eval_file, "w") as writer:
            self._logger.info(f"***** {language} eval results *****")
            for key, value in sorted(eval_output.items()):
                self._logger.info(f"  {key} = {value}")
                writer.write(f"{key} = {value}\n")

    def _can_resume_training(self):
        train_dir_exists = os.path.exists(self.train_config.output_dir)
        config_exists = Path(self.train_config.output_dir).joinpath("config.json").exists()
        model_exists = Path(self.train_config.output_dir).joinpath("pytorch_model.bin").exists()
        return train_dir_exists and config_exists and model_exists

    def _is_done_training(self):
        training_eval_results_file = os.path.join(self.train_config.output_dir, "all_eval.txt")
        if not os.path.exists(training_eval_results_file):
            return False

        self._logger.info(
            f"Traing eval results file exists: {training_eval_results_file}. "
            f"Checking how many epochs have been completed...")
        with open(training_eval_results_file, "r") as reader:
            for line in reader:
                if "epoch" in line:
                    try:
                        epochs = float(line.split("=")[1].strip())
                        self._logger.info(f"Training has already completed {epochs} epochs")
                        return epochs >= self.train_config.num_train_epochs or epochs > 10  # Minimum 10 epochs
                    except TypeError:
                        self._logger.exception(f"Error parsing epoch number from line: {line}")
                        break
        return False

    def _prepare_for_resume_training_or_train_new(self) -> None:
        """Prepares the model for training.

        - Checks if a local training checkpoint exists and if so, it resumes training from that checkpoint.
        - Otherwise if the train_from_scratch flag is set to True, it will train a new model from scratch.
        - If the train_from_scratch flag is set to False, train from a default Huggingface Hub checkpoint
        """
        if self.common_cfg.resume_training or self._can_resume_training():
            self.model_path = self.train_config.output_dir
            self._logger.info(f"Training will resume from local {self.model_path}")
            self.model = XLMRobertaForMaskedLM.from_pretrained(self.model_path)
            self._logger.info(
                f"Model loaded from {self.model_path} with num parameters: {self.model.num_parameters()}")

        elif self.common_cfg.train_from_scratch:
            self._logger.info("Training from scratch...")
            self.model_config["vocab_size"] = self.tokenizer.vocab_size
            # TODO: Magic number 2?? What is this for? For special tokens?
            self.model_config["max_position_embeddings"] = self.model_config["max_length"] + 2

            # Initialise a brand new model with random weights
            self._logger.info(f"Initialising brand new language model from config:\n{self.model_config}")
            xlm_roberta_config = XLMRobertaConfig(**self.model_config)
            self.model = XLMRobertaForMaskedLM(xlm_roberta_config)
            self._logger.info(f"Model initialised ok. TraininableParameters={self.model.num_parameters()}")

            # Tell Trainer.train() that there is no model to resume from
            self.model_path = None

        else:  # Used if we want to continue training a model from huggingface library e.g. castorini/afriberta_base
            default_hf_checkpoint = self.common_cfg.default_xlm_checkpoint
            self._logger.info(f"Resume training on a remote {default_hf_checkpoint=}")
            self._logger.debug(f"Initialising tokenizer from pretrained {default_hf_checkpoint} checkpoint")
            self._tokenizer = XLMRobertaTokenizerFast.from_pretrained(default_hf_checkpoint)
            self._logger.debug(f"Initialising model from pretrained {default_hf_checkpoint} checkpoint")
            self.model = XLMRobertaForMaskedLM.from_pretrained(default_hf_checkpoint)

    # endregion

    def train(self) -> None:
        self._logger.info("Getting ready to start training")
        if self._is_done_training():
            self._logger.info("Training has already completed. Skipping training...")
            return

        # This will load the datasets, model and tokenizer that will be used for training
        self._prepare_for_resume_training_or_train_new()
        self._build_datasets()
        self._logger.info("Training pre-requisites tasks completed")

        if self.common_cfg.evaluate_only:
            self._logger.info("Abort trraining because evaluate_only is set to True")
            self.evaluate()
            return

        self._logger.debug(f"Setting up trainer")
        # train_config_dict = self.train_config.to_dict()
        # # There is a weird bug where this gets converted to number and causes Huggingface
        # # to throw an error. So we convert it back to string
        # train_config_dict["log_level"] = self.common_cfg.log_level.lower()
        # train_config_dict["log_level_replica"] = self.common_cfg.log_level.lower()
        # train_config_dict.pop("_n_gpu")  # Bug: Why/Where is this added to the config?
        training_args = self.train_config
        self.trainer = CustomHuggingFaceTrainer(
            model=self.model,
            args=training_args,
            data_collator=self.data_collator,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=10)]
        )
        self._logger.debug(f"Trainer setup ok")

        self._logger.info(f"Starting training xlm model using args:\n{training_args}")
        train_results = self.trainer.train(model_path=self.model_path)
        self._logger.info(f"Training completed: Type of train_results: {type(train_results)}")

        self.__save_final_training_results(train_results)

        self._logger.info("Saving model and model state...")
        self.trainer.save_model()  # if push_to_hub is True, this will also push the model to huggingface
        self.trainer.state.save_to_json(os.path.join(training_args.output_dir,
                                                     "trainer_state.json"))
        if not self.common_cfg.train_only:
            self.evaluate()
        else:
            self._logger.info("Skipping evaluation as train_only is set to True")

    def evaluate(self) -> None:
        self._logger.info("Evaluating model on all languges")
        self._evaluate_helper(self.eval_dataset, "all")
        self._logger.info("Done evaluating on all languages")

        self._logger.info("Evaluating model on per language datasets")
        eval_dataset_path = Path(self.data_config.eval_for_lang)
        eval_file_pattern = self.data_config.eval_file_pattern
        self._logger.debug(
            f"Search for evaluation data from eval_dataset_path: {eval_dataset_path} with pattern: {eval_file_pattern}")

        eval_file_paths = eval_dataset_path.glob(eval_file_pattern)
        # This assumes there is only one file per language
        # Otherwise we would need a group by language step before evaluating..
        for file_path in eval_file_paths:
            language = file_path.suffix.replace(".", "")
            dataset = XLMREvaluationDataset(tokenizer=self.tokenizer,
                                            eval_file_path=str(file_path),
                                            config=self.common_cfg)
            self._logger.info(f"Evaluating {language} with {file_path}...")
            self._evaluate_helper(dataset, language)

        self._logger.info("Model evaluation done!")

# endregion
