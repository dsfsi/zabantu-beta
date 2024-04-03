.PHONY: clean setup train_lite sync_data_to_s3 sync_data_from_s3 sync_data_to_azure sync_data_from_azure sync_data_to_gcs sync_data_from_gcs

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
BUCKET = mungana-public-assets
PROFILE = default
PROJECT_NAME = train-lowres-polyglot-llms
PYTHON_INTERPRETER = python3

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install NVIDIA Drivers and CUDA Toolkit
nvdia_setup:
	/bin/bash scripts/nvidia_setup.sh

## Install Python Dependencies + Other OS-level dependencies
setup:
	/bin/bash scripts/server_setup.sh

## Verify the environment is setup correctly
verify: test_environment
	$(PYTHON_INTERPRETER) scripts/check_gpu.py

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Train a light-weight model for demonstration purposes
train_lite:
	@echo ">>> Training a light-weight model to demonstrate the training process."
	/bin/bash scripts/train_sp_tokenizer.sh --input-texts-path demos/data/sample.bantu.ven.txt --sampled-texts-path data/temp/tshivenda-tokenizer/0 --seed 47 --alpha 0.5 --tokenizer-model-type unigram --vocab-size 5000 --tokenizer-output-dir data/tokenizers/tshivenda-xlmr-5k
	@echo ">>> Tokenizer ready, moving on to training the language model."
	/bin/bash scripts/train_masked_xlm.sh --config configs/tshivenda-xlmr-base.yml --training_data demos/data --experiment_name tshivenda-xlmr-lite --tokenizer_path data/tokenizers/tshivenda-xlmr-5k  --epochs 5

## Upload Data to S3 - Requires AWS CLI to be configured
# Authentication guide: 
sync_data_to_s3:
	dvc push -r s3

## Download Data from S3
sync_data_from_s3:
	dvc pull -r s3

## Upload data to Azure Blob Storage - 
# Authentication guide:
sync_data_to_azure:
	dvc push -r az

## Download data from Azure Blob Storage
sync_data_from_azure:
	dvc pull -r az

## Upload data to Google Cloud Storage
# Authentication guide: 
sync_data_to_gcs:
	dvc push -r gcs

## Download data from Google Cloud Storage
sync_data_from_gcs:
	dvc pull -r gcs

## Set up python interpreter environment
env:
ifeq (True,$(HAS_CONDA))
		@echo ">>> Detected conda, creating conda environment."
		conda env create -f environment.yml
		@echo ">>> New conda env created. Activate with:\nsource activate $(PROJECT_NAME)"
else
	$(PYTHON_INTERPRETER) -m pip install -q virtualenv virtualenvwrapper
	@echo ">>> Installing virtualenvwrapper if not already installed.\nMake sure the following lines are in shell startup file\n\
	export WORKON_HOME=$$HOME/.virtualenvs\nexport PROJECT_HOME=$$HOME/Devel\nsource /usr/local/bin/virtualenvwrapper.sh\n"
	@bash -c "source `which virtualenvwrapper.sh`;mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER)"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
endif

### Convert pyproject.toml to requirements.txt
freeze:
	poetry export -f requirements.txt --output requirements.txt --without-hashes

## Test python environment is setup correctly
test_environment:
	$(PYTHON_INTERPRETER) test_environment.py

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
