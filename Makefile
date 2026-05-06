#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = LLM-Behavior-XAI
PYTHON_VERSION = 3.12
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	@rm -rf venv
	$(PYTHON_INTERPRETER)$(PYTHON_VERSION) -m venv venv
	@echo ">>> New python interpreter environment created. Activate it using 'source venv/bin/activate'"


## Install Python dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	$(PYTHON_INTERPRETER) -m nltk.downloader wordnet
	$(PYTHON_INTERPRETER) -m nltk.downloader punkt
	$(PYTHON_INTERPRETER) -m nltk.downloader punkt_tab
	$(PYTHON_INTERPRETER) -m nltk.downloader vader_lexicon
	$(PYTHON_INTERPRETER) -m spacy download pl_core_news_sm
	$(PYTHON_INTERPRETER) -m spacy download en_core_web_sm
	

## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check


## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format


## Run tests
.PHONY: test
test:
	python -m pytest tests


#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
