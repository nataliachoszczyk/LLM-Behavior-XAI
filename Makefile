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


## Check type hints with mypy
.PHONY: mypy
mypy:
	mypy llm_behavior_xai tests config.py file_utils.py


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


## Run not slow tests
.PHONY: test_not_slow
test_not_slow:
	python -m pytest -m "not slow" tests


## Run slow tests
.PHONY: test_slow
test_slow:
	python -m pytest -m slow tests


## Run tests with coverage
.PHONY: coverage
coverage:
	$(PYTHON_INTERPRETER) -m pytest --cov=llm_behavior_xai --cov-report=term-missing tests


#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Collect LLM responses with the response collector
.PHONY: collect_responses
collect_responses:
	$(PYTHON_INTERPRETER) -m llm_behavior_xai.llm_response_collector.main


## Analyze LLM responses with the response analyzer
.PHONY: analyze_responses
analyze_responses:
	$(PYTHON_INTERPRETER) -m llm_behavior_xai.llm_response_analyzer.main


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
