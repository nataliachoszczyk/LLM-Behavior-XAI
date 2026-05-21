# LLM Behavior XAI

This repository contains tools and notebooks for analyzing behavioral differences
across large language models (LLMs). It provides pipelines to collect model
responses from multiple providers, merge and validate datasets, extract
linguistic and stylistic features, and generate visualizations and XAI artifacts.

Main goals:
- Compare response style and statistical features across models and providers
- Produce feature sets for downstream modeling and explainability (XAI)
- Provide dashboards and notebooks for exploration and reporting

---

## Requirements

- Python 3.12
- Git
- Optional: GPU for running large local models (Hugging Face)

Dependencies are listed in `requirements.txt` and `pyproject.toml`.

## Quick start

1. Clone the repository.

2. Create and activate a virtual environment.

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Linux / macOS:

```bash
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:

```bash
make requirements
# or
pip install -r requirements.txt
```

## Configuration

Project configuration and default paths are stored in `config.py`.
Sensitive keys (API tokens for Gemini, Groq, or Hugging Face) must be provided
via environment variables or a `.env` file. Common variables used:

- `GEMINI_API_KEY_*`
- `GROQ_API_KEY`
- `HF_TOKEN`

Important behavior flags:
- `RUN_MODE` — controls whether pipelines write timestamped temporary files (`full`) or use fixed paths.

## Main scripts and Makefile targets

The `Makefile` includes common tasks. Key targets and the equivalent module calls:

- Collect LLM responses (runs collector pipeline for train/val/test prompts):

```bash
make collect_responses
# or
python -m llm_behavior_xai.llm_response_collector.main
```

- Merge and validate collected responses:

```bash
make merge_and_validate_responses
# or
python -m llm_behavior_xai.llm_response_merger_and_validator.main
```

- Analyze responses (extract features and generate plots):

```bash
make analyze_responses
# or
python -m llm_behavior_xai.llm_response_analyzer.main
```

- Run tests:

```bash
make test
```

- Run local Streamlit dashboard:

```bash
make dashboard
# or
python -m streamlit run llm_behavior_xai/dashboard/app.py
```

## Repository layout

- `llm_behavior_xai/` — package containing the implementation:
    - `llm_response_collector/` — collector CLI and pipelines
    - `llm_response_merger_and_validator/` — merging, cleaning and validation
    - `llm_response_analyzer/` — feature extraction, plots, and XAI helpers
    - `dashboard/` — Streamlit app for visual exploration
- `data/` — prompts, temporary outputs and processed datasets
- `models/` — stored XAI models and metadata (`models/xai/`)
- `notebooks/` — Jupyter notebooks for analysis and experiments
- `reports/` — generated reports and style profiles

## Data paths (defined in `config.py`)

- `PROMPTS_DIR` — path to CSV prompts (train/val/test)
- `PROCESSED_DATA_DIR` — base for processed outputs
- `FINAL_RESPONSES_DIR` — final merged responses
- `RESPONSE_FEATURES_DIR` — CSVs with extracted features and plots

Refer to `config.py` for exact filenames and temporary dataset paths.

## Notes and best practices

- The full pipeline may require network access and API keys to collect responses.
- Running local HF models (e.g., Mistral, Phi) may need sufficient VRAM/GPU.
- Keep API keys out of version control — use `.env` or environment variables.
- Use `RUN_MODE = "full"` in `config.py` to generate timestamped temporary results.
