# Style Profiles

Profiles are generated from engineered features derived from data/processed/final responses.

## gemini-flash-latest

Top descriptive differences:
- `text_punctuation_density` is higher than other models (effect size: 1.903).
- `text_newline_count` is higher than other models (effect size: 1.343).
- `text_paragraph_count` is higher than other models (effect size: 1.267).
- `text_sentence_count` is higher than other models (effect size: 1.216).
- `text_punctuation_count` is higher than other models (effect size: 1.052).
- `text_repeated_trigram_ratio` is lower than other models (effect size: -0.766).
- `text_repeated_bigram_ratio` is lower than other models (effect size: -0.722).
- `text_negation_word_count` is higher than other models (effect size: 0.697).

Top SHAP features for this model-key classifier class:
- `text_punctuation_density` (mean absolute SHAP: 0.060656).
- `text_punctuation_count` (mean absolute SHAP: 0.046243).
- `text_unique_word_count` (mean absolute SHAP: 0.033674).
- `text_repeated_bigram_ratio` (mean absolute SHAP: 0.030524).
- `text_entropy` (mean absolute SHAP: 0.026927).

## llama-3.1-8b-groq

Top descriptive differences:
- `text_hapax_ratio` is lower than other models (effect size: -1.010).
- `text_repeated_bigram_ratio` is higher than other models (effect size: 0.945).
- `text_repetition_rate` is higher than other models (effect size: 0.925).
- `text_type_token_ratio` is lower than other models (effect size: -0.925).
- `text_entropy` is lower than other models (effect size: -0.905).
- `text_unique_word_count` is lower than other models (effect size: -0.669).
- `text_avg_word_frequency` is higher than other models (effect size: 0.646).
- `text_repeated_trigram_ratio` is higher than other models (effect size: 0.640).

Top SHAP features for this model-key classifier class:
- `text_hapax_ratio` (mean absolute SHAP: 0.048889).
- `text_unique_word_count` (mean absolute SHAP: 0.037376).
- `text_entropy` (mean absolute SHAP: 0.035286).
- `text_type_token_ratio` (mean absolute SHAP: 0.032432).
- `text_repetition_rate` (mean absolute SHAP: 0.031641).

## mistral-7b-hf

Top descriptive differences:
- `text_punctuation_density` is lower than other models (effect size: -0.946).
- `text_newline_count` is lower than other models (effect size: -0.750).
- `text_word_count` is lower than other models (effect size: -0.740).
- `text_punctuation_count` is lower than other models (effect size: -0.721).
- `text_char_count` is lower than other models (effect size: -0.715).
- `source_response_length` is lower than other models (effect size: -0.710).
- `text_paragraph_count` is lower than other models (effect size: -0.603).
- `text_unique_word_count` is lower than other models (effect size: -0.598).

Top SHAP features for this model-key classifier class:
- `text_punctuation_density` (mean absolute SHAP: 0.054178).
- `text_punctuation_count` (mean absolute SHAP: 0.040437).
- `text_hapax_ratio` (mean absolute SHAP: 0.019739).
- `text_type_token_ratio` (mean absolute SHAP: 0.017617).
- `text_unique_word_count` (mean absolute SHAP: 0.015527).

## phi-3-mini-hf

Top descriptive differences:
- `text_hapax_ratio` is higher than other models (effect size: 1.052).
- `text_punctuation_density` is lower than other models (effect size: -1.027).
- `text_repetition_rate` is lower than other models (effect size: -0.921).
- `text_type_token_ratio` is higher than other models (effect size: 0.921).
- `text_colon_density` is lower than other models (effect size: -0.917).
- `text_avg_word_length` is higher than other models (effect size: 0.829).
- `text_avg_sentence_chars` is higher than other models (effect size: 0.754).
- `text_avg_sentence_words` is higher than other models (effect size: 0.726).

Top SHAP features for this model-key classifier class:
- `text_hapax_ratio` (mean absolute SHAP: 0.033886).
- `text_repetition_rate` (mean absolute SHAP: 0.027621).
- `text_punctuation_density` (mean absolute SHAP: 0.027120).
- `text_type_token_ratio` (mean absolute SHAP: 0.026727).
- `source_response_length` (mean absolute SHAP: 0.022359).
