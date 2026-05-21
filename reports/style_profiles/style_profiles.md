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
- `text_punctuation_density` (mean absolute SHAP: 0.060966).
- `text_punctuation_count` (mean absolute SHAP: 0.046104).
- `text_unique_word_count` (mean absolute SHAP: 0.033797).
- `text_repeated_bigram_ratio` (mean absolute SHAP: 0.030643).
- `text_repeated_trigram_ratio` (mean absolute SHAP: 0.026237).

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
- `text_hapax_ratio` (mean absolute SHAP: 0.048931).
- `text_unique_word_count` (mean absolute SHAP: 0.038269).
- `text_entropy` (mean absolute SHAP: 0.034574).
- `text_type_token_ratio` (mean absolute SHAP: 0.033050).
- `text_repetition_rate` (mean absolute SHAP: 0.031920).

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
- `text_punctuation_density` (mean absolute SHAP: 0.054105).
- `text_punctuation_count` (mean absolute SHAP: 0.040892).
- `text_hapax_ratio` (mean absolute SHAP: 0.019022).
- `text_type_token_ratio` (mean absolute SHAP: 0.017363).
- `text_unique_word_count` (mean absolute SHAP: 0.015888).

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
- `text_hapax_ratio` (mean absolute SHAP: 0.032358).
- `text_type_token_ratio` (mean absolute SHAP: 0.027039).
- `text_repetition_rate` (mean absolute SHAP: 0.026822).
- `text_punctuation_density` (mean absolute SHAP: 0.026561).
- `source_response_length` (mean absolute SHAP: 0.023453).
