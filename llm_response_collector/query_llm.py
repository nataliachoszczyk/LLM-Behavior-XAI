import time
import math
import torch

from google.genai import types

from config import MODELS_CONFIG


def query_gemini(prompt_text, config, gemini_client):
    generation_config = types.GenerateContentConfig(
        temperature=config["temperature"],
        max_output_tokens=config["max_tokens"],
        top_p=config.get("top_p", 0.95),
    )

    configured = config.get("model_id", "gemini-flash-latest")

    if configured.startswith("models/"):
        configured_short = configured.replace("models/", "", 1)
    else:
        configured_short = configured

    model_candidates = [
        configured,
        configured_short,
        f"models/{configured_short}",
        "gemini-flash-latest",
        "models/gemini-flash-latest",
        "gemini-2.0-flash",
        "models/gemini-2.0-flash",
    ]
    last_error = None

    for model_id in dict.fromkeys(model_candidates):
        try:
            response = gemini_client.models.generate_content(
                model=model_id,
                contents=prompt_text,
                config=generation_config,
            )
            text = (response.text or "").strip()

            if not text:
                return None, "Empty Gemini response", summarize_logprobs([])

            finish = getattr(response.candidates[0], "finish_reason", None)

            if str(finish) not in ("STOP", "FinishReason.STOP", "1"):
                print(f"  ⚠️ finish_reason={finish} (response may have been truncated)")

            return text, None, summarize_logprobs([])
        except Exception as e:
            err = str(e)
            last_error = err

            if "NOT_FOUND" not in err and "not found" not in err.lower():
                return None, err, summarize_logprobs([])

    return None, last_error or "Gemini request failed", summarize_logprobs([])


def query_groq(prompt_text, config, groq_client):
    try:
        request = {
            "model": config["model_id"],
            "messages": [{"role": "user", "content": prompt_text}],
            "temperature": config["temperature"],
            "top_p": config.get("top_p", 1.0),
            "max_tokens": config["max_tokens"],
            "logprobs": True,
            "top_logprobs": 1,
        }

        try:
            completion = groq_client.chat.completions.create(**request)
        except Exception as e:
            if "logprob" not in str(e).lower():
                raise

            request.pop("logprobs", None)
            request.pop("top_logprobs", None)
            completion = groq_client.chat.completions.create(**request)

        choice = completion.choices[0]
        response_text = (choice.message.content or "").strip()

        token_logprobs = []
        choice_logprobs = getattr(choice, "logprobs", None)

        if choice_logprobs is not None:
            content_items = getattr(choice_logprobs, "content", None)

            if content_items is not None:
                for item in content_items:
                    lp = getattr(item, "logprob", None)

                    if lp is not None:
                        token_logprobs.append(float(lp))

        return response_text, None, summarize_logprobs(token_logprobs)
    except Exception as e:
        return None, str(e), summarize_logprobs([])


def query_local_hf(prompt_text, model_key, config, local_models={}):
    try:
        local_bundle = local_models[model_key]
        model = local_bundle["model"]
        tokenizer = local_bundle["tokenizer"]

        messages = [{"role": "user", "content": prompt_text}]
        prompt_for_model = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(prompt_for_model, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=config["max_new_tokens"],
                temperature=config["temperature"],
                top_p=config.get("top_p", 1.0),
                repetition_penalty=config.get("repetition_penalty", 1.0),
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )

        prompt_len = int(inputs["input_ids"].shape[-1])
        generated_ids = generated.sequences[0][prompt_len:]

        response_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        token_logprobs = []

        for step, step_scores in enumerate(generated.scores[: len(generated_ids)]):
            token_id = int(generated_ids[step].item())
            log_probs = torch.log_softmax(step_scores[0], dim=-1)
            token_logprobs.append(float(log_probs[token_id].item()))

        return response_text, None, summarize_logprobs(token_logprobs)
    except Exception as e:
        return None, str(e), summarize_logprobs([])


def query_model(model_key, prompt_text, client=None, local_models=None):
    if local_models is None:
        local_models = {}

    config = MODELS_CONFIG[model_key]

    start_time = time.time()

    if config["provider"] == "Google Gemini API" or model_key.startswith("gemini"):
        response, error, logprob_stats = query_gemini(prompt_text, config, client)
    elif model_key == "llama-3.1-8b-groq":
        response, error, logprob_stats = query_groq(prompt_text, config, client)
    elif model_key in local_models:
        response, error, logprob_stats = query_local_hf(prompt_text, model_key, config)
    else:
        response, error, logprob_stats = None, f"Unknown model: {model_key}", summarize_logprobs([])

    elapsed = round(time.time() - start_time, 2)

    return response, error, elapsed, logprob_stats


def summarize_logprobs(token_logprobs):
    if not token_logprobs:
        return {
            "logprob_available": False,
            "sum_logprob": None,
            "avg_logprob": None,
            "generated_tokens": 0,
            "perplexity": None,
        }

    total = float(sum(token_logprobs))
    n_tokens = int(len(token_logprobs))
    avg = total / n_tokens
    ppl = float(math.exp(-avg))

    return {
        "logprob_available": True,
        "sum_logprob": round(total, 6),
        "avg_logprob": round(avg, 6),
        "generated_tokens": n_tokens,
        "perplexity": round(ppl, 6),
    }
