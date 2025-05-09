import os
os.environ['TRANSFORMERS_CACHE'] = "/data/remy/huggingface_hub"
os.environ['HF_HOME'] = "/data/remy/huggingface_hub"


import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import pycountry
import statistics
from tqdm import tqdm
import re
from pathlib import Path

import configparser

credential_file = "credentials.ini"
credential_config = configparser.ConfigParser()
credential_config.read(credential_file)

hf_access_token = credential_config['HF']["HF_API"]
from huggingface_hub import login; login(token=hf_access_token)

tqdm.pandas()


data_dir = "./output"
eurostat_data_files = [
    {
        "indicator": "income",
        "path": f"{data_dir}/eurostat_income_2017.csv",
        "complete_year": "2017",
        "unit": "EUR_HAB"
    },
    {
        "indicator": "pop_density",
        "path": f"{data_dir}/eurostat_pop_density_2018.csv",
        "complete_year": "2018",
        "unit": "PER_KM2",
        "nan": ":", # null value are ":" and not empty cell
    },
    {
        "indicator": "poverty",
        "path": f"{data_dir}/eurostat_poverty_2022.csv",
        "complete_year": "2022",
        "unit": "PC",
        "nan": ":"
    },
    {
        "indicator": "age_index",
        "path": f"{data_dir}/eurostat_age_index_2021.csv",
        "complete_year": "2021",
        "unit": "NR"
    }
]

def prediction(NUTS_ID, country, indicator, year):
    NUTS_ID = NUTS_ID
    country = country
    if indicator == "income":
        prompt = f"What is the average income per inhabitant for {NUTS_ID} in {country} in {year}?"
        messages = [
            {"role": "system", 
             "content": (
                 f'You are a statistician working for the European commission at EUROSTAT.'
                 f'You have to give the average income per inhabitant by NUTS 1 or 2 or 3 level for year {year}.'
                 f"Don\'t compute it, just guess the income. Answer only with the average income (a single number)"
                 f"without any other words or repetition of the question. Don\'t repeat the prompt neither. Example of answer: '30000'."
                )
            },
            {"role": "user", "content": prompt}
        ]
    elif indicator == "pop_density":
        prompt = f"What is the population density (in PER/km2) for {NUTS_ID} in {country} in {year}?"
        messages = [
            {"role": "system",
             "content": (
                 f"You are a statistician working for the European commission at EUROSTAT. "
                 f"You have to give the population density (nb the person per km2) by NUTS 1 or 2 or 3 level for year {year}. "
                 f"Don't compute it, just guess the density. Answer only with the density (a single number) "
                 f"without any other words or repetition of the question. Don't repeat the prompt neither. "
                 f"Example of answer: '146.3'."
                )
            },
            {"role": "user", "content": prompt}
        ]
    elif indicator == "poverty":
        prompt = f"What is the percent of the population considered as poor by Eurostat for region: {NUTS_ID} in {country} in {year}?"
        messages = [
            {"role": "system",
             "content": (
                 f"You are a statistician working for the European commission at EUROSTAT. "
                 f"You have to give the percent à poop people by NUTS 1 or 2 or 3 level for year {year}. "
                 f"Don't compute it, just guess the ratio. Answer only with the ratio (a single number) "
                 f"without any other words or repetition of the question. Don't repeat the prompt neither. "
                 f"Example of answer: '25.3'."
                )
            },
            {"role": "user", "content": prompt}
        ]
    elif indicator == "age_index":
        prompt = f"What is the percent of the population under 15 years old or older than 65 years old for region: {NUTS_ID} in {country} in {year}?"
        messages = [
            {"role": "system",
             "content": (
                 f"You are a statistician working for the European commission at EUROSTAT. "
                 f"You have to give the percent of young (<15year old) + old (>65 years) people by NUTS 1 or 2 or 3 level for year {year}. "
                 f"Don't compute it, just guess the ratio. Answer only with the ratio (a single number) "
                 f"without any other words or repetition of the question. Don't repeat the prompt neither. "
                 f"Example of answer: '25.3'."
                )
            },
            {"role": "user", "content": prompt}
        ]
    else:
        print("wrong indicator !!")

    # Tokenize input prompt
    try:
        inputs = tokenizer.apply_chat_template(
            messages, 
            tokenize=False,
            add_generation_prompt=True
            )
        inputs = tokenizer([inputs], return_tensors="pt").to("cuda")
    except: # there is no chat template into the tokenizer.json
        inputs = tokenizer(messages[0]["content"], return_tensors="pt").to("cuda")
    
    # Generate output
    outputs = model.generate(
        **inputs,
        max_length=256,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=True,
        temperature=0.3,
        # top_p=0.9,
        # top_k=50,
        return_dict_in_generate=True,
        output_scores=True,
        # repetition_penalty=1.2
    )

    # Extract generated tokens and logits
    generated_tokens = outputs.sequences[0].to("cpu")
    logits = torch.stack(outputs.scores, dim=0).to("cpu")  # Shape: (seq_length, vocab_size)

    # Compute log probabilities for each generated token
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs = log_probs.squeeze(1)
    generated_token_ids = generated_tokens[len(inputs.input_ids[0]):]  # Only new tokens: [len(inputs.input_ids[0]):] removes the prompt 
    token_log_probs = log_probs[range(len(generated_token_ids)), generated_token_ids]
    average_token_logprobs = float(torch.mean(token_log_probs))


    # Decode the generated response
    generated_text = tokenizer.decode(generated_tokens[len(inputs.input_ids[0]):], skip_special_tokens=True)
    generated_text = re.sub(r"[^\d.]", "", generated_text)
    
    #return generated_text, generated_token_ids, average_token_logprobs
    return generated_text, average_token_logprobs

def relative_prediction(NUTS_ID, country, country_indicator, indicator, year):
    prompt = f"What is the difference of income per inhabitant between {NUTS_ID} ({country}) and {country} (country average {indicator}: {country_indicator})  in {year}?"
    messages = [
        {"role": "system", "content": f'You are a statistician working for the European commission at EUROSTAT. You have to give the difference of average income per inhabitant between a sub regions at NUTS 1 or 2 or 3 level and its country for {year}. Don\'t compute it, just guess the diffrence of income. Answer only with the difference income (a single number) without any other words or repetition of the question. Don\'t repeat the prompt neither. Example of answer: 3000.'},
        {"role": "user", "content": prompt}
    ]
    if indicator == "income":
        prompt = f"What is the difference of income per inhabitant between {NUTS_ID} ({country}) and {country} (country average {indicator}: {country_indicator})  in {year}?"
        messages = [
            {"role": "system", 
             "content": (
                 f'You are a statistician working for the European commission at EUROSTAT.'
                 f'You have to give the difference of average income per inhabitant between a sub regions at NUTS 1 or 2 or 3 level and its country for {year}'
                 f"Don\'t compute it, just guess the diffrence of income. Answer only with the difference income (a single number) without any other words or repetition of the question"
                 f"without any other words or repetition of the question. Don\'t repeat the prompt neither. Example of answer: 3000."
                )
            },
            {"role": "user", "content": prompt}
        ]
    elif indicator == "pop_density":
        prompt = f"What is the difference of population density (in PER/km2) between {NUTS_ID} ({country}) and {country} (country average {indicator}: {country_indicator})  in {year}?"
        messages = [
            {"role": "system", 
             "content": (
                 f'You are a statistician working for the European commission at EUROSTAT.'
                 f'You have to give the difference of population density (in PER/km2) between a sub regions at NUTS 1 or 2 or 3 level and its country for {year}'
                 f"Don\'t compute it, just guess the diffrence of population density (in PER/km2). Answer only with the difference population density (in PER/km2) (a single number) without any other words or repetition of the question"
                 f"without any other words or repetition of the question. Don\'t repeat the prompt neither. Example of answer: 30."
                )
            },
            {"role": "user", "content": prompt}
        ]
    elif indicator == "poverty":
        prompt = f"What is the difference of percent of the population considered as poor between {NUTS_ID} ({country}) and {country} (country average {indicator}: {country_indicator})  in {year}?"
        messages = [
            {"role": "system", 
             "content": (
                 f'You are a statistician working for the European commission at EUROSTAT.'
                 f'You have to give the difference of percent of the population considered as poor between a sub regions at NUTS 1 or 2 or 3 level and its country for {year}'
                 f"Don\'t compute it, just guess the diffrence of percent of the population considered as poor. Answer only with the difference ratio (a single number) without any other words or repetition of the question"
                 f"without any other words or repetition of the question. Don\'t repeat the prompt neither. Example of answer: 30."
                )
            },
            {"role": "user", "content": prompt}
        ]
    elif indicator == "age_index":
        prompt = f"What is the difference of percent of the population under 15 years old or older than 65 years old between {NUTS_ID} ({country}) and {country} (country average {indicator}: {country_indicator})  in {year}?"
        messages = [
            {"role": "system", 
             "content": (
                 f'You are a statistician working for the European commission at EUROSTAT.'
                 f'You have to give the difference of percent of the population under 15 years old or older than 65 years old between a sub regions at NUTS 1 or 2 or 3 level and its country for {year}'
                 f"Don\'t compute it, just guess the diffrence of ratio. Answer only with the difference ratio (a single number) without any other words or repetition of the question"
                 f"without any other words or repetition of the question. Don\'t repeat the prompt neither. Example of answer: 30."
                )
            },
            {"role": "user", "content": prompt}
        ]
    else:
        print("wrong indicator !!")

    # Tokenize input prompt
    try:
        inputs = tokenizer.apply_chat_template(
            messages, 
            tokenize=False,
            add_generation_prompt=True
            )
        inputs = tokenizer([inputs], return_tensors="pt").to("cuda")
    except: # there is no chat template into the tokenizer.json
        inputs = tokenizer(messages[0]["content"], return_tensors="pt").to("cuda")
    
    # Generate output
    outputs = model.generate(
        **inputs,
        max_length=256,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=True,
        temperature=0.3,
        # top_p=0.9,
        # top_k=50,
        return_dict_in_generate=True,
        output_scores=True,
        # repetition_penalty=1.2
    )

    # Extract generated tokens and logits
    generated_tokens = outputs.sequences[0].to("cpu")
    logits = torch.stack(outputs.scores, dim=0).to("cpu")  # Shape: (seq_length, vocab_size)

    # Compute log probabilities for each generated token
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs = log_probs.squeeze(1)
    generated_token_ids = generated_tokens[len(inputs.input_ids[0]):]  # Only new tokens: [len(inputs.input_ids[0]):] removes the prompt 
    token_log_probs = log_probs[range(len(generated_token_ids)), generated_token_ids]
    average_token_logprobs = float(torch.mean(token_log_probs))


    # Decode the generated response
    generated_text = tokenizer.decode(generated_tokens[len(inputs.input_ids[0]):], skip_special_tokens=True)
    
    #return generated_text, generated_token_ids, average_token_logprobs
    return generated_text, average_token_logprobs


def average_prediction(row, indicator, year):
    """
    Compute the average and deviation of predictions for a given NUTS_ID and country.
    Expects the row to contain both 'NUTS_ID' and 'country'.
    """
    number_prediction = 10
    predictions = []
    logprobs = []

    for _ in range(number_prediction):
        prediction_value, logprob = prediction(row["NUTS_NAME"], row["country"], indicator, year)
        predictions.append(prediction_value)
        logprobs.append(logprob)

    # Handle potential NaN values in predictions
    try:
        predictions = [float(p) for p in predictions if not pd.isna(p)]
    except:
        predictions_filtred = []
        for p in predictions:
            try:
                p_ = float(p)
                predictions_filtred.append(p_)
            except:
                print(f"Could not parse: {p}")
        predictions = predictions_filtred
    logprobs = [float(p) for p in logprobs if not pd.isna(p)]
    print(f'{row["NUTS_NAME"]}/{row["country"]}: {predictions} | {logprobs}')

    if predictions:
        average = sum(predictions) / len(predictions)
        deviation = statistics.stdev(predictions) if len(predictions) > 1 else 0
    else:
        average = np.nan
        deviation = np.nan
    if logprobs:
        logprobs_average = sum(logprobs) / len(logprobs)
        logprobs_deviation = statistics.stdev(logprobs) if len(logprobs) > 1 else 0
    else:
        logprobs_average = np.nan
        logprobs_deviation = np.nan

    return average, deviation, logprobs_average, logprobs_deviation

def average_relative_prediction(row, indicator, year):
    """
.
    """
    number_prediction = 10
    predictions = []
    logprobs = []

    for _ in range(number_prediction):
        # relative_prediction(NUTS_ID, country, country_indicator, indicator, year)
        prediction_value, logprob = relative_prediction(row["NUTS_NAME"], row["country"], row["average_country_indicator"], indicator, year)
        predictions.append(prediction_value)
        logprobs.append(logprob)

    # Handle potential NaN values in predictions
    try:
        predictions = [float(p) for p in predictions if not pd.isna(p)]
    except:
        predictions_filtred = []
        for p in predictions:
            try:
                p_ = float(p)
                predictions_filtred.append(p_)
            except:
                try: # remove explanations between paranthesis (Llama-8B-instruct)
                    p_ = float(re.sub(r"\s*\(.*?\)", "", p))
                    predictions_filtred.append(p_)
                except:
                    print(f"Could not parse: {p}")
        predictions = predictions_filtred

    logprobs = [float(p) for p in logprobs if not pd.isna(p)]
    print(f'{row["NUTS_NAME"]}/{row["country"]}: {predictions} | {logprobs}')

    if predictions:
        average = sum(predictions) / len(predictions)
        filtered_predictions = [x for x in predictions if not np.isnan(x)]
        if len(filtered_predictions) > 1:
            deviation = statistics.stdev(filtered_predictions)
        else:
            deviation = np.nan
    else:
        average = np.nan                #load_in_8bit=True,  # Enable 8-bit quantization
        deviation = np.nan
    if logprobs:
        logprobs_average = sum(logprobs) / len(logprobs)
        logprobs_deviation = statistics.stdev(logprobs) if len(logprobs) > 1 else 0
    else:
        logprobs_average = np.nan
        logprobs_deviation = np.nan

    return average, deviation, logprobs_average, logprobs_deviation

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)

    try:
        parser = argparse.ArgumentParser(description="Load and configure a language model.")
        parser.add_argument(
            "--model",
            type=str,
            # default="Qwen/Qwen2.5-7B-Instruct",
            default="meta-llama/Llama-3.1-8B-Instruct",
            help="Name of the model to load. Default is 'meta-llama/Llama-3.1-8B-Instruct'."
        )
        args = parser.parse_args()
        MODEL = args.model
    except:
        # MODEL = "Qwen/Qwen2.5-7B-Instruct"
        MODEL = "meta-llama/Llama-3.1-8B-Instruct"
    model_short_name = MODEL.split('/')[-1]  # Extract short name from full model path

    print(f"Loading model: {MODEL} ({model_short_name})")

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        if "70B" in MODEL or "72B" in MODEL:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL,
                device_map="auto",
                do_sample=True,
                torch_dtype=torch.float16,
                quantization_config=BitsAndBytesConfig(load_in_4bit=True)
            )
        else: # no quantization
            model = AutoModelForCausalLM.from_pretrained(
                MODEL,
                device_map="auto",
                do_sample=True,
                torch_dtype=torch.float16,
            )
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        print("Model and tokenizer loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")

    for eurostat_dict in eurostat_data_files:
        indicator = eurostat_dict["indicator"]
        path = eurostat_dict["path"]
        year = eurostat_dict["complete_year"]

        print(f"Work on: {indicator}")
        df = pd.read_csv(path)
        df["country"] = df["CNTR_CODE"].apply(lambda code: pycountry.countries.get(alpha_2=code).name if pycountry.countries.get(alpha_2=code) else "Unknown")

        # Absolute
        file_path = Path(f'./output/bash/{indicator}_{year}_nuts_llm_{model_short_name}_absolute.csv')
        if file_path.exists():
            print("\t Absolute: already processed")
        else:
            df[[f"{indicator}_absolute_predicted", f"{indicator}_absolute_deviation", f"{indicator}_absolute_logprobs", f"{indicator}_absolute_logprobs_deviation"]] = df.progress_apply(lambda row: average_prediction(row, indicator, year), axis=1).apply(pd.Series)
            try :
                df.to_csv(f'./output/bash/{indicator}_{year}_nuts_llm_{model_short_name}_absolute.csv')
            except:
                df.to_csv(f'./output/bash/{indicator}_{year}_nuts_llm_error_on_name.csv')

        # Relative
        file_path = Path(f'./output/bash/{indicator}_{year}_nuts_llm_{model_short_name}_relative.csv')
        if file_path.exists():
            print("\t Relative: already processed")
        else:
            df[[f"{indicator}_relative_predicted", f"{indicator}_relative_deviation", f"{indicator}_relative_logprobs", f"{indicator}_relative_logprobs_deviation"]] = df.progress_apply(lambda row: average_relative_prediction(row, indicator, year), axis=1).apply(pd.Series)
            try :
                df.to_csv(f'./output/bash/{indicator}_{year}_nuts_llm_{model_short_name}_relative.csv')
            except:
                df.to_csv(f'./output/bash/{indicator}_{year}_nuts_llm_error_on_name.csv')