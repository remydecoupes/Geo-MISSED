
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import pycountry
import statistics
from tqdm import tqdm
tqdm.pandas()

# Model and tokenizer setup
MODEL = "Qwen/Qwen2.5-7B-Instruct"  # Example model ID
model_short_name = "Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL)

# Load model with INT8 quantization
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    device_map="auto",
    torch_dtype=torch.float16,
    load_in_8bit=True,  # Enable 8-bit quantization
)

year = 2017

def prediction(NUTS_ID, country):
    NUTS_ID = NUTS_ID
    country = country
    prompt = f"What is the average income per inhabitant for {NUTS_ID} in {country} in {year}?"
    messages = [
        {"role": "system", "content": f'You are a statistician working for the European commission at EUROSTAT. You have to give the average income per inhabitant by NUTS 1 or 2 or 3 level for year {year}. Don\'t compute it, just guess the income.Answer only with the average income (a single number) without any other words or repetition of the question. Don\'t repeat the prompt neither. Example of answer: "30000".'},
        {"role": "user", "content": prompt}
    ]
    # Tokenize input prompt
    inputs = tokenizer.apply_chat_template(
        messages, 
        tokenize=False,
        add_generation_prompt=True
        )
    inputs = tokenizer([inputs], return_tensors="pt").to("cuda")
    
    # Generate output
    outputs = model.generate(
        **inputs,
        max_length=256,
        pad_token_id=tokenizer.pad_token_id,
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

def relative_prediction(NUTS_ID, country, country_income):
    prompt = f"What is the difference of income per inhabitant between {NUTS_ID} ({country}) and {country} (country income: {country_income})  in {year}?"
    messages = [
        {"role": "system", "content": f'You are a statistician working for the European commission at EUROSTAT. You have to give the difference of average income per inhabitant between a sub regions at NUTS 1 or 2 or 3 level and its country for {year}. Don\'t compute it, just guess the diffrence of income. Answer only with the difference income (a single number) without any other words or repetition of the question. Don\'t repeat the prompt neither. Example of answer: 3000.'},
        {"role": "user", "content": prompt}
    ]
    # Tokenize input prompt
    inputs = tokenizer.apply_chat_template(
        messages, 
        tokenize=False,
        add_generation_prompt=True
        )
    inputs = tokenizer([inputs], return_tensors="pt").to("cuda")
    
    # Generate output
    outputs = model.generate(
        **inputs,
        max_length=256,
        pad_token_id=tokenizer.pad_token_id,
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


def average_prediction(row):
    """
    Compute the average and deviation of predictions for a given NUTS_ID and country.
    Expects the row to contain both 'NUTS_ID' and 'country'.
    """
    number_prediction = 3
    predictions = []
    logprobs = []

    for _ in range(number_prediction):
        prediction_value, logprob = prediction(row["NUTS_NAME"], row["country"])
        predictions.append(prediction_value)
        logprobs.append(logprob)

    # Handle potential NaN values in predictions
    predictions = [int(p) for p in predictions if not pd.isna(p)]
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

def average_relative_prediction(row):
    """
.
    """
    number_prediction = 3
    predictions = []
    logprobs = []

    for _ in range(number_prediction):
        #relative_prediction(NUTS_ID, country, capital, capital_income):
        prediction_value, logprob = relative_prediction(row["NUTS_NAME"], row["country"], row["country_income"],)
        predictions.append(prediction_value)
        logprobs.append(logprob)

    # Handle potential NaN values in predictions
    predictions = [float(p) for p in predictions if not pd.isna(p)]
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
        average = np.nan
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
    # Example usage
    # NUTS_ID = "Auvergne"
    # country = "France"
    # result = prediction(NUTS_ID, country)
    # print("Final Prediction Result:", result)

    df = pd.read_csv(f"./output/gdp_{year}.csv")
    df["country"] = df["CNTR_CODE"].apply(lambda code: pycountry.countries.get(alpha_2=code).name if pycountry.countries.get(alpha_2=code) else "Unknown")

    # df = df.iloc[0:100]

    df[[f"{year}_predicted", f"{year}_deviation", f"{year}_logprobs", f"{year}_logprobs_deviation"]] = df.progress_apply(average_prediction, axis=1).apply(pd.Series)
    df[[f"{year}_relative_predicted", f"{year}_relative_deviation", f"{year}_relative_logprobs", f"{year}_relative_logprobs_deviation"]] = df.progress_apply(average_relative_prediction, axis=1).apply(pd.Series)
    
    try :
        df.to_csv(f'./output/gdp_{year}_nuts_llm_{model_short_name}.csv')
    except:
        df.to_csv(f'./output/gdp_{year}_nuts_llm_error_on_name.csv')