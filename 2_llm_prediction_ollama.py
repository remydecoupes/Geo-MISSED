import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="solidrust/Codestral-22B-v0.1-hf-AWQ")
args = parser.parse_args()

import configparser

credential_file = "credentials.ini"
credential_config = configparser.ConfigParser()
credential_config.read(credential_file)

ISDM_API_KEY = credential_config['ISDM']["ISDM_API_KEY"]

import re
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
import pandas as pd
from langchain_core.output_parsers import StrOutputParser
import statistics
import timeout_decorator
import time
import numpy as np
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import sys, os
from langchain_core.output_parsers import StrOutputParser
import pycountry
from tqdm import tqdm
tqdm.pandas()

# Set the environment variables from shell environment
if args.model == "codestral": 
    OPENAI_API_KEY = ISDM_API_KEY
    OPENAI_CHAT_MODEL = "solidrust/Codestral-22B-v0.1-hf-AWQ"
    OPENAI_CHAT_API_URL = "https://isdm-chat.crocc.meso.umontpellier.fr/openai"
    model = OPENAI_CHAT_MODEL
    model_short_name = "codestral"

    llm = ChatOpenAI(
        model=OPENAI_CHAT_MODEL,
        openai_api_key=OPENAI_API_KEY,
        openai_api_base=OPENAI_CHAT_API_URL,
    )
elif args.model == "llama3:8b":
    from langchain_community.llms import Ollama
    MODEL = "llama3:8b"
    llm = Ollama(model=MODEL)
    model_short_name = "llama3-8b"
elif args.model == "qwen2.5:7b":
    from langchain_community.llms import Ollama
    MODEL = "qwen2.5:7b"
    llm = Ollama(model=MODEL)
    model_short_name = "qwen2.5-7b"

year = 2017
NUTS_Level = 2

parser = StrOutputParser()

template = """
You are a statician working for the European commission at EUROSTAT. 
You have to give the average income per inhabitant by NUTS_2 levels for year {year}
Don't compute it, just guess the income.
Only answer with the average income (not a range, just the income) without any other words.
Example of answer: "30000", with no thousands separator.

Question: What is the average income per inhabitant for {NUTS_ID} in {country} in {year}
"""

prompt = PromptTemplate.from_template(template)
# result = prompt.format(NUTS_ID = "Auvergne")
# result

chain = prompt | llm | parser

@timeout_decorator.timeout(10, use_signals=False)  # Set the timeout to 10 seconds
def invoke_with_timeout(NUTS_ID, country):
    return chain.invoke({"NUTS_ID": NUTS_ID, "year": year, "country": country})

def prediction(NUTS_ID, country):
    try:
        result = invoke_with_timeout(NUTS_ID, country)  # Try to invoke within the timeout
    except timeout_decorator.timeout_decorator.TimeoutError:
        print(f"Timeout occurred for NUTS_ID: {NUTS_ID}. Retrying...")
        time.sleep(2)  # Optional sleep time before retrying
        try:
            result = invoke_with_timeout(NUTS_ID)  # Retry the operation
        except:
            return np.nan
    number = re.search(r'\d+', result).group()
    number = int(number)
    return number

def average_prediction(row):
    """
    Compute the average and deviation of predictions for a given NUTS_ID and country.
    Expects the row to contain both 'NUTS_ID' and 'country'.
    """
    number_prediction = 3
    predictions = []

    for _ in range(number_prediction):
        prediction_value = prediction(row["NUTS_NAME"], row["country"])
        predictions.append(prediction_value)

    # Handle potential NaN values in predictions
    predictions = [p for p in predictions if not pd.isna(p)]

    if predictions:
        average = sum(predictions) / len(predictions)
        deviation = statistics.stdev(predictions) if len(predictions) > 1 else 0
    else:
        average = np.nan
        deviation = np.nan

    return average, deviation

df = pd.read_csv(f"./output/gdp_{year}_nuts_{NUTS_Level}.csv")
df["country"] = df["CNTR_CODE"].apply(lambda code: pycountry.countries.get(alpha_2=code).name if pycountry.countries.get(alpha_2=code) else "Unknown")

df[[f"{year}_predicted", f"{year}_deviation"]] = df.progress_apply(average_prediction, axis=1).apply(pd.Series)
try :
    df.to_csv(f'./output/gdp_{year}_nuts_{NUTS_Level}_llm_{model_short_name}.csv')
except:
    df.to_csv(f'./output/gdp_{year}_nuts_{NUTS_Level}_llm_error_on_name.csv')