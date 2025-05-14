#!/bin/bash

# Script: run_all_models.sh
# Description: Runs 2_llm_prediction_transformers.py for a list of predefined models


# Define model lists
mistral_models=(
  "Mistral-7B-Instruct-v0.3" 
  "Mistral-Small-24B-Instruct-2501"
)
llama_models=(
  "Llama-3.1-8B-Instruct" 
  "Llama-3.1-70B-Instruct"
)
qwen_models=(
  "Qwen2.5-7B-Instruct"
  "Qwen2.5-14B-Instruct"
  "Qwen2.5-32B-Instruct"
  "Qwen2.5-72B-Instruct"
)

# Add prefixes
for model in "${mistral_models[@]}"; do
  full_model="mistralai/$model"
  echo "Running for model: $full_model"
  python 2_llm_prediction_transformers.py --model "$full_model"
done

for model in "${llama_models[@]}"; do
  full_model="meta-llama/$model"
  echo "Running for model: $full_model"
  python 2_llm_prediction_transformers.py --model "$full_model"
done

for model in "${qwen_models[@]}"; do
  full_model="Qwen/$model"
  echo "Running for model: $full_model"
  python 2_llm_prediction_transformers.py --model "$full_model"
done

