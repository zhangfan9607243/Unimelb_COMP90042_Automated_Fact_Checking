# Import packages and load data
import pandas as pd
import numpy as np

data_tran = pd.read_json('/root/COMP90042/data/data2/data_tran.json', orient='records', lines=True)
data_vald = pd.read_json('/root/COMP90042/data/data2/data_vald.json', orient='records', lines=True)
data_test = pd.read_json('/root/COMP90042/data/data2/data_test.json', orient='records', lines=True)
data_evdn = pd.read_json('/root/COMP90042/data/data2/data_evdn.json', orient='records', lines=True)

# Create temporary 'retain' label for evidences
data_evdn['retain'] = 0

# Load model & tokenizer (with suggested code for speeding up model & tokenizer loading from cloud computing service provider)
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import subprocess
import os

result = subprocess.run('bash -c "source /etc/network_turbo && env | grep proxy"', shell=True, capture_output=True, text=True)
output = result.stdout
for line in output.splitlines():
    if '=' in line:
        var, value = line.split('=', 1)
        os.environ[var] = value

from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4"

llama_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto").to(device)
llama_token = AutoTokenizer.from_pretrained(model_id)

# Ask question to llama about whether a evidence is related to climate change
def llama_ask_question(text, max_new_tokens=16):

    question = f"""
    You are an expert in climate change. 
    Please determine if the following text constitutes a statement related to climatology, meteorology, geology, or broadly within the fields of physics, chemistry, biology, or engineering. 
    Text: '{text}'
    Respond with only one word: Yes or No. Please do not respond with anything else.
    """

    input_ids = llama_token.encode(question, return_tensors="pt").to(device)

    with torch.no_grad():
        output = llama_model.generate(
            input_ids,
            do_sample=True,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            pad_token_id=llama_token.eos_token_id,
            early_stopping=True
        )

    answer = llama_token.decode(output[0], skip_special_tokens=True).strip().split("Please do not respond with anything else.")[-1].strip().split()[0]

    return answer

data_evdn['retain'] = data_evdn['evidence'].apply(lambda text: 1 if llama_ask_question(text).lower == 'yes' else 0)

# Reain all the evidence in taining and validation dataset
all_evidence_ids = set([eid for sublist in data_tran['evidences'] for eid in sublist]) | set([eid for sublist in data_vald['evidences'] for eid in sublist])
data_evdn['retain'] = data_evdn.apply(lambda row: 1 if row['evidence_id'] in all_evidence_ids else row['retain'], axis=1)

# Save filtered evidences
data_evdn_filtered = data_evdn[data_evdn['retain'] == 1]
data_evdn_filtered.to_json('/root/COMP90042/data/data2/data_evdn_filtered.json', orient='records', lines=True)