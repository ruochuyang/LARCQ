import argparse
import json
import logging
import math
import os
import subprocess
from enum import Enum
from typing import Any, List
import pickle
import h5py
import pandas as pd
import csv
import numpy as np
from tqdm import tqdm
from vllm import LLM
from vllm.sampling_params import SamplingParamsSamplingParams

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

################ specify the LLM prompts here ################
from prompt_ranking import examples, instruction
##############################################################


def load_data(benchmark, ALM):
# load retrieved results

    info_path = f'results/alm_results/{benchmark}/retrieved_audios_{ALM}.csv'
     
    info = pd.read_csv(info_path)
    
    data = []
        
    for _, item in info.iterrows():  
            
        data.append({'text_query': item['query'], 
                     'text_description_1': item[f'{ALM}_response_1'],
                     'text_description_2': item[f'{ALM}_response_2'],
                     'text_description_3': item[f'{ALM}_response_3'],
                     'text_description_4': item[f'{ALM}_response_4'],
                     'text_description_5': item[f'{ALM}_response_5'],
                     'groundtruth_audio': item['groundtruth_audio']
                    })
                
    return data


def process_prompts(data):
    
    batch_prompts = []
    
    for item in data:
        
        text_query = item['text_query']
        text_description_1 = item['text_description_1']
        text_description_2 = item['text_description_2']
        text_description_3 = item['text_description_3']
        text_description_4 = item['text_description_4']
        text_description_5 = item['text_description_5']

        prompt = instruction.substitute(examples=examples, text_query=text_query, text_description_1=text_description_1, text_description_2=text_description_2, text_description_3=text_description_3, text_description_4=text_description_4, text_description_5=text_description_5)
        
        # format prompts for mixtral instruct version
        prompt = f"<s> [INST] {prompt} [/INST]"
        
        batch_prompts.append(prompt)

    return batch_prompts


def process_outputs(inference_outputs, data, output_path):
    
    res = []

    for ii in tqdm(range(len(inference_outputs))):
        
        # original information 
        text_query = data[ii]['text_query']
        text_description_1 = data[ii]['text_description_1']
        text_description_2 = data[ii]['text_description_2']
        text_description_3 = data[ii]['text_description_3']
        text_description_4 = data[ii]['text_description_4']
        text_description_5 = data[ii]['text_description_5']
        groundtruth_audio = data[ii]['groundtruth_audio']
        
        # ranking result
        response = inference_outputs[ii]
        
        llm_rank = response.outputs[0].text
        
        llm_confidence = 100 * np.exp(response.outputs[0].cumulative_logprob)
                
        res.append({'text_query': text_query, 
                    'text_description_1': text_description_1,
                    'text_description_2': text_description_2,
                    'text_description_3': text_description_3,
                    'text_description_4': text_description_4,
                    'text_description_5': text_description_5,
                    'llm_rank': llm_rank,
                    'llm_confidence (%)': int(llm_confidence),
                    'groundtruth_audio': groundtruth_audio
                   })
        
    # save to csv file
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
       
        fieldnames = ['text_query', 'text_description_1', 'text_description_2', 
                 'text_description_3', 'text_description_4', 'text_description_5',
                 'llm_rank', 'llm_confidence (%)', 'groundtruth_audio']
    
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writeheader()  

        writer.writerows(res)

    return


if __name__ == "__main__": 

    # select benchmark to evaluate
    benchmark = 'Clotho_LARCQ'  # Clotho_LARCQ, SoundDescs_LARCQ, <Your Own Dataset>

     # select ALM
    ALM = 'gama' # gama, flamingo
    
    # specify Mixtral configurations
    model_name = "mistralai/Mistral-NeMo-Instruct-2407"
    num_of_GPUs = 2 # The number of GPUs to be used for inference
    temperature = 0.7 # Temperature for generating dialog
    top_p = 0.9 # Threshold probability and selects the top tokens whose cumulative probability exceeds the threshold
    max_tokens = 2048 # Maximum number of tokens in generated dialog

    sampling_params = SamplingParams(max_tokens=max_tokens)

    llm = LLM(
        model=model_name,
        tokenizer_mode="mistral",
        load_format="mistral",
        config_format="mistral",
    )

    # load retrieved results
    data = load_data(benchmark, ALM)
    
    # convert captions into LLM prompts
    batch_prompts = process_prompts(data)
    
    # run Mixtral inference
    inference_outputs = []
    
    for llm_input in batch_prompts:

        res = llm.chat(messages=llm_input, sampling_params=sampling_params)
        
        inference_outputs.append(res[0].outputs[0].text)

    # store Mixtral outputs
    output_path = f'results/llm_results/{benchmark}/{ALM}_llm_ranking.csv'
        
    process_outputs(inference_outputs, data, output_path)
    