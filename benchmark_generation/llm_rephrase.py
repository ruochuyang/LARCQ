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
from tqdm import tqdm
from vllm import LLM
from vllm.sampling_params import SamplingParamsSamplingParams

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


################ specify the LLM prompts here ################
from prompt_rephrase_captions import examples, instruction
##############################################################


def load_data(info_path):

    info = pd.read_csv(info_path)
    
    captions = []
        
    for _, item in info.iterrows():

        audio_id = item['audio_id']

        caption = item['caption']
               
        captions.append({'audio_id': audio_id, 'caption': caption})
        
    return captions



def process_prompts(captions):
    
    batch_prompts = []
    
    for item in captions:
        
        input_caption = item['caption']
        
        prompt = instruction.substitute(examples = examples, input_caption = input_caption)

        # format prompts for mixtral instruct version
        prompt = f"<s> [INST] {prompt} [/INST]"
        
        batch_prompts.append(prompt)

    return batch_prompts



def batch_inference(llm, sampling_params, batch_size, batch_prompts):
    
    all_outputs = []
    
    length = len(batch_prompts)
    
    for i in range(0, length, batch_size):
        
        batch_inputs = batch_prompts[i : i + batch_size]
        
        outputs = llm.generate(batch_inputs, sampling_params)
        
        all_outputs.extend(outputs)
        
    return all_outputs





def process_outputs(inference_outputs, captions, output_path):
    
    res = []

    for i in tqdm(range(len(inference_outputs))):
        
        # original information
        audio_id = captions[i]['audio_id']
        orginial_caption = captions[i]['caption']
        
        # rephrased_caption: 'Rhesus monkey, Macaca mulatta, emits a loud cry'
        response = inference_outputs[i]
        
        rephrased_caption = response.outputs[0].text
   
        res.append({'audio_id': audio_id, 'original_caption': orginial_caption, 'rephrased_caption': rephrased_caption})
    
    
    # save to csv file
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
       
        fieldnames = ['audio_id', 'original_caption', 'rephrased_caption']
    
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writeheader()  

        writer.writerows(res)
    
    return





if __name__ == "__main__":    
    
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
    
    # load data
    info_path = 'datasets/Clotho_LARCQ/raw_LARCQ_captions.csv'
    captions = load_data(info_path)
   
    # convert captions into LLM prompts
    batch_prompts = process_prompts(captions)
    
   # run Mixtral inference
    inference_outputs = []
    
    for llm_input in batch_prompts:

        res = llm.chat(messages=llm_input, sampling_params=sampling_params)
        
        inference_outputs.append(res[0].outputs[0].text)
    
    # store LLM outputs
    output_path = 'datasets/Clotho_LARCQ/rephrased_caption.csv'
        
    process_outputs(inference_outputs, captions, output_path)
 
