import os
import string
import yaml
import pandas as pd
from tqdm import tqdm
import csv
import pickle
from copy import deepcopy
import torch
from transformers import AutoTokenizer, set_seed 
set_seed(0)

from data import AudioTextDataProcessor
from src.factory import create_model_and_transforms


def prepare_tokenizer(model_config):
    tokenizer_path = model_config['tokenizer_path']
    cache_dir = model_config['cache_dir']
    
    text_tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=True,
        trust_remote_code=True,
        cache_dir=cache_dir,
    )
    
    text_tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<audio>", "<|endofchunk|>"]}
    )
    if text_tokenizer.pad_token is None:
        text_tokenizer.add_special_tokens({"pad_token": "<PAD>"})
    if text_tokenizer.sep_token is None:
        text_tokenizer.add_special_tokens({"sep_token": "<SEP>"})
    return text_tokenizer


def prepare_model(model_config, clap_config, checkpoint_path, device_id=0):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # disable the tokenizer parallelism warning
    model, tokenizer = create_model_and_transforms(
        **model_config,
        clap_config=clap_config,
        use_local_files=False,
        gradient_checkpointing=False,
        freeze_lm_embeddings=False,
    )
    model.eval()
    model = model.to(device_id)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_state_dict = checkpoint["model_state_dict"]
    model_state_dict = {k.replace("module.", ""): v for k, v in model_state_dict.items()}
    model.load_state_dict(model_state_dict, False)

    return model


def inference(model, tokenizer, item, processed_item, inference_kwargs, device_id=0):
    filename, audio_clips, audio_embed_mask, input_ids, attention_mask = processed_item
    audio_clips = audio_clips.to(device_id, dtype=None, non_blocking=True)
    audio_embed_mask = audio_embed_mask.to(device_id, dtype=None, non_blocking=True)
    input_ids = input_ids.to(device_id, dtype=None, non_blocking=True).squeeze()

    media_token_id = tokenizer.encode("<audio>")[-1]
    eoc_token_id = tokenizer.encode("<|endofchunk|>")[-1]
    sep_token_id = tokenizer.sep_token_id
    eos_token_id = tokenizer.eos_token_id
    
    outputs = model.generate(
        audio_x=audio_clips.unsqueeze(0),
        audio_x_mask=audio_embed_mask.unsqueeze(0),
        lang_x=input_ids.unsqueeze(0),
        eos_token_id=eos_token_id,
        max_new_tokens=128,
        **inference_kwargs,
    )

    outputs_decoded = [
        tokenizer.decode(output).split(tokenizer.sep_token)[-1].replace(tokenizer.eos_token, '').replace(tokenizer.pad_token, '').replace('<|endofchunk|>', '') for output in outputs
    ]

    return outputs_decoded




def main(config_file, data_root, checkpoint_path, items, inference_kwargs, is_dialogue=False, do_dialogue_last=False):
    config = yaml.load(open(config_file), Loader=yaml.FullLoader)
    clap_config = config['clap_config']
    model_config = config['model_config']

    text_tokenizer = prepare_tokenizer(model_config)
    DataProcessor = AudioTextDataProcessor(
        data_root=data_root,
        clap_config=clap_config,
        tokenizer=text_tokenizer,
        max_tokens=512,
    )

    print("===== checkpoint_path: {} =====".format(checkpoint_path))
    model = prepare_model(model_config=model_config, 
        clap_config=clap_config, 
        checkpoint_path=checkpoint_path
    )
    
    
    responses = []
        
    for ii in tqdm(range(len(items))):
        
        item = items[ii]
         
        print('----- File: {} -----'.format(item['name']))

        if is_dialogue:
            staged_item = deepcopy(item)
            if do_dialogue_last:
                if "assistant" in staged_item['dialogue'][-1]:
                    del staged_item['dialogue'][-1]["assistant"]

                processed_item = DataProcessor.process(staged_item)
                outputs = inference(
                    model, text_tokenizer, staged_item, processed_item,
                    inference_kwargs
                )[0]

                print('Prompt:', item['dialogue'][-1]['user'])
                print('Audio Flamingo:', outputs)

            else:
                
                staged_item['dialogue'] = []
                for each_round in item['dialogue'] :
                    staged_item['dialogue'].append({'user': each_round['user']})

                    processed_item = DataProcessor.process(staged_item)
                    outputs = inference(
                        model, text_tokenizer, staged_item, processed_item,
                        inference_kwargs
                    )[0]

                    staged_item['dialogue'][-1]['assistant'] = outputs

                    print('Prompt:', each_round['user'])
                    print('Audio Flamingo:', outputs)

        else:
            processed_item = DataProcessor.process(item)
            outputs = inference(
                model, text_tokenizer, item, processed_item,
                inference_kwargs
            )

            print('Prompt:', item['prompt'])
            print('Audio Flamingo:', outputs)
            
            responses.append(outputs[0])
    
    return responses


        
def load_data(benchmark):
# load in audios and queries

    audios_path = f'datasets/{benchmark}/audios/'

    info_path = f'datasets/{benchmark}/queries.csv'  

    info = pd.read_csv(info_path)

    audios = []
    queries = []
    
    for _, item in info.iterrows():

        audio_id = item['audio_id']
        audios.append({'audio_id': audio_id, 'file_name': audios_path + audio_id})

        query = item['query']     
        queries.append({'audio_id': audio_id, 'query': query, 'file_name': audios_path + audio_id})
            
    assert len(audios) == len(queries), "audio size should match query size!"

    return audios, queries



if __name__ == "__main__":

    # select benchmark to evaluate
    benchmark = 'Clotho_LARCQ'  # Clotho_LARCQ, SoundDescs_LARCQ, <Your Own Dataset>
    
    # select retrieved audios to run ALM reasoning
    retrieval_type = 'best' # best (query audio chunking), query_chunking, audio_chunking, no_chunking
    
    # data path
    data_root = f'datasets/{benchmark}/audios/'

    # ---------- Audio-Flamingo Model ---------- #
    config_file = 'models/flamingo_config.yaml'
    
    checkpoint_path = 'models/foundation.pt'
    
    # num_return_sequences: the number of responses returned by Flamingo
    inference_kwargs = {
        "do_sample": True,
        "top_k": 30,
        "top_p": 0.95,
        "num_return_sequences": 1
    }

   
    # generate responses for benchmarks
    audios, queries = load_data(benchmark)
    
    items = []
        
    for idx in range(len(queries)):
        
        audio_path =  queries[idx]['audio_id']
        item = {'name': audio_path, 'prefix': 'The task is audio reasoning.', 'prompt': 'Describe this audio in detail.'}
        items.append(item)
    

    # run inference
    responses = main(config_file, data_root, checkpoint_path, items, inference_kwargs, is_dialogue=False)
      
    # save responses to csv file
    alm_response_path = f'results/alm_results/{benchmark}/flamingo_response.csv'
        
    with open(alm_response_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # write the header row
        writer.writerow(['audio_id', 'query', 'flamingo_response'])
        
        # write the data rows
        for idx in range(len(responses)):
            audio_id = queries[idx]['audio_id']
            query = queries[idx]['query']
            response = responses[idx]

            writer.writerow([audio_id, query, response])

    # read ALM responses    
    info = pd.read_csv(alm_response_path)

    alm_responses = {}

    for _, item in info.iterrows():
        response = item['flamingo_response']
        audio = item['audio_id']
        alm_responses[audio] = response

    # get ALM responses for retrieved audios
    csv_path = f'results/retrieved_results/{benchmark}/retrieved_audios_{retrieval_type}.csv'

    existing_df = pd.read_csv(csv_path)

    responses_1 = []
    responses_2 = []
    responses_3 = []
    responses_4 = []
    responses_5 = []

    for _, item in existing_df.iterrows():

        audio_1 = item['retrieved_audio_1']
        r_1 = alm_responses[audio_1]
        responses_1.append(r_1)

        audio_2 = item['retrieved_audio_2']
        r_2 = alm_responses[audio_2]
        responses_2.append(r_2)

        audio_3 = item['retrieved_audio_3']
        r_3 = alm_responses[audio_3]
        responses_3.append(r_3)

        audio_4 = item['retrieved_audio_4']
        r_4 = alm_responses[audio_4]
        responses_4.append(r_4)

        audio_5 = item['retrieved_audio_5']
        r_5 = alm_responses[audio_5]
        responses_5.append(r_5)


    # store retrieved responses to csv files  
    if len(existing_df) != len(responses_1):
        input('Error!')

    existing_df.insert(2, 'flamingo_response_1', responses_1)
    existing_df.insert(4, 'flamingo_response_2', responses_2)
    existing_df.insert(6, 'flamingo_response_3', responses_3)
    existing_df.insert(8, 'flamingo_response_4', responses_4)
    existing_df.insert(10, 'flamingo_response_5', responses_5)

    new_csv_path = f'results/alm_results/{benchmark}/retrieved_audios_flamingo.csv'  

    existing_df.to_csv(new_csv_path, index=False)
            
    print('\nDone!')