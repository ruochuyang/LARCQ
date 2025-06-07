import os
import pickle
import csv
from tqdm import tqdm
import torch
import torchaudio
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer, LlamaConfig
from utils.prompter import Prompter
import datetime
import time, json
import pandas as pd



# select GPU: cuda:0, cuda:1, ..., cuda:7
device = torch.device('cuda:0')


# model.eval()
base_model = "./models/Llama-2-7b-chat-hf-qformer/"

prompter = Prompter('alpaca_short')
tokenizer = LlamaTokenizer.from_pretrained(base_model)
model = LlamaForCausalLM.from_pretrained(base_model, device_map="auto", torch_dtype=torch.float32) 

config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.0,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)
temp, top_p, top_k = 0.1, 0.95, 500

# GAMA model
eval_mdl_path = './models/stage5_epoch2/pytorch_model.bin'

state_dict = torch.load(eval_mdl_path, map_location='cpu')
msg = model.load_state_dict(state_dict, strict=False)

model.is_parallelizable = True
model.model_parallel = True

# unwind broken decapoda-research config
model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
model.config.bos_token_id = 1
model.config.eos_token_id = 2

model.eval()
eval_log = []
cur_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
log_save_path = './inference_log/'
if os.path.exists(log_save_path) == False:
    os.mkdir(log_save_path)
log_save_path = log_save_path + cur_time + '.json'

SAMPLE_RATE = 16000
AUDIO_LEN = 1.0

def load_audio(filename):
    waveform, sr = torchaudio.load(filename)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
        sr = 16000
    audio_info = 'Original input audio length {:.2f} seconds, number of channels: {:d}, sampling rate: {:d}.'.format(waveform.shape[1]/sr, waveform.shape[0], sr)
    waveform = waveform - waveform.mean()
    fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr,
                                              use_energy=False, window_type='hanning',
                                              num_mel_bins=128, dither=0.0, frame_shift=10)
    target_length = 1024
    n_frames = fbank.shape[0]
    p = target_length - n_frames
    if p > 0:
        m = torch.nn.ZeroPad2d((0, 0, 0, p))
        fbank = m(fbank)
    elif p < 0:
        fbank = fbank[0:target_length, :]
    # normalize the fbank
    fbank = (fbank + 5.081) / 4.4849
    return fbank, audio_info



def predict(audio_path, question):
    
    begin_time = time.time()

    instruction = question
    prompt = prompter.generate_prompt(instruction, None)
    #print('Input prompt: ', prompt)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)

    if audio_path != 'empty':
        cur_audio_input, audio_info = load_audio(audio_path)
        cur_audio_input = cur_audio_input.unsqueeze(0)
        if torch.cuda.is_available() == False:
            pass
        else:
            # cur_audio_input = cur_audio_input.half().to(device)
            cur_audio_input = cur_audio_input.to(device)
    else:
        cur_audio_input = None
        audio_info = 'Audio is not provided, answer pure language question.'

    generation_config = GenerationConfig(
        do_sample=True,
        temperature=0.1,
        top_p=0.95,
        max_new_tokens=400,
        bos_token_id=model.config.bos_token_id,
        eos_token_id=model.config.eos_token_id,
        pad_token_id=model.config.pad_token_id,
        num_return_sequences=1
    )

    # Without streaming
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids.to(device),
            audio_input=cur_audio_input,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=400,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)[len(prompt)+6:-4] # trim <s> and </s>
    end_time = time.time()
    
    cur_res = {'audio_id': audio_path, 'input': instruction, 'output': output}
    eval_log.append(cur_res)
    with open(log_save_path, 'w') as outfile:
        json.dump(eval_log, outfile, indent=1)
    print('elapsed time: ', end_time - begin_time, ' seconds')
    return audio_info, output


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
    
    # load data
    audios, queries = load_data(benchmark)
    
    responses = []
        
    for idx in tqdm(range(len(queries))):
    
        audio_path = queries[idx]['file_name']
        query = 'describe this audio'
        audio_info, response = predict(audio_path, query)
        responses.append(response)
    
    # save responses to csv file
    alm_response_path = f'results/alm_results/{benchmark}/gama_response.csv'

    with open(alm_response_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # write the header row
        writer.writerow(['audio_id', 'query', 'gama_response'])
        
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
        response = item['gama_response']
        audio = item['audio_id']
        alm_responses[audio] = response

    # get responses for retrieved audios
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

    existing_df.insert(2, 'gama_response_1', responses_1)
    existing_df.insert(4, 'gama_response_2', responses_2)
    existing_df.insert(6, 'gama_response_3', responses_3)
    existing_df.insert(8, 'gama_response_4', responses_4)
    existing_df.insert(10, 'gama_response_5', responses_5)

    new_csv_path = f'results/alm_results/{benchmark}/retrieved_audios_gama.csv'  

    existing_df.to_csv(new_csv_path, index=False)

    print('\nDone!')