import json
import torch
import os
import h5py
import faiss
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, ClapModel, AutoFeatureExtractor
import pickle
import random
import csv
import math


# set environment variable fo huggingface 
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_data(benchmark):
# load in audios and captions

    audios_path = f'./datasets/{benchmark}/audios'
    queries_path = f'./datasets/{benchmark}/queries.csv'  
    info = pd.read_csv(queries_path)

    audios = []
    captions = []
        
    for _, item in info.iterrows():

        audio_id = item['audio_id']
       
        audios.append({'audio_id': audio_id, 'file_name': audios_path + audio_id})

        caption = item['query']
            
        captions.append({'audio_id': audio_id, 'caption': caption, 'file_name': audios_path + audio_id})

    assert len(audios) == len(captions), "audio size should match caption size!"

    return audios, captions





def filter_captions(data):
# ruled out auido files whose caption tokens are longer than the maximum tokens for LAION-CLAP model (512)
# data: {audio id, caption}
# separate data to filtered_audio_ids, filtered_captions


    decoder_name = './models/gpt2'
    tokenizer = AutoTokenizer.from_pretrained(decoder_name, local_files_only=True)
    bs = 512

    audio_ids = [d['audio_id'] for d in data]
    captions = [d['caption'] for d in data]
    encodings = []
    
    for idx in range(0, len(data), bs):
        encodings += tokenizer.batch_encode_plus(captions[idx:idx+bs], return_tensors='np')['input_ids'].tolist()

    filtered_audio_ids, filtered_captions = [], []

    assert len(audio_ids) == len(captions) and len(captions) == len(encodings)

    for audio_id, caption, encoding in zip(audio_ids, captions, encodings):
        
        if len(encoding) <= 512:
            filtered_audio_ids.append(audio_id)
            filtered_captions.append(caption)
        else:
            print('Ruled out auido files whose caption embeddings are longer than 512:', audio_id)
            

    return filtered_audio_ids, filtered_captions




def encode_captions(captions, model, tokenizer, device):

    bs = 256
    text_embeddings = []

    for idx in tqdm(range(0, len(captions), bs)):

        with torch.no_grad():

            input_ids = tokenizer(captions[idx:idx+bs], padding=True, return_tensors="pt").to(device)        
            text_embeddings.append(model.get_text_features(**input_ids).cpu().numpy())

    text_embeddings = np.concatenate(text_embeddings)

    return text_embeddings





def encode_audios(audios, model, feature_extractor, device):

    audio_paths = [i['file_name'] for i in audios]

    batch_size = 64

    audio_embeddings = []

    for idx in tqdm(range(0, len(audio_paths), batch_size)):

        audio_read = [librosa.resample(librosa.load(i, sr=16000)[0],orig_sr=16000,target_sr=48000) for i in audio_paths[idx : idx + batch_size]]

        audio_input = feature_extractor(audio_read, sampling_rate=48000, return_tensors="pt").to(device)

        with torch.no_grad():
            audio_embeddings.append(model.get_audio_features(**audio_input).detach().cpu().numpy())

    audio_embeddings = np.concatenate(audio_embeddings)

    return audio_embeddings





def extract_ranking(benchmark, ranking_path, result_path):
    
    # load data
    data_path = f'datasets/{benchmark}/queries.csv'  
    data = pd.read_csv(data_path)
    
    audio_ids = []
    queries = []
    
    for _, item in data.iterrows():
        
        audio_ids.append(item['audio_id']) 
        queries.append(item['query'])

    audio_ids = np.array(audio_ids)

    # load the ranking from the pickle file
    # ranking: (text queries, audio rank index)
    with open(ranking_path, 'rb') as f:
        ranking = pickle.load(f)
        ranking = ranking.numpy()
        
    retrieved_audios = []

    # select top 5 audios
    for query in ranking:  
        top_5_index = query[0:5]  
        top_5_audios = audio_ids[top_5_index]  
        retrieved_audios.append(top_5_audios)
    
    # Open a CSV file for writing
    with open(result_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['query', 'retrieved_audio_1', 'retrieved_audio_2', 'retrieved_audio_3', 'retrieved_audio_4', 'retrieved_audio_5', 'groundtruth_audio'])

        for query, retrieved_audio, audio in zip(queries, retrieved_audios, audio_ids):
            csvwriter.writerow([query, retrieved_audio[0], retrieved_audio[1], retrieved_audio[2], retrieved_audio[3], retrieved_audio[4], audio])
    
    return 




def compute_metrics(filtered_captions, audio_embeddings, text_embeddings):
# compute R@ metrics for the whole text queries, use the raw captions as the text queries

    ground_truth = np.arange(0, len(filtered_captions))
    ground_truth = torch.tensor(ground_truth).view(-1, 1)
    print('ground_truth:', ground_truth.shape)

    # compute text-to-audio retrieval metrics from a similiarity matrix sims
    # sims: matrix of similarities between embeddings, where x_{i,j} = <text_embedding[i], audio_embedding[j]>
    # sims = torch.matmul(text_embedding, audio_embedding.t())

    print('audio embeddings:', audio_embeddings.shape)
    print('text embeddings:', text_embeddings.shape)
    ranking = torch.argsort(torch.tensor(text_embeddings) @ torch.tensor(audio_embeddings).t(), descending=True)

    preds = torch.where(ranking == ground_truth)[1]
    preds = preds.cpu().numpy()
    print('preds:', preds.shape, '\n')

    # compute metrics
    metrics = {}
    metrics[f"mean_rank"] = preds.mean() + 1
    metrics[f"median_rank"] = np.floor(np.median(preds)) + 1

    for k in [1, 5, 10]:
        metrics[f"R@{k}"] = np.mean(preds < k) * 100

    # map@10
    metrics[f"mAP@10"] = np.mean(np.where(preds < 10, 1 / (preds + 1), 0.0)) * 100

    print(f"Text-to-audio Retrieval Results: " + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()]))
    
    return ranking





def evaluate(benchmark, result_path): 

    print('Loading audios and queries...')
    audios, captions = load_data(benchmark)
    
    print('Filtering captions...')    
    filtered_audio_ids, filtered_captions = filter_captions(captions)
   
    print('Encoding captions...')
    text_embeddings = encode_captions(filtered_captions, clap_model, tokenizer, device)
    
    # freshly encode audios OR load precomputed audio embeddings
    audio_embedding_path = f'./datasets/{benchmark}/features/audio-embeddings-no-chunking.hdf5'
    
    if os.path.isfile(audio_embedding_path):
        
        print('Found audio embeddings, directly load them...')

        with h5py.File(audio_embedding_path, 'r') as hf:
            audio_embeddings = hf['audio_embeddings'][()]
    else:
        print('Not found audio embeddings, freshly encode audios...')
        
        audio_embeddings = encode_audios(audios, clap_model, feature_extractor, device)
        
        # save audio embeddings to hdf5 file
        with h5py.File(audio_embedding_path, 'w') as hf:
            hf.create_dataset('audio_embeddings', data = audio_embeddings)
            
        print('audio embeddings saved!')


    # compute metrics
    print('Computing metrics...') 
    ranking = compute_metrics(filtered_captions, audio_embeddings, text_embeddings)

    
    # save rank list to a pickle file
    ranking_path = f'./datasets/{benchmark}/clap_ranking/audio-ranking-no-chunking.pkl'
    
    with open(ranking_path, 'wb') as f:
        pickle.dump(ranking, f)
    
    print('\nRanking list saved to...')
    
    # load the ranking from the pickle file
    with open(ranking_path, 'rb') as f:
        ranking = pickle.load(f)
    
    print('Ranking:', ranking)

    # extract top 5 retrieved audios
    extract_ranking(benchmark, ranking_path, result_path)

    return

        


if __name__ == '__main__':

    # select benchmark to evaluate
    benchmark = 'Clotho_LARCQ'  # Clotho_LARCQ, SoundDescs_LARCQ, <Your Own Dataset>

    # select GPU
    device = torch.device('cuda:3')
    print('Using device:', device)

    # enable_fusion to deal with long auidos
    clap_model_name = 'clap-htsat-fused'
    clap_model = ClapModel.from_pretrained("./models/" + clap_model_name , local_files_only=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained("./models/" + clap_model_name, local_files_only=True)
    feature_extractor = AutoFeatureExtractor.from_pretrained("./models/" + clap_model_name, local_files_only=True)   
    
    # execute the evaluation process on the benchmark
    result_path = f'results/retrieved_results/{benchmark}/retrieved_audios_no_chunking.csv'
    
    evaluate(benchmark, result_path)

    print('Done!')