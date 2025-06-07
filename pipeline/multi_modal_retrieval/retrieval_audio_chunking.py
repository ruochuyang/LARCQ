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


# set environment variable fo huggingface 
os.environ["TOKENIZERS_PARALLELISM"] = "false"



def load_data(benchmark):
# load in audios and captions

    audios_path = f'datasets/{benchmark}/audios/'
    info_path = f'datasets/{benchmark}/queries.csv'  

    info = pd.read_csv(info_path)

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
# rule out auido files whose caption tokens are longer than the maximum tokens 512 for LAION-CLAP model
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
    encoded_captions = []

    for idx in tqdm(range(0, len(captions), bs)):

        with torch.no_grad():
            input_ids = tokenizer(captions[idx:idx+bs], padding=True, return_tensors="pt").to(device)
            encoded_captions.append(model.get_text_features(**input_ids).cpu().numpy())

    encoded_captions = np.concatenate(encoded_captions)

    return encoded_captions



def encode_10_second_audio_chunks(audios, chunk_duration, model, feature_extractor, device):
# split audios into chunks of fixed duration and then encode them
# NOTE: The duration of each chunk is fixed as 10 seconds

    audio_paths = [i['file_name'] for i in audios]
    
    audio_chunks_embeddings = []

    for idx in tqdm(range(len(audio_paths))):
        
        audio_file = audio_paths[idx]
        
        # get audio duration (seconds)
        audio_duration = librosa.get_duration(path = audio_file)
    
        # chunk the audio into segments of fixed chunk_duration
        split = np.ceil(audio_duration / chunk_duration)
                     
        audio_read = librosa.load(audio_file, sr=16000)[0]
        audio_read = librosa.resample(audio_read, orig_sr=16000, target_sr=48000)
            
        audio_chunks = np.array_split(audio_read, split)
            
        chunk_embeddings = []
            
        for audio_chunk in audio_chunks:
            audio_input = feature_extractor(audio_chunk, sampling_rate=48000, return_tensors="pt").to(device)
                
            with torch.no_grad():
                chunk_embeddings.append(model.get_audio_features(**audio_input).detach().cpu().numpy())
        
        # chunk_embeddings (split, embedding_length), (?, 512)
        chunk_embeddings = np.concatenate(chunk_embeddings)

        audio_chunks_embeddings.append(chunk_embeddings)

    return audio_chunks_embeddings



def compute_metrics_audio_chunks(filtered_captions, audio_chunks_embeddings, text_embeddings):
# compute R@ metrics for the text-to-audio retrieval task, use the raw captions as the text queries
#
# compute R@ metrics from a similiarity matrix sims
# sims: matrix of similarities between embeddings, where x_{i,j} = <text_embedding[i], audio_embedding[j]>
# sims = torch.matmul(text_embedding, audio_embedding.t())
#
# audio_chunks_embeddings (audio_files, 4, 512)
# text embeddings: (audio_files, 512)


    ground_truth = np.arange(0, len(filtered_captions))
    ground_truth = torch.tensor(ground_truth).view(-1, 1)
    print('ground_truth:', ground_truth.shape)
    
    
    # audio chunks embeddings (1000, ?, 512)
    print('audio chunks embeddings:', len(audio_chunks_embeddings)) 
    print('audio chunks embeddings:', audio_chunks_embeddings[0].shape)  
    print('text embeddings:', text_embeddings.shape)

    queries_score = []

    # iterate through each text file
    for idx in tqdm(range(len(text_embeddings))):
        
        text_embedding = text_embeddings[idx]

        audios_score = []

        # iterate through each audio file
        for audio_embedding in audio_chunks_embeddings:

            chunks_score = []

            # iterate through chunks of the current audio file
            for chunk_embedding in audio_embedding:
                # audio_embedding (split ?, 512)
                # text_embedding: (512, )
                score = torch.tensor(text_embedding) @ torch.tensor(chunk_embedding).t()

                chunks_score.append(score)

            # max/sum vote for audio chunking
            # chunk_scores (split ?, )
            chunks_max_score = np.max(chunks_score)
            
            # audio_scores (audio files, )
            audios_score.append(chunks_max_score)
            
        # append the audios score of the current text query    
        queries_score.append(audios_score)
           
    # queries_score (text queries, audio files)
    queries_score = np.array(queries_score)
    queries_score = torch.tensor(queries_score)
    print('queries_score', queries_score.shape)

    # ranking: (text queries, audio index)
    ranking = torch.argsort(queries_score, descending=True)
    print('ranking:', ranking.shape)

    # preds
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




def evaluate(benchmark): 

    print('Loading audios and captions...')
    audios, captions = load_data(benchmark)
    
    print('Filtering captions...')    
    filtered_audio_ids, filtered_captions = filter_captions(captions)
    
    print('Encoding captions...')
    text_embeddings = encode_captions(filtered_captions, clap_model, tokenizer, device)

    # freshly encode audios OR load precomputed audio embeddings
    audio_embedding_path = f'./datasets/{benchmark}/features/audio-embeddings-audio-chunking.pkl'
    
    if os.path.isfile(audio_embedding_path):
        print('Found audio embeddings, directly load them...')
            
        # load the embeddings from the pickle file
        with open(audio_embedding_path, 'rb') as f:
            
            audio_chunks_embeddings = pickle.load(f)
            
    else:
        print('Not found audio embeddings, freshly encode audios...')
        
        audio_chunks_embeddings = encode_10_second_audio_chunks(audios, chunk_duration, clap_model, feature_extractor, device)
    
        # save audio embeddings to a pickle file
        with open(audio_embedding_path, 'wb') as f:
            pickle.dump(audio_chunks_embeddings, f)
    
        print('audio chunks embeddings saved!')
    
     
    # compute metrics
    print('Computing metrics...')
    ranking = compute_metrics_audio_chunks(filtered_captions, audio_chunks_embeddings, text_embeddings)
    

    # save rank list to a pickle file
    ranking_path = f'./datasets/{benchmark}/clap_ranking/audio-ranking-audio-chunking.pkl'

    with open(ranking_path, 'wb') as f:
        pickle.dump(ranking, f)
    
    print('\nRanking list saved...')
        
    # load the ranking from the pickle file
    with open(ranking_path, 'rb') as f:
        ranking = pickle.load(f)
    
    print('Ranking:', ranking)

    # extract top 5 retrieved audios from the ranking list
    extract_ranking(benchmark, ranking_path, result_path)

    return
        

    
   
  
if __name__ == '__main__':

    # select benchmark to evaluate
    benchmark = 'Clotho_LARCQ'  # Clotho_LARCQ, SoundDescs_LARCQ, <Your Own Dataset>
    
    # select GPU
    device = torch.device('cuda:5')
    print('Using device:', device)
    
    # enable fusion to deal with long audios
    clap_model_name = 'clap-htsat-fused'
    clap_model = ClapModel.from_pretrained("./models/" + clap_model_name , local_files_only=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained("./models/" + clap_model_name, local_files_only=True)
    feature_extractor = AutoFeatureExtractor.from_pretrained("./models/" + clap_model_name, local_files_only=True)   
    
    # chunk duration in seconds
    chunk_duration = 10
    
    # execute the evaluation process on the benchmark
    result_path = f'results/retrieved_results/{benchmark}/retrieved_audios_audio_chunking.csv'
    
    evaluate(benchmark, result_path)

    print('Done!')