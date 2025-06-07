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
import csv


# set environment variable fo huggingface 
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_data(benchmark):
# load in audios and captions

    audios_path = f'./datasets/{benchmark}/audios'
    queries_path = f'./datasets/{benchmark}/queries.csv'
    info = pd.read_csv(queries_path)
    audios = []
  
    for _, item in info.iterrows():
        audio_id = item['audio_id']
        audios.append({'audio_id': audio_id, 'file_name': audios_path + audio_id})

    split_captions = pd.read_csv(queries_path).to_dict('records')
    
    return audios, split_captions




def encode_captions(split_captions, model, tokenizer, device):
    
    split_text_embebddings = []

    for idx in tqdm(range(len(split_captions))):
    
        text_embeddings = []

        num_of_splits = len(split_captions[idx])

        for ii in range(1, num_of_splits-1):

            caption = split_captions[idx]['split_' + str(ii)]
   
            with torch.no_grad():

                input_ids = tokenizer(caption, padding=True, return_tensors="pt").to(device)

                text_embeddings.append(model.get_text_features(**input_ids).cpu().numpy())

        split_text_embebddings.append(text_embeddings)


    return split_text_embebddings





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



def compute_metrics(split_captions, audio_embeddings, split_text_embeddings):
# compute R@ metrics for the split captions

    ground_truth = np.arange(0, len(split_captions))
    ground_truth = torch.tensor(ground_truth).view(-1, 1)
    print('ground_truth:', ground_truth.shape)
    

    # compute text-to-audio retrieval metrics from a similiarity matrix sims
    # sims: matrix of similarities between embeddings, where x_{i,j} = <text_embedding[i], audio_embedding[j]>
    # sims = torch.matmul(text_embedding, audio_embedding.t())
    ranking = []
    
    for query_embeddings in split_text_embeddings:
        
        scores = []
        
        for split_embedding in query_embeddings:
            
            score = torch.tensor(split_embedding) @ torch.tensor(audio_embeddings).t()
            
            scores.append(score)
        
        # scores (?, )
        scores = np.concatenate(scores)
        
        # max/sum vote for query chunking
        final_score = np.max(scores, axis=0)
      
        # query_ranking
        query_ranking = torch.argsort(torch.from_numpy(final_score), descending=True)
        
        query_ranking = np.array(query_ranking)

        ranking.append(query_ranking)
            
    
    # ranking: (text queries, audio rank)
    ranking = np.array(ranking)
    ranking = torch.from_numpy(ranking)
    print('ranking:', ranking.shape)
    
     
    # preds
    preds = torch.where(ranking == ground_truth)[1]
    preds = preds.cpu().numpy()
    print('preds:', preds, '\n')

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
    
    print('Loading audios and captions...')
    audios, split_captions = load_data(benchmark)
    
    print('Encoding captions...')
    split_text_embeddings = encode_captions(split_captions, clap_model, tokenizer, device)
            
    
    # freshly encode audios OR load precomputed audio embeddings
    audio_embedding_path = f'./datasets/{benchmark}/features/audio-embeddings-query-chunking.hdf5'
    
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
    ranking = compute_metrics(split_captions, audio_embeddings, split_text_embeddings)
    
    
    # save rank list to a pickle file
    ranking_path = f'./datasets/{benchmark}/clap_ranking/audio-ranking-query-chunking.pkl'
    
    with open(ranking_path, 'wb') as f:
        pickle.dump(ranking, f)
    
    print('\nRanking list saved...')
    
        
    # load the ranking from the pickle file
    with open(ranking_path, 'rb') as f:
        ranking = pickle.load(f)

    print('ranking:', ranking)

    # extract top 5 retrieved audios
    extract_ranking(benchmark, ranking_path, result_path)

    return
    
        


if __name__ == '__main__':

    # select benchmark to evaluate
    benchmark = 'Clotho_LARCQ'  # Clotho_LARCQ, SoundDescs_LARCQ, <Your Own Dataset>
    
    # select GPU
    device = torch.device('cuda:1')
    print('Using device:', device)
    
    # load CLAP model 
    # for auios > 10 seconds
    # fusion: split audios into three chunks, extract features, and then fuse features
    clap_model_name = 'clap-htsat-fused'
    clap_model = ClapModel.from_pretrained("./models/" + clap_model_name , local_files_only=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained("./models/" + clap_model_name, local_files_only=True)
    feature_extractor = AutoFeatureExtractor.from_pretrained("./models/" + clap_model_name, local_files_only=True)   
    
    # execute the main process
    result_path = f'results/retrieved_results/{benchmark}/retrieved_audios_query_chunking.csv'

    evaluate(benchmark, result_path)

    print('Done!')