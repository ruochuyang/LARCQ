import csv
import pandas as pd
import re
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm 



def cross_encoder_ranking(data_path, ALM):
    
    data = pd.read_csv(data_path).to_dict('records')
    
    for idx in tqdm(range(len(data))):
        
        item = data[idx]
            
        query = item['query']

        features = tokenizer([query, query, query, query, query], [item[f'{ALM}_response_1'], item[f'{ALM}_response_2'], item[f'{ALM}_response_3'], item[f'{ALM}_response_4'], item[f'{ALM}_response_5']], padding=True, truncation=True, return_tensors="pt")

        model.eval()

        with torch.no_grad():
            scores = np.array(model(**features).logits)
   
        item['cross_encoder_rank'] = np.argmax(scores) + 1

    result_path = f'results/final_results/{benchmark}/{ALM}_cross_encoder_ranking.csv'
        
    pd.DataFrame(data).to_csv(result_path, index=False)

    return
        
        
        
if __name__ == "__main__":

    # select benchmark to evaluate
    benchmark = 'Clotho_LARCQ'  # Clotho_LARCQ, SoundDescs_LARCQ, <Your Own Dataset>

    # select ALM
    ALM = 'gama' # gama, flamingo
    
    # load miniLM model
    model = AutoModelForSequenceClassification.from_pretrained('models/ms-marco-MiniLM-L-6-v2')
    tokenizer = AutoTokenizer.from_pretrained('models/ms-marco-MiniLM-L-6-v2')
    
    data_path = f'results/alm_results/{benchmark}/retrieved_audios_{ALM}.csv' 
    
    cross_encoder_ranking(data_path, ALM)
    
    print('\nDone!')
