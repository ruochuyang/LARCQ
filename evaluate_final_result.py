import csv
import pandas as pd
import re
import numpy as np


def evaluate_flamingo_llm_ranking(benchmark):
    
    data_path = f'results/llm_results/{benchmark}/flamingo_llm_ranking.csv'
    data = pd.read_csv(data_path)
    data = data.to_dict('records')

    R1_count = 0
    R5_data = []
    
    for item in data:
        
        if item['retrieved_audio_1'] != item['groundtruth_audio']:
            R5_data.append(item)
        else:
            R1_count = R1_count + 1
                  
    # R1 boost out of R5 rank
    R1_boost = 0

    for item in R5_data:
        
        result = [item['retrieved_audio_1'], item['retrieved_audio_2'], item['retrieved_audio_3'], item['retrieved_audio_4'], item['retrieved_audio_5']]
        
        alm_rank_1 = result[item['llm_rank'] - 1]
         
        if alm_rank_1 == item['groundtruth_audio']:
            R1_boost = R1_boost + 1
            
    print('\n-------Flamingo LLM Ranking-------')
    print('\nCLAP R1:', round(R1_count / len(data) * 100), '%')
    print('\nFinal R1:', round((R1_count + R1_boost) / len(data) * 100), '%')
    print('\nR1 Boost:', round(R1_boost / R1_count * 100), '%')

    return
    


def evaluate_flamingo_cross_encoder_ranking(benchmark):
    
    data_path = f'results/llm_results/{benchmark}/flamingo_cross_encoder_ranking.csv'
    
    data = pd.read_csv(data_path)
    data = data.to_dict('records')
    
    R1_count = 0
    R5_data = []
    
    for item in data:
        
        if item['retrieved_audio_1'] != item['groundtruth_audio']:    
            R5_data.append(item)     
        else:
            R1_count = R1_count + 1
                
    # R1 boost out of R5 rank
    R1_boost = 0

    for item in R5_data:

        result = [item['retrieved_audio_1'], item['retrieved_audio_2'], item['retrieved_audio_3'], item['retrieved_audio_4'], item['retrieved_audio_5']]

        alm_rank_1 = result[item['cross_encoder_rank'] - 1]
        
        if alm_rank_1 == item['groundtruth_audio']:
            R1_boost = R1_boost + 1
            
    print('\n-------Flamingo Cross_encoder Ranking-------')
    print('\nCLAP R1:', round(R1_count / len(data) * 100), '%')
    print('\nFinal R1:', round((R1_count + R1_boost) / len(data) * 100), '%')
    print('\nR1 Boost:', round(R1_boost / R1_count * 100), '%')

    return
    
 
def evaluate_gama_llm_ranking(benchmark):
    
    data_path = f'results/llm_results/{benchmark}/gama_llm_ranking.csv'
    
    data = pd.read_csv(data_path)
    data = data.to_dict('records')
    
    R1_count = 0
    R5_data = []
    
    for item in data:
        
        if item['retrieved_audio_1'] != item['groundtruth_audio']:      
            R5_data.append(item)       
        else:
            R1_count = R1_count + 1
            
    # R1 boost out of R5 rank
    R1_boost = 0

    for item in R5_data:
        
        result = [item['retrieved_audio_1'], item['retrieved_audio_2'], item['retrieved_audio_3'], item['retrieved_audio_4'], item['retrieved_audio_5']]
        
        alm_rank_1 = result[item['llm_rank'] - 1]
        
        if alm_rank_1 == item['groundtruth_audio']:
            R1_boost = R1_boost + 1
    
    print('\n-------GAMA LLM Ranking-------')
    print('\nCLAP R1:', round(R1_count / len(data) * 100), '%')
    print('\nFinal R1:', round((R1_count + R1_boost) / len(data) * 100), '%')
    print('\nR1 Boost:', round(R1_boost / R1_count * 100), '%')

    return
    
    

def evaluate_gama_cross_encoder_ranking(benchmark):

    data_path = f'results/llm_results/{benchmark}/gama_cross_encoder_ranking.csv'
    
    data = pd.read_csv(data_path)
    data = data.to_dict('records')
    
    R1_count = 0
    R5_data = []
    
    for item in data:
        
        if item['retrieved_audio_1'] != item['groundtruth_audio']:        
            R5_data.append(item)       
        else:
            R1_count = R1_count + 1
            
    # R1 boost out of R5 rank
    R1_boost = 0

    for item in R5_data:

        result = [item['retrieved_audio_1'], item['retrieved_audio_2'], item['retrieved_audio_3'], item['retrieved_audio_4'], item['retrieved_audio_5']]
    
        alm_rank_1 = result[item['cross_encoder_rank'] - 1]  
        
        if alm_rank_1 == item['groundtruth_audio']:
            R1_boost = R1_boost + 1
            
    print('\n-------GAMA Cross_encoder Ranking-------') 
    print('\nCLAP R1:', round(R1_count / len(data) * 100), '%')
    print('\nFinal R1:', round((R1_count + R1_boost) / len(data) * 100), '%')
    print('\nR1 Boost:', round(R1_boost / R1_count * 100), '%')

    return
    

    

if __name__ == "__main__":

    # select benchmark to evaluate
    benchmark = 'Clotho_LARCQ'  # Clotho_LARCQ, SoundDescs_LARCQ, <Your Own Dataset>
    
    evaluate_flamingo_llm_ranking(benchmark)
    
    evaluate_flamingo_cross_encoder_ranking(benchmark)
    
    evaluate_gama_llm_ranking(benchmark)
    
    evaluate_gama_cross_encoder_ranking(benchmark)
    

    
    
   
