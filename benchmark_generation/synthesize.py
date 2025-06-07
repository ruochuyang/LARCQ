import csv
import pandas as pd
import librosa
import numpy as np
import random
from tqdm import tqdm
import soundfile as sf
import json


def load_data(caption_path, audio_path):
    
    info = pd.read_csv(caption_path)
    
    captions = {}
    
    audio_files = []
        
    for _, item in info.iterrows():
        
        audio_id = audio_path + str(item['file_name'])
        
        caption = item['caption_1'].removesuffix('.')
        
        audio_files.append(audio_id)
        
        captions[audio_id] = caption

    return captions, audio_files



def combine_audio(audio_files, output_file):

    audio1, sr1 = librosa.load(audio_files[0], sr=16000)
    
    combined_audio = np.array([])

    for audio_file in audio_files:

        audio, sr = librosa.load(audio_file, sr=16000)

        # ensure that the sample rates and number of channels match
        assert sr == sr1, f"Sample rate mismatch for file: {audio_file}"
        
        assert audio.ndim == audio1.ndim, f"Channel mismatch for file: {audio_file}"

        # concatenate the audio data
        combined_audio = np.concatenate((combined_audio, audio), axis=-1)

    # save new audio
    sf.write(output_file, combined_audio, sr1)

    return
    

    

def combine_caption(audio_files, captions):
    
    new_caption = ''
    
    for audio in audio_files:    
        new_caption = new_caption + captions[audio] + ', '
  
    new_caption = new_caption[:-2]

    new_caption = new_caption.lower()
        
    return new_caption
        
    

if __name__ == "__main__":
    
    caption_path = 'datasets/Clotho/clotho_captions_evaluation.csv'
    audio_path = 'datasets/Clotho/clotho_audio_evaluation/'
    
    captions, audio_files = load_data(caption_path, audio_path)
    
    # randomly select audio files to combine
    num_subsets = 1000

    subset_size = 5

    subsets = []
    for _ in range(num_subsets):
        subset = random.sample(audio_files, subset_size)
        subsets.append(subset)

    new_data = []

    for ii in tqdm(range(len(subsets))):

        subset = subsets[ii]

        new_audio_path = 'datasets/Clotho_LARCQ/audios/{}.wav'.format(ii+1) 

        combine_audio(subset, new_audio_path)

        new_caption = combine_caption(subset, captions)

        tmp = {'audio_id': '{}.wav'.format(ii+1),
               'caption': new_caption}

        for jj in range(len(subset)):
            tmp['split_' + str(jj+1)] = captions[subset[jj]]

        new_data.append(tmp)


    # save new data to csv file
    output_path = "datasets/Clotho_LARCQ/raw_LARCQ_captions.csv"

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['audio_id', 'caption']
        
        # Create CSV writer
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        # Write header
        writer.writeheader()
        
        # Write data rows
        writer.writerows(new_data)

    print('\nDone!')
