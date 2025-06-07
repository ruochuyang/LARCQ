
# Prerequisite

## 1. Configure environments

```
conda create -n larcq python=3.10
conda activate larcq
pip install -r requirements.txt
pip install -e hf-dev-train/transformers-main
pip install -e peft-main
```

## 2. Download benchmarks
Download Clotho_LARCQ and SoundDescs_LARCQ benchmarks from Hugging Face [benchmark link](https://huggingface.co/datasets/ruochuyang/LARCQ). Save the benchmarks in the `datasets` folder.

You can also download save any dataset you want to evaluate.

## 3. Nvidia GPUs
The results in the paper are generated in a computer with Nvidia GPUs. Better to configure `nvidia-smi` ready.


# Run Pipeline

Our pipeline consists of two main parts: multi-modal retrieval and ALM/LLM refining.

## 1. Run multi-modal rertieval

Download the `clap-htsat-fused` model from the Hugging Face [model link](https://huggingface.co/laion/clap-htsat-fused). Save the model in the `models` folder.

Download the `gpt2` model from the Hugging Face [model link](https://huggingface.co/openai-community/gpt2). Save the model in the `models` folder.

The retrieval scripts are in the folder `pipeline/multi_modal_retrieval`. Each script is independent and can be directly executed, which means that you can evaluate any method on any dataset for comprehensive comparison.

(1)`retrieval_no_chunking.py` is to retrieve the relevant audios given the queries without any audio chunking or query chunking applied.  
Run terminal command `python -m pipeline.multi_modal_retrieval.retrieval_no_chunking`  
retrieved short-list audios are saved as `results/retrieved_results/{benchmark}/retrieved_audios_no_chunking.csv`

(2)`retrieval_audio_chunking.py` is to retrieve the relevant audios given the queries with only audio chunking max/sum vote and without any query chunking.  
Run terminal command `python -m pipeline.multi_modal_retrieval.retrieval_audio_chunking`  
retrieved short-list audios are saved as `results/retrieved_results/{benchmark}/retrieved_audios_audio_chunking.csv`

(3)`retrieval_query_chunking.py` is to retrieve the relevant audios given the queries with only query chunking max/sum vote and without any audio chunking.  
Run terminal command `python -m pipeline.multi_modal_retrieval.retrieval_query_chunking`  
retrieved short-list audios are saved as `results/retrieved_results/{benchmark}/retrieved_audios_query_chunking.csv`

(4)`retrieval_audio_chunking_query_chunking.py` is to apply the four combinations of  `audio chunking max vote × query chunking sum vote`, `audio chunking sum vote × query chunking sum vote`, `audio chunking sum vote × query chunking max vote`, `audio chunking max vote × query chunking max vote` to retrieve the audios.  
Run terminal command `python -m pipeline.multi_modal_retrieval.retrieval_audio_chunking_query_chunking`  
retrieved short-list audios are saved as `results/retrieved_results/{benchmark}/retrieved_audios_best.csv`

## 2. Run ALM/LLM refining

### 2.1 Run ALM captioning on the retrieved audios

In our paper, we use two ALMs, GAMA and Audio-Flamingo, to generate captions for the retrieved audios.

(1) Downlowad the `Llama-2-7b-chat-hf-qformer` folder from the Google Drive [website link](https://drive.google.com/drive/u/0/folders/1W8ZtlhXNZ2IdVcKWsQpLD4jVw98brYDM). Save the folder in the `models` folder.

Downlowad the `stage5_epoch2` folder from the Google Drive [website link](https://drive.google.com/drive/u/0/folders/1W8ZtlhXNZ2IdVcKWsQpLD4jVw98brYDM). Unzip the folder and save the entire folder in the `models` folder.

Run terminal command `python -m pipeline.alm_llm_refining.run_gama`

GAMA captioning results on the retrieved audios are saved as `results/alm_results/{benchmark}/retrieved_audios_gama.csv`

(2) Downlowad the `clapcap_weights_2023.pth` checkpoint from the Hugging Face [website link](https://huggingface.co/microsoft/msclap/blob/main/clapcap_weights_2023.pth). Save the checkpoint in the `models` folder.

Downlowad the `opt-iml-max-1.3b` folder from the Hugging Face [website link](https://huggingface.co/facebook/opt-iml-max-1.3b). Save the entire folder in the `models` folder.

Downlowad the `foundation.pt` checkpoint from the Hugging Face [website link](https://huggingface.co/nvidia/audio-flamingo). Save the checkpoint in the `models` folder.

Run terminal command `python -m pipeline.alm_llm_refining.run_flamingo`

Audio-Flamingo captioning results on the retrieved audios are saved as `results/alm_results/{benchmark}/retrieved_audios_flamingo.csv`

### 2.2 Run LLM Re-ranking on the retrieved audios

In our paper, we use LLM or miniLM to compare the ALM generated response with the text query. You can use any LLM pr miniLM model you want.

(1) Use LLM 

* In our paper, we use Mixtral as LLM. Follow the tutorial on the Mistral AI [website link](https://docs.mistral.ai/deployment/self-deployment/vllm/) to set up Mixtral.

install the `vllm` package (version `>=0.6.1.post1` to ensure maximum compatibility with all Mistral models).

authenticate on the HuggingFace Hub using your access token `$HF_TOKEN` :

```huggingface-cli login --token $HF_TOKEN```

* Choose an ALM captioning file `results/alm_results/{benchmark}/retrieved_audios_{ALM}.csv`, like `results/alm_results/Clotho_LARCQ/retrieved_audios_gama.csv`

* Run terminal command `python -m pipeline.alm_llm_refining.llm_ranking` 
LLM re-ranking results are saved as `results/llm_results/{benchmark}/{ALM}_llm_ranking.csv`


(2) Use miniLM

* Downlowad the `ms-marco-MiniLM-L-6-v2` folder from the Hugging Face [website link](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2/tree/main). Save the entire folder in the `models` folder.

* Choose an ALM captioning file `results/alm_results/{benchmark}/retrieved_audios_{ALM}.csv`, like `results/alm_results/Clotho_LARCQ/retrieved_audios_gama.csv`

* Run terminal command `python -m pipeline.alm_llm_refining.cross_encoder_ranking` 
LLM re-ranking results are saved as `results/llm_results/{benchmark}/{ALM}_cross_encoder_ranking.csv`


## 3. Evaluate final results
Finally, we evalute the following final results to obtain all the metrics R@1 and R@5 in our paper.

`benchmark = Clotho_LARCQ, SoundDescs_LARCQ`
`ALM = gama, flamingo`
LLM results: `results/llm_results/{benchmark}/{ALM}_llm_ranking.csv`
miniLM results: `results/llm_results/{benchmark}/{ALM}_cross_encoder_ranking.csv`

Run terminal command `python -m evaluate_final_result` 


# LARCQ Benchmark Generation

We provide our code of generating our Clotho_LARCQ benchmark based on Clotho Version 2.1 dataset so that you can follow it to create any LARCQ benchmark you want.

(1) Downlowad the `clotho_audio_evaluation.7z` folder and the `clotho_captions_evaluation.csv` file from the Zenodo [website link](https://zenodo.org/records/4783391). Save them in the `datasets/Clotho` folder.

(2) Synthesize long-audio-long-query pairs as LARCQ benchmarks

Run terminal command `python -m benchmark_generation.synthesize`

The raw LARCQ captions are saved as `datasets/Clotho_LARCQ/raw_LARCQ_captions.csv`
The LARCQ audios are saved as `'datasets/Clotho_LARCQ/audios/`

(3) Run LLMs to refine the raw LARCQ captions

We use two options to refine the raw LARCQ captions into natural long queries.

* Condense the raw captions
Run terminal command `python -m benchmark_generation.llm_condense`  
The condensed LARCQ captions are saved as `datasets/Clotho_LARCQ/condensed_caption.csv`

* Rephrase the raw captions
Run terminal command `python -m benchmark_generation.llm_rephrase`  
The rephrased LARCQ captions are saved as `datasets/Clotho_LARCQ/rephrased_caption.csv`