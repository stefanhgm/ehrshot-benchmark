# External validation on UKB

In the following, the steps for performing the external validation on the UK Biobank data are provided.

### Environment

Due to different requirements of the different models, at least two conda environments have to be created. 

LLM2Vec - for general use

```python
#Requires python=3.10.16
conda env create -f environment_llm2vec.yml
```

Qwen - for creation of Qwen3 embeddings

```python
# Requires python=3.10.18
conda env create -f environment_qwen.yml
```

## Validation on UKB and CLMBR

### Embedding creation UKB

**Required data**

- UKB data is usable in case you are an eligible researcher.
- Different tables have to be present in specific location (change location path in [LM2Vec.py](http://LM2Vec.py) - UKB_data_path)
    - big_dataset: dataportal_final_records_omop_240625_mapped_eids_inpatient_updated
        - generated using raw UKB data, mapped following the code from https://github.com/JakobSteinfeldt/MedicalHistoryPhenomeWide/tree/cb5d2c1d60b5fe479d88f1d498f0a610b846f9b9/1_data_preparation
        - After creating larger table with patient information, data is checked that no patient information is going over the admission date - which would be data leakage (Inpatient_mapping.py)
    - length_of_stay_path file: contains information of patient’s length of stay in hospital - taken from UKB
    - covariates: File containing [eid, gender, ethnic_background, birth_date] of patients. Columns should be named ['eid', 'sex_f31_0_0', 'ethnic_background_f21000_0_0', 'birth_date']
- Huggingface access to models LLM2Vec, gte-Qwen2-7B-instruct and Qwen3-Embedding-8B
    - In [LLM2Vec.py](http://LLM2Vec.py) script the HF token has to be included
    - Clone LLM2Vec into Project folder (https://github.com/McGill-NLP/llm2vec) - required in LLM2Vec_embeddingscreation.py to generate LLM2Vec embeddings
    - HF models:
        - https://huggingface.co/McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised
        - https://huggingface.co/Alibaba-NLP/gte-Qwen2-7B-instruct
        - https://huggingface.co/Qwen/Qwen3-Embedding-8B

**Steps**

- Generate embeddings for all models (LLM2Vec, Qwen2 and Qwen3)
    - bash script LLM2Vec_run_diseases.sh can be used. Here, most importantly, select the model (LLM2Vec/Qwen(2)/Qwen3) for which to calculate the embeddings for. In case only inference should be performed, just select the inference option.
- Save all embeddings in a folder

### Embedding creation CLMBR

**Required data**

- Access to CLMBR data can be gained by applying for it (for more visit https://som-shahlab.github.io/ehrshot-website/)
- After gaining access, download weights:
    - https://huggingface.co/StanfordShahLab/clmbr-t-base
- Download athena library from https://athena.ohdsi.org/search-terms/start
- UKB data:
    - dataportal_final_records_omop_240625_mapped_eids(_inpatient_updated*)* (see required data for UKB)
    - covariates file (see required data for UKB)
- After downloading, data was cleaned using the script …

**Steps**

- Having all data in the correct location, the following scripts can be run for calculating the evaluation for CLMBR. Both scripts are in folder clmbr_baseline. Two steps are required where 1. UKB data is mapped to the format usable by CLMBR and 2. this converted data is used for creating embeddings. Bash scripts clmbr_baseline.sh and clmbr_calc_embeddings.sh contain a parameter configuration for how to run the programs.
- First, run script clmbr_baseline.py for converting UKB data into clmbr format (generates feather files) - folder has to be adapted to your structure
- Use this file to create embeddings for the clmbr baseline using script clmbr_baseline_create_embeddings.py (also change folder path here). This script has to be run twice for generating 1. data in MEDS format and using this format in clmbr to 2. generate the embeddings. (For this the value of Calculate_MEDS_format_only has to be set to True or False)

### Evaluation

- Results are written into folders “images” and “Tables”, in which a folder “death”, “disease_onset” and “hospitalization” should be present.
- For evaluation, the bash script LLM2Vec_run_diseases.sh can be used

### Paper Figure creation

- Figure generation was done using script “Final_experiment_results.ipynb” in the “Scripts” folder. Potentialy change the parameters on top of the script if max_token_length is different or auprc/ auroc should be calculated. Required Folder “Results_Paper” for result files is generated in code as well
