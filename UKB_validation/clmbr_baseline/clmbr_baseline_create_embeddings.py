from datetime import datetime
import os
import pandas as pd
import pathlib
from collections import Counter
from tqdm import tqdm

import sys
import torch
from torch import autocast

import numpy as np
import gc

import pickle

import FEMR_functions

Calculate_MEDS_format_only = False

print("Starting...")

# Add the path to sys.path
# project_root = os.path.abspath("../../UKB_CLMBR/femr/src/femr/")
# if project_root not in sys.path:
#     sys.path.append(project_root)

import femr.models.transformer
import femr.models.tokenizer
import femr.models.processor

shared_cache_dir = "/sc-projects/sc-proj-dh-ag-eils-ml/shared_hf_cache/hub" #if models are saved in specific directory, otherwise comment out
os.environ["HF_HOME"] = shared_cache_dir
hf_token_path = os.path.expanduser("~/.huggingface/token") #add your own path to hf token
with open(hf_token_path, "r") as f:
    os.environ["HF_TOKEN"] = f.read().strip()
os.environ['TRANSFORMERS_CACHE'] = shared_cache_dir


#Load model
model_name = "StanfordShahLab/clmbr-t-base"

# Load tokenizer / batch loader
tokenizer = femr.models.tokenizer.FEMRTokenizer.from_pretrained(model_name)
batch_processor = femr.models.processor.FEMRBatchProcessor(tokenizer)

# Load model
model = femr.models.transformer.FEMRModel.from_pretrained(model_name)



if(Calculate_MEDS_format_only):
    ## Read in own data
    data_path = pathlib.Path(
        "/sc-projects/sc-proj-ukb-cvd/data/3_datasets_post/231012_ukb_preprocessing/ukb_data_portal/2_final"
    )

    covariates_path = data_path / "baseline_covariates_231016.feather"
    covariates_df = pd.read_feather(covariates_path, columns=['eid', 'sex_f31_0_0', 'ethnic_background_f21000_0_0', 'birth_date'])
    covariates_df.head(), covariates_df.shape

    clmbr_UKB_mapping_complete_path = "/sc-projects/sc-proj-ukb-cvd/projects/llm2vec/data/filtered_records_mapped_clmbrwithnames.feather"
    clmbr_UKB_mapping_complete = pd.read_feather(clmbr_UKB_mapping_complete_path)

    ## convert ethnicity to clmbr format
    data_ethnicity_to_clmbr = {
        "-3": None,  # Prefer not to answer
        "-1": None,  # Do not know
        "1": 5,  # White -> White
        "2": None,  # Mixed -> No direct equivalent
        "3": 2,  # Asian or Asian British -> Asian
        "4": 3,  # Black or Black British -> Black or African American
        "5": 2,  # Chinese -> Considered part of Asian
        "6": None,  # Other ethnic group -> No direct match
        "1001": 5,  # British -> White
        "1002": 5,  # Irish -> White
        "1003": 5,  # Any other white background -> White
        "2001": None,  # White and Black Caribbean -> Mixed (No direct equivalent)
        "2002": None,  # White and Black African -> Mixed (No direct equivalent)
        "2003": None,  # White and Asian -> Mixed (No direct equivalent)
        "2004": None,  # Any other mixed background -> Mixed (No direct equivalent)
        "3001": 2,  # Indian -> Asian
        "3002": 2,  # Pakistani -> Asian
        "3003": 2,  # Bangladeshi -> Asian
        "3004": 2,  # Any other Asian background -> Asian
        "4001": 3,  # Caribbean -> Black or African American
        "4002": 3,  # African -> Black or African American
        "4003": 3,  # Any other Black background -> Black or African American
    }

    covariates_df["clmbr_race"] = covariates_df["ethnic_background_f21000_0_0"].map(data_ethnicity_to_clmbr)


    print("Reading in UKB data...")
    clmbr_UKB_mapping_complete = clmbr_UKB_mapping_complete.join(covariates_df[["eid", "birth_date", "clmbr_race"]].set_index("eid"), on="eid")
    print("Finished reading in UKB data...")



    # add gender to the records in specified notation - add with birthdate
    # add for each eid individually to df filtered_records
    def add_gender(eid, birth_date, gender):
        gender_code = "F" if gender == 'Female' else "M"
        return {"eid": eid, "birth_date": birth_date, "code": gender_code, "source": "Gender", "date": birth_date}

    filtered_records_unique = (
        clmbr_UKB_mapping_complete.drop_duplicates(subset=["eid"])
        .merge(covariates_df[["eid", "sex_f31_0_0"]], on="eid", how="left")
    )

    # Create a new DataFrame with the additional rows
    new_rows = filtered_records_unique.apply(
        lambda row: add_gender(row["eid"], row["birth_date"], row["sex_f31_0_0"]),
        axis=1,
    )

    new_rows_df = pd.DataFrame(new_rows.tolist())

    # Append new_rows_df to the original filtered_records DataFrame
    filtered_records_mapped = pd.concat([clmbr_UKB_mapping_complete, new_rows_df], ignore_index=True)
    filtered_records_mapped["code"] = filtered_records_mapped["code"].astype(str)
    filtered_records_mapped["birth_date"] = filtered_records_mapped["birth_date"].astype(str)
    filtered_records_mapped['date'] = pd.to_datetime(filtered_records_mapped['date'])
    filtered_records_mapped['birth_date'] = pd.to_datetime(filtered_records_mapped['birth_date'])  # Convert birth_date to datetime

    # filter out patients without birthdate
    patients_filter = filtered_records_mapped[filtered_records_mapped["birth_date"].isna()]["eid"].unique()
    filtered_records_mapped = filtered_records_mapped[~filtered_records_mapped["eid"].isin(patients_filter)]
    print(filtered_records_mapped["birth_date"].isna().sum())


    # Function to build the desired structure with a birthdate event
    def build_patient_structure_with_birthdate(df):

        # Adjust the code to prepend "SNOMED/" to the code and convert date strings to datetime objects
        df['clmbr_code'] = df["source"] + "/" + df["code"]

        patients = []
        grouped = df.groupby('eid')
        

        for eid, patient_records in grouped:
            events = []
            
            # Add the birthdate event
            birth_date = patient_records['birth_date'].iloc[0]  # Get the birthdate for the patient
            events.append({
                'time': datetime(birth_date.year, birth_date.month, birth_date.day),  # Ensure it's a datetime object
                'measurements': [{'code': 'SNOMED/184099003'}]
            })

            # Add the ethnicity event
            birth_date = patient_records['birth_date'].iloc[0]  # Get the birthdate for the patient
            race_val = patient_records['clmbr_race'].iloc[0]  # Get the race val for patient
            if race_val.is_integer():
                raceval = str(int(race_val))
                events.append({
                    'time': datetime(birth_date.year, birth_date.month, birth_date.day),  # Ensure it's a datetime object
                    'measurements': [{'code': f'Race/{raceval}'}]
                })
            
            # Add all other events - needs to be sorted by date
            for date, date_records in patient_records.groupby('date', sort=True):
                measurements = [{'code': row['clmbr_code']} for _, row in date_records.iterrows()]
                events.append({
                    'time': datetime(date.year, date.month, date.day),  # Ensure it's a datetime object
                    'measurements': measurements
                })
            
            # Append the patient record
            patients.append({
                'patient_id': eid,
                'events': events
            })
        
        return patients


    # Create the structure
    print("Building patient structure...")
    MEDS_schema = build_patient_structure_with_birthdate(filtered_records_mapped)
    # with open("result.txt", "w") as f:
    #     for item in MEDS_schema:
    #         f.write(f"{item}\n")

    # Save patient data to a pickle file
    def save_patients_to_pickle(patients, filename):
        with open(filename, 'wb') as f:
            pickle.dump(patients, f)


    # Example usage
    save_patients_to_pickle(MEDS_schema, '/sc-projects/sc-proj-ukb-cvd/projects/llm2vec/data/clmbr_ukb_meds.pkl')

else:
    # Load MEDS schema
    #with open('patients.pkl', 'rb') as f:
    with open('/sc-projects/sc-proj-ukb-cvd/projects/llm2vec/data/clmbr_ukb_meds.pkl', 'rb') as f:
        MEDS_schema = pickle.load(f)
    
    #MEDS_schema = MEDS_schema[0:1000]
    #MEDS_schema = MEDS_schema[81012:81014]  # For testing
    tokens_per_batch = 5000

    print("Calculating batches...")
    # This code works for converting batches from a list of patients
    batch_processor = FEMR_functions.FEMRBatchProcessor(tokenizer)
    batches = batch_processor.convert_dataset_from_list_with_synthetic_labels(MEDS_schema, tokens_per_batch=tokens_per_batch, min_patients_per_batch=1) #previously 512

    # for testing
    #batches = batch_processor.convert_dataset(MEDS_schema, tokens_per_batch=1024, min_patients_per_batch=1)

    del MEDS_schema  # Free memory after processing
    gc.collect()
    torch.cuda.empty_cache()  # If using GPU


    def combine_results(results):
        latest_records = {}

        # return df
        for record in results:
            for pid, rep in zip(record['patient_ids'], record['representations']):
                latest_records[pid.item()] = rep  # We assume timestamps are ordered, so last seen wins

        return pd.DataFrame({
            'eid': list(latest_records.keys()),
            'q_reps': list(latest_records.values())
        })

    print("Calculating embeddings...")

    results = []

    temp_dir = f"/sc-projects/sc-proj-ukb-cvd/projects/llm2vec/data/embeddings/temp_batches_{tokens_per_batch}"
    os.makedirs(temp_dir, exist_ok=True)

    batch_count = 0  # Track batch numbers

    # Convert to tensor format
    #tensor_batches = []

    #output_path_csv = "/sc-projects/sc-proj-ukb-cvd/projects/llm2vec/data/embeddings/embeddings_clmbr_1024-1.csv"
    output_path_feather = f"/sc-projects/sc-proj-ukb-cvd/projects/llm2vec/data/embeddings/embeddings_clmbr_{tokens_per_batch}-1.feather"

    #with open (output_path_feather, 'wb') as filename:
    for batch in tqdm(batches, desc="Processing", unit="batch"):
        tensor_batch = {}  # Wrap everything inside 'batch'
        
        for key, value in batch.items():
            if isinstance(value, dict):
                tensor_batch[key] = {
                    k: torch.tensor(v) if isinstance(v, (list, np.ndarray, torch.Tensor)) else torch.tensor([v])
                    for k, v in value.items()
                }
            elif isinstance(value, (list, np.ndarray, torch.Tensor)):
                tensor_batch[key] = torch.tensor(value, dtype=torch.float32)
            else:
                tensor_batch[key] = torch.tensor([value])  # Convert single values to tensor


        # for i in tqdm(range(0, len(tensor_batches))):
        batch = batch_processor.collate([tensor_batch])

        # Run model
        with torch.no_grad():
            _, result = model(**batch)

        df = combine_results([result])  # Convert only the current batch
        df["q_reps"] = df["q_reps"].apply(lambda x: x.tolist())

        # Save each batch separately
        batch_file = os.path.join(temp_dir, f"batch_{batch_count}.feather")
        df.to_feather(batch_file)
        batch_count += 1

           


    # **After processing all batches, merge into one file**
    batch_files = [os.path.join(temp_dir, f) for f in sorted(os.listdir(temp_dir))]

    dfs = [pd.read_feather(f) for f in batch_files]
    final_df = pd.concat(dfs, ignore_index=True)
    final_df.to_feather(output_path_feather)

    # Clean up temporary files (optional)
    for f in batch_files:
        os.remove(f)
    os.rmdir(temp_dir)




    