import gc
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pathlib
from functools import reduce

import re
#from lifelines import CoxPHFitter
#from lifelines.utils import concordance_index
import pyarrow as pa
import wandb

#from sklearn.decomposition import PCA

import argparse
from sklearn.manifold import TSNE
from tqdm import tqdm
#import optuna
import random
import torch
from torch import Tensor
from transformers import AutoModel

import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
#from sentence_transformers import SentenceTransformer

import polars as pl
import sys
import time
from sklearn.metrics import roc_auc_score

sys.path.append('./LLM2Vec_project/')  # Path where LLM2Vec.py - main script is located

#leave out these imports in case only testsplits are calculated (only requires cpu)
def get_cuml():
    try:
        #from llm2vec import LLM2Vec
        import cuml
        from cuml.decomposition import PCA
        from cuml.linear_model import LogisticRegression
        from cuml.metrics import accuracy_score
        return True
    except ImportError:
        print("Cuml is not installed. Using CPU instead.")
        return False

import LLM2Vec_functions as llm2vec
import LLM2Vec_load_process_data as load_process_data
from LLM2Vec_embeddingscreation import process_embeddings





hf_token_path = os.path.expanduser("/home/gear11/.huggingface/token") #change hf token to your own token
with open(hf_token_path, "r") as f:
    os.environ["HF_TOKEN"] = f.read().strip()



def main(instruction, task, **kwargs):

    data_path = pathlib.Path(
        "/sc-projects/sc-proj-ukb-cvd/data/3_datasets_post/231012_ukb_preprocessing/ukb_data_portal/2_final"
    )
    embedding_path = "/sc-projects/sc-proj-ukb-cvd/projects/llm2vec/data/embeddings/" #change to your own path - here, all embeddings should be stored
    UKB_data_path = "/sc-projects/sc-proj-ukb-cvd/projects/llm2vec/data/"

    #if kwargs["model"] != "Qwen3": #might not be required since we are loading it in LLM2Vec_functions.py
    #    get_cuml()

    # Path components based on configuration
    path_components = {
        'datespath': "dates" if kwargs["include_dates"] else "nodates",
        'querypath': "query" if kwargs["includequeries"] else "noquery",
        'diseasespecific': "diseaseunspecific" if kwargs["diseaseunspecific"] else "diseasespecific", #only for image and table
        'bigdataset': "bigdataset_" if kwargs["use_big_dataset"] else "rawdataset_" if kwargs["use_raw_dataset"] else "",
        'balanced': "balanced_" if kwargs["balanced"] else "", #only for image and table
        'tabpfn_path': "tabpfn_" if kwargs["tabpfn"] else "", #only for image and table
        'gbm_path': "gbm_" if kwargs["gbm"] else "",
        'tokenlength': f"{kwargs['tokenlength']}_" if kwargs["tokenlength"] != 4096 else "",
        'agesexincluded': "agesexincluded_" if kwargs["add_agesex_tensor"] else "",
        'clmbrcodes': "clmbrcodes_" if kwargs["clmbrcodes"] else "",
        'serialized': "notserialized_" if not kwargs["ehr_format"] else "",
    }

    if(not kwargs["infer_all"]):
        if(not kwargs["diseaseunspecific"]):
            #use diseaseunspecific embeddingfile
            embeddingfile = f"{embedding_path}embeddings_{task}_{path_components['datespath']}_{path_components['querypath']}_" + kwargs["phecode"] + "_" + kwargs["indication"] + "_" + kwargs["model"] + kwargs["phecode"] + str(int(12*kwargs["minyears"])) + "-" + str(kwargs["maxyears"]) + f"_{path_components['tokenlength']}.feather"
        else:
            #embeddingfile = f"{embedding_path}embeddings_{task}_{path_components['bigdataset']}{path_components['serialized']}{path_components['tokenlength']}{path_components['datespath']}_{path_components['querypath']}_{path_components['clmbrcodes']}" + kwargs["model"] + str(int(12*kwargs["minyears"])) + "-" + str(kwargs["maxyears"]) + ".feather" 
            embeddingfile = f"{embedding_path}embeddings_{task}_{path_components['bigdataset']}{path_components['tokenlength']}{path_components['datespath']}_{path_components['querypath']}_{path_components['clmbrcodes']}" + kwargs["model"] + str(int(12*kwargs["minyears"])) + "-" + str(kwargs["maxyears"]) + '_' + str(kwargs["start"]) + '_' + str(kwargs["end"]) + ".feather" #for calculating in smaller parts
        wandb.config.embeddingfile = embeddingfile
        print(embeddingfile)
    else:
        embeddingfile_qwen = f"{embedding_path}embeddings_{task}_{path_components['bigdataset']}{path_components['tokenlength']}{path_components['datespath']}_{path_components['querypath']}_Qwen" + str(int(12*kwargs["minyears"])) + "-" + str(kwargs["maxyears"]) + ".feather"
        embeddingfile_qwen3 = f"{embedding_path}embeddings_{task}_{path_components['bigdataset']}{path_components['tokenlength']}{path_components['datespath']}_{path_components['querypath']}_Qwen3" + str(int(12*kwargs["minyears"])) + "-" + str(kwargs["maxyears"]) + ".feather"
        embeddingfile_llm2vec = f"{embedding_path}embeddings_{task}_{path_components['bigdataset']}{path_components['tokenlength']}{path_components['datespath']}_{path_components['querypath']}_LLM2Vec" + str(int(12*kwargs["minyears"])) + "-" + str(kwargs["maxyears"]) + ".feather"
        embeddingfile_nvembed = f"{embedding_path}embeddings_{task}_{path_components['bigdataset']}{path_components['tokenlength']}{path_components['datespath']}_{path_components['querypath']}_{path_components['clmbrcodes']}NVEmbed" + str(int(12*kwargs["minyears"])) + "-" + str(kwargs["maxyears"]) + ".feather"
        embeddingfile_clmbr = f"{embedding_path}embeddings_clmbr_5000-1.feather"


    config = dict(
        # Model settings
        batch_size=kwargs["batch_size"],
        num_cv_rounds = kwargs["cv_rounds"],
        max_number_testsets = 10000,
        device="cuda" if torch.cuda.is_available() else "cpu",
        #num_samples=kwargs["num_samples"], 
        num_patients_test=kwargs["pat_test_size"],
        years_threshold=kwargs["maxyears"], #number of years to consider in the future for logistic regression
        years_threshold_min = int(12*kwargs["minyears"]), #months from recruitment date to consider patients as diseased
        selected_phecode=kwargs["phecode"], 
        disease=kwargs["indication"], #disease of interest in full name
        withPCA=kwargs["withPCA"] if kwargs["withPCA"] != None else 0, #number of PCA components
        addagesex=kwargs["add_agesex_tensor"], #add age and sex to embeddings/ counts table
        #instruction=f"You are a medical expert. Given the medical history of a person (prior hospitalisations, diagnoses, medications, and lab results){disease_description}, predict the risk of developing {disease} in the future." 
        instruction=instruction,
        lr= True, #kwargs["lr"], # use logistic regression for classification
        useNVEmbed = False, #kwargs["useNVEmbed"], # use NVEmbed embeddings
        clmbrethnicities = True, #kwargs["clmbrethnicities"], # use CLMBR embeddings
        clmbrcodes = kwargs["clmbrcodes"], #kwargs["clmbrcodes"], # use CLMBR embeddings

        # Feature settings
        with_dates=kwargs["include_dates"],
        with_query=kwargs["includequeries"],
        random_seed=42,
        num_codes_in_records=50,
        balanced=kwargs["balanced"],
        tabpfn=kwargs["tabpfn"],
        gbm=kwargs["gbm"],

        # Dataset options
        use_big_dataset = kwargs["use_big_dataset"],
        use_raw_dataset = kwargs["use_raw_dataset"],
        diseaseunspecific = kwargs["diseaseunspecific"],

        # Data paths
        covariates_path = data_path / "baseline_covariates_231016.feather",
        standard_dataset_path =  f"{UKB_data_path}dataportal_final_records_omop_240625_mapped_eids_inpatient_updated.feather",
        big_dataset_path = f"{UKB_data_path}dataportal_final_records_omop_240625_mapped_eids_inpatient_updated.feather", #from inpatient_mapping.py file
        big_dataset_path_clmbr = f"{UKB_data_path}clmbr_ukb_mapping.feather",
        raw_dataset_path = "/sc-projects/sc-proj-ukb-cvd/projects/llm2vec/messy/messy_data_before_recruitment.feather", #changed sourcepath
        length_of_stay_path = f"{UKB_data_path}codes_admin_231012_mapped_eids_filtered.feather",
    
        embeddingfile_qwen = embeddingfile_qwen if "embeddingfile_qwen" in locals() else None,
        embeddingfile_qwen3 = embeddingfile_qwen3 if "embeddingfile_qwen3" in locals() else None,
        embeddingfile_llm2vec = embeddingfile_llm2vec if "embeddingfile_llm2vec" in locals() else None,
        embeddingfile_nvembed = embeddingfile_nvembed if "embeddingfile_nvembed" in locals() else None,
        embeddingfile_clmbr = embeddingfile_clmbr if "embeddingfile_clmbr" in locals() else None,
        embeddingfile = embeddingfile if "embeddingfile" in locals() else None,
    )


    # read in records and covariates data
    records, selected_covariates, hospital_stay_length, patients_future = load_process_data.load_data(config)


    # select only necessary columns from records dataframe
    if(kwargs["use_big_dataset"]):
        records = pl.from_pandas(records[["eid", "date", "concept_name", "concept_id", "recruitment_date", "ehr_class", "code", "vocabulary_id"]])
    else:  
        if(kwargs["use_raw_dataset"]):
            records = pl.from_pandas(records[["eid", "date", "concept_name", "concept_id", "recruitment_date", "ehr_class", "code", "vocabulary_id"]])
        else:
            records = pl.from_pandas(records[["eid", "date", "concept_name", "concept_id", "recruitment_date"]])

    # Continue with the rest of the processing - check that date, recruitment_date and concept_name are not null (all required for embedding creation/ inference)
    records = records.filter(
        pl.col("date").is_not_null()
        & pl.col("recruitment_date").is_not_null()
        & pl.col("concept_name").is_not_null()
    )

    #filter out patients if calculation should be diseasespecific
    if((not kwargs["diseaseunspecific"]) or ((kwargs["diseaseunspecific"]) & (not kwargs["calculate_embeddings"]))):
        records = records.with_columns(
            pl.col("date").str.to_datetime("%Y-%m-%d"),  # Adjust format as needed
            pl.col("recruitment_date").str.to_datetime("%Y-%m-%d")
        )
        if(config["disease"] != "hospitalization"):
            # Filter out patients that have the disease before the recruitment date (+min years)

            patients_with_disease_before_recruitment = set(records.filter(
                (pl.col("concept_id") == config["selected_phecode"]) & 
                (pl.col("date") < pl.col("recruitment_date").dt.offset_by(str(config["years_threshold_min"])+"m"))
            )["eid"])
            records = records.filter(~pl.col("eid").is_in(patients_with_disease_before_recruitment))

        else:
            #do not filter anything
            patients_with_disease_before_recruitment = {} 
            # set(records.filter( 
            #     (pl.col("concept_id") == config["selected_phecode"]) & 
            #     (pl.col("date") > pl.col("recruitment_date")) & 
            #     (pl.col("date") < pl.col("recruitment_date").dt.offset_by(str(config["years_threshold_min"])+"m"))
            # )["eid"])
            # records = records.filter(~pl.col("eid").is_in(patients_with_disease_before_recruitment))


    # select patients having at least one entry before the recruitment date
    eids_with_prior_records = (
        records.filter(pl.col("date") < pl.col("recruitment_date"))
        .group_by("eid")
        .len()
        .filter(pl.col("len") > 0)
        .select("eid")
    )

    #if(not kwargs["use_raw_dataset"]): #check for patients with future records - so only patients with entries and therefore known disease state will be considered in prediction
    if(type(records["date"][0]) == str):
        records = records.with_columns(
            pl.col("date").str.to_datetime("%Y-%m-%d"),  # Adjust format as needed
            pl.col("recruitment_date").str.to_datetime("%Y-%m-%d")
        )
    if(kwargs["use_big_dataset"] or kwargs["use_raw_dataset"]):
        eids_with_future_records = pl.DataFrame({
            "eid": patients_future
        })
    else:
        # select patients having at least one entry after the recruitment date
        eids_with_future_records = (
            records.filter(pl.col("date") >= pl.col("recruitment_date").dt.offset_by(str(config["years_threshold_min"])+"m"))
            .group_by("eid")
            .len()
            .filter(pl.col("len") > 0)
            .select("eid")
        )

    # save all eids of patients with records both prior and after their recruitment
    filter_eids = eids_with_prior_records.join(eids_with_future_records, on="eid", how="inner").select(
        "eid"
    )
    print("Number of patients after filtering: ", len(filter_eids))

    # sort the filter_eids
    filter_eids = filter_eids.sort("eid")

    # filter the initial records such that only eids remain that have diagnoses prior or after to recruitment
    filtered_records = records.filter(pl.col("eid").is_in(filter_eids["eid"]))

    # Convert 'eid' column to int
    filtered_records = filtered_records.with_columns(
        pl.col("eid").cast(pl.Int32)
    )


    # Continue with the rest of the processing
    # remove all records in the table in which the date of the condition/ diagnosis occurs prior to the recruitment date
    filtered_records = filtered_records.filter((pl.col("date") < pl.col("recruitment_date")))

    # if calculations should be done without dates, remove all "duplicated" medical codes occurring on different dates and only keep one
    if(not config["with_dates"]):
        # Filter out duplicates based on 'eid' and 'concept_id'
        filtered_records = filtered_records.unique(subset=['eid', 'concept_name'])



    # Ensure correct data types
    filtered_records = filtered_records.with_columns(
        [
            pl.col("date").cast(pl.Date),
            pl.col("recruitment_date").cast(pl.Date),
            pl.col("concept_name").cast(pl.Utf8),
            pl.col("eid").cast(pl.Utf8),
        ]
    )

    # Calculate years ago if specified in the config
    #if config["with_dates"]:
    filtered_records = filtered_records.with_columns(
        [((pl.col("recruitment_date") - pl.col("date")).dt.total_days() / 365.25).alias("years_ago")]
    )

    # Format the records
    filtered_records = filtered_records.with_columns(
        pl.concat_str([
            pl.when(config["with_dates"])
                #.then(pl.col("years_ago").cast(pl.Float64).round(2).cast(pl.Utf8))
                .then(pl.col("date"))
                .otherwise(pl.lit("")),
            pl.when(config["with_dates"])
                .then(pl.lit(": "))
                .otherwise(pl.lit("")),
            pl.col("concept_name")
        ]).alias("formatted_record")
    )

    if(kwargs["use_big_dataset"] & kwargs["ehr_format"]):
        # Group by eid and aggregate
        prepared_records = filtered_records.group_by("eid").agg([
            pl.col("recruitment_date").first().alias("recruitment_date"),

            pl.col("formatted_record")
            .filter(pl.col("ehr_class") == "Conditions")
            .alias("condition_records"),

            pl.col("formatted_record")
            .filter(pl.col("ehr_class") == "Medications")
            .alias("medication_records"),
            
            pl.col("formatted_record")
            .filter(pl.col("ehr_class") == "Procedures")
            .alias("procedure_records")

        ])
    else:
        if(kwargs["use_raw_dataset"] & kwargs["ehr_format"]):
            # Group by eid and aggregate
            prepared_records = filtered_records.group_by("eid").agg([
                pl.col("recruitment_date").first().alias("recruitment_date"),

                pl.col("formatted_record")
                .filter(pl.col("ehr_class") == "Procedures")
                .alias("procedure_records"),

                pl.col("formatted_record")
                .filter(pl.col("ehr_class") == "Conditions")
                .alias("condition_records"),

                pl.col("formatted_record")
                .filter(pl.col("ehr_class") == "Medications")
                .alias("medication_records"),

                pl.col("formatted_record")
                .filter(pl.col("ehr_class") == "Lab Values")
                .alias("labval_records")
            ])
        else:
            # Group by eid and aggregate
            prepared_records = filtered_records.group_by("eid").agg(
                [
                pl.col("recruitment_date").first().alias("recruitment_date"),
                pl.col("formatted_record").alias("record_list")
                ]
            )

    # Convert selected_covariates to a Polars DataFrame
    selected_covariates_polars = pl.from_pandas(selected_covariates)

    # Ensure the data types match for the join
    prepared_records = prepared_records.with_columns(pl.col("eid").cast(pl.Int32))
    selected_covariates_polars = selected_covariates_polars.with_columns(pl.col("eid").cast(pl.Int32))

    # Merge prepared_records with selected_covariates
    prepared_records = prepared_records.join(selected_covariates_polars, on='eid', how='left')

    prepared_records = prepared_records.with_columns([
        pl.col("recruitment_date")
        .map_elements(llm2vec.datetime_date_to_markdown)  # Use map_elements instead of apply
        .alias("recruitment_date")
    ])

    if(kwargs["calculate_embeddings"]):
        if(kwargs["ehr_format"] == True):

            ## test new representation of queries
            prepared_records = prepared_records.with_columns([
                pl.format(
                    "\n"
                    "# Electronic Healthcare Record: \n\n"
                    "Current time: {} \n\n"
                    "## Patient Demographics \n"
                    "- Patient age: {} \n"
                    "- {} \n"
                    "- {} \n\n",
                    pl.col("recruitment_date"),
                    pl.col("age"),
                    pl.col("sex"),
                    pl.col("ethnicity_name").fill_null("Unknown"),
                ).alias("new_vis")
            ])

            if(kwargs["include_dates"]):
                # Add medical history in specific format in case dates are present
                if(kwargs["use_big_dataset"]):
                    # Add medical history in specific format in case dates are present
                    prepared_records = prepared_records.with_columns([
                        pl.struct(["procedure_records", "condition_records", "medication_records", "recruitment_date", "eid"])
                        .map_elements(lambda x: llm2vec.format_medical_history(x["procedure_records"], x["condition_records"], x["medication_records"], "Procedures", x["recruitment_date"], x["eid"], hospital_stay_length))
                        .alias("formatted_history")
                    ])
                else:
                    if(kwargs["use_raw_dataset"]):
                        # Add medical history in specific format in case dates are present
                        prepared_records = prepared_records.with_columns([
                            pl.struct(["labval_records", "condition_records", "medication_records", "procedure_records", "recruitment_date", "eid"])
                                .map_elements(lambda x: llm2vec.format_medical_history_rawdata(x["labval_records"], x["condition_records"], x["medication_records"], x["procedure_records"], x["recruitment_date"], x["eid"], hospital_stay_length
                                ), return_dtype=pl.String
                            )
                            .alias("formatted_history")
                        ])
                    else:
                        # Add medical history in specific format in case dates are present
                        prepared_records = prepared_records.with_columns([
                            pl.struct(["record_list"])
                            .map_elements(lambda x: llm2vec.format_medical_history_single(x["record_list"]))
                            .alias("formatted_history")
                        ])

                # Then, combine this new information with the existing data
                prepared_records = prepared_records.with_columns([
                    pl.struct(["formatted_history", "new_vis"])
                    .map_elements(lambda x: (
                        config["instruction"],# + " " +
                        x["new_vis"] + " " +
                        x["formatted_history"]
                    ))
                    .alias("queries")
                ])

            else:
                if(kwargs["use_big_dataset"]):
                    # If without dates, combine this new information with the existing data
                    prepared_records = prepared_records.with_columns([
                        pl.struct(["procedure_records", "condition_records", "medication_records", "new_vis"])
                        .map_elements(lambda x: (
                            config["instruction"],# + " " +
                            x["new_vis"] + " " +
                            "## Medical History\n" +
                            "### Conditions\n" +
                            "\n".join(f"- {condition.strip()}" for condition in x["condition_records"]),  # Iterate over the list directly
                            "### Medications\n" +
                            "\n".join(f"- {medication.strip()}" for medication in x["medication_records"]),  # Iterate over the list directly
                            "### Procedures\n" +
                            "\n".join(f"- {procedure.strip()}" for procedure in x["procedure_records"])  # Iterate over the list directly
                        ))
                        .alias("queries")
                    ])
                else:
                    if(kwargs["use_raw_dataset"]):
                        # If without dates, combine this new information with the existing data
                        prepared_records = prepared_records.with_columns([
                            pl.struct(["labval_records", "condition_records", "medication_records", "procedure_records", "new_vis"])
                            .map_elements(lambda x: (
                                config["instruction"],# + " " +
                                x["new_vis"] + " " +
                                "## Medical History\n" +
                                "### Conditions\n" +
                                "\n".join(f"- {condition.strip()}" for condition in x["condition_records"]),  # Iterate over the list directly
                                "### Medications\n" +
                                "\n".join(f"- {medication.strip()}" for medication in x["medication_records"]),  # Iterate over the list directly
                                "### Procedures\n" +
                                "\n".join(f"- {procedure.strip()}" for procedure in x["procedure_records"]),  # Iterate over the list directly
                                "### Lab values\n" +
                                "\n".join(f"- {procedure.strip()}" for procedure in x["labval_records"])  # Iterate over the list directly
                            ))
                            .alias("queries")
                        ])
                    else:
                        # If without dates, combine this new information with the existing data
                        prepared_records = prepared_records.with_columns([
                            pl.struct(["record_list", "new_vis"])
                            .map_elements(lambda x: (
                                config["instruction"],# + " " +
                                x["new_vis"] + " " +
                                "## Medical History\n" +
                                "\n".join(f"- {record.strip()}" for record in x["record_list"])  # Iterate over the list directly
                            ))
                            .alias("queries")
                        ])

            #print(prepared_records.head()["queries"][0])

        else:
            # First, create a new column with the demographic information
            prepared_records = prepared_records.with_columns([
                pl.format(
                    "The person is {} years old and {} and has the following medical history:",
                    pl.col("age"),
                    pl.col("sex")
                ).alias("demographic_info")
            ])

            # Then, combine this new information with the existing data
            prepared_records = prepared_records.with_columns([
                pl.struct(["record_list", "demographic_info"])
                .map_elements(lambda x: (
                        config["instruction"],
                        x["demographic_info"] + " " +
                        ", ".join(x["record_list"])
                    )
                )
                .alias("queries")
            ])

    prepared_records = prepared_records.to_pandas()


    # Calculate or load embeddings
    if(kwargs["calculate_embeddings"]):
        prepared_records.sort_values(by="eid", inplace=True)
        #embedding_df = process_embeddings(prepared_records[kwargs["start"]:min(kwargs["end"], len(prepared_records))], config, **kwargs) #include in case embeddings should be calculated in smaller patient subsets
        embedding_df = process_embeddings(prepared_records, config, **kwargs)
        
        # Return early if disease unspecific and only calculating embeddings
        if((kwargs["calculate_embeddings"]) and (kwargs["diseaseunspecific"])):
            return
    else:
        embedding_df = process_embeddings(prepared_records, config, **kwargs)





    records = records.to_pandas()
    selected_eids = pl.DataFrame(filtered_records["eid"].unique())
    selected_eids = selected_eids.with_columns(pl.col("eid").cast(pl.Int32))
    censoring_times = (
        records.groupby("eid").first()[["recruitment_date"]].loc[selected_eids["eid"]].copy()
    )

    #censoring date set to the last date present in the data
    censoring_times["censoring_date"] = pd.to_datetime("2022-12-19")
    censoring_times["censoring_time"] = (
        censoring_times["censoring_date"] - censoring_times["recruitment_date"]
    ).dt.days / 365.25


    times = []
    exclusions = []


    phecode_name = config["selected_phecode"]

    outcomes = llm2vec.get_phenotypes(records, phecode_name, censoring_times).loc[selected_eids["eid"]]

    #events.append(outcomes[f"{phecode_name_short}_event"].rename(phecode_name_short))
    times.append(outcomes[f"{phecode_name}_time"].rename(phecode_name))
    exclusions.append(outcomes[f"{phecode_name}_prior"].rename(phecode_name))

    #events = pd.concat(events, axis=1)
    times = pd.concat(times, axis=1)
    exclusions = pd.concat(exclusions, axis=1)


    target = (times[config["selected_phecode"]] <= config["years_threshold"]) & (times[config["selected_phecode"]] > (config["years_threshold_min"]/12)).astype(int)
    #target.value_counts()  


    ## Perform prediction for Llama model
    if((kwargs["model"] == "Llama")):
        merged_df = pd.merge(target, embedding_df[['eid', 'probability', 'yes_proba', 'no_proba']], 
                left_index=True, right_on='eid', how='left')
        clean_df = merged_df.dropna()

        auroc = roc_auc_score(clean_df[target.name], clean_df['probability'])
        print(auroc)
        wandb.log({"auroc_with_ratio": auroc})
        clean_df.to_csv(f"Llama_results/{kwargs['model']}_probabilities_{kwargs['indication']}_ratio.csv", index=False)
        return

    

    # from filtered_records, filter out the entries that only occur less than 50 times (need to check if some patients do not have any remaining entries)
    # also before add ontology extension if wanted
    # Count occurrences of each concept code combined with vocabulary_id
    if(config["use_big_dataset"]):
        filtered_records = filtered_records.with_columns(
            (filtered_records["vocabulary_id"].cast(str) + "/" + filtered_records["code"].cast(str)).alias("codes")
        )
    else:
        filtered_records = filtered_records.with_columns(
            (filtered_records["concept_id"].cast(str)).alias("codes")
        )


    # remove columns in filtered_records that are not required anymore
    filtered_records = filtered_records[["eid", "codes"]]
    #filtered_records.unique()
    filtered_records = filtered_records.to_pandas()
    

    # Add ontology extension for all codes in filtered_records for big_dataset
    if(config["use_big_dataset"]):
        # MAKE PRETTIER! - For now: add ontology expansion for all codes in filtered_records
        code_to_parents = {}
        with open("Data_preprocessing/mappings/code_to_parent_mapping.csv", "r") as f:
            # Skip header
            next(f)
            for line in f:
                code, parent_code = line.strip().split(",")
                if code not in code_to_parents:
                    code_to_parents[code] = []
                code_to_parents[code].append(parent_code)

        mapping_records = [
            (child, parent)
            for child, parents in code_to_parents.items()
            for parent in parents
        ]
        df_mapping = pd.DataFrame(mapping_records, columns=['codes', 'parent_code'])

        # Step 2: Merge to bring original row data to parents
        df_parents = df_mapping.merge(filtered_records, on='codes', how='inner')

        # Step 3: Replace 'code' with 'parent_code' for parent rows
        df_parents['codes'] = df_parents['parent_code']
        df_parents = df_parents.drop(columns=['parent_code'])

        # Step 4: Concatenate original and parent rows
        filtered_records = pd.concat([filtered_records, df_parents], ignore_index=True)
        filtered_records = filtered_records.drop_duplicates(subset=['eid', 'codes'])
        filtered_records = pl.from_pandas(filtered_records)
        

        concept_counts = (
            filtered_records
            .group_by('codes')
            .len()
            .rename({"len": "occurrences"})  # Renaming for clarity
        )

        concept_counts = concept_counts.with_columns(pl.col("occurrences").cast(pl.Int32))

        # Filter concepts appearing at least `num_codes_in_records` times
        valid_concepts = concept_counts.filter(
            pl.col("occurrences") >= config["num_codes_in_records"]
        )["codes"]  # Extract valid concept_ids

        # Filter the DataFrame to keep only those rows
        valid_concepts = valid_concepts.to_list()

        filtered_records = filtered_records.filter(
            pl.col("codes").is_in(valid_concepts)
        )

        ## Add ontology extension - precalculated for two hops
        filtered_records = filtered_records.to_pandas()


        ## filter out patients that have no entries after removing the ones with less than 50 occurrences
        eids_with_records = set(list(map(int, filtered_records["eid"])))
        prepared_records = prepared_records[prepared_records["eid"].isin(eids_with_records)]



    prepared_records["eid"] = prepared_records["eid"].astype(int)
    #read in the prepared records (with queries) and add embedding vectors
    if(not kwargs["infer_all"]):
        prepared_records = prepared_records.merge(embedding_df[['eid', 'q_reps']], on='eid', how='left')
    else:
        columns_to_include = ['eid', 'q_reps_qwen', 'q_reps_qwen3', 'q_reps_llm2vec', 'q_reps_clmbr']
        if(config["useNVEmbed"]):
            columns_to_include.append('q_reps_nvembed')
        prepared_records = prepared_records.merge(embedding_df[columns_to_include], on='eid', how='left')


    records["eid"] = records["eid"].astype(int)
    filtered_records["eid"] = filtered_records["eid"].astype(int)

    selected_covariates['sex'] = selected_covariates['sex'].cat.rename_categories({'Male': 0, 'Female': 1})
    
    selected_covariates["age"] = (
        selected_covariates["age"] - selected_covariates["age"].mean()
    ) / selected_covariates["age"].std()

    # Create a new column 'age_sex_tensor' that combines 'age' and 'sex' into a tensor
    selected_covariates['age_sex_tensor'] = selected_covariates.apply(lambda row: torch.tensor([row['age'], row['sex']], dtype=torch.float32), axis=1)
    selected_covariates

    # Merge the dataframes on 'eid'
    prepared_records = prepared_records.merge(selected_covariates[['eid', 'age_sex_tensor']], on='eid', how='left')
    
    def combine_tensors_inplace(df, colname, batch_size=1000):
        total_rows = len(df)
            
        for i in range(0, total_rows, batch_size):
            # Create a Series first, then assign
            batch_indices = df.index[i:i + batch_size]
            batch = df.loc[batch_indices]
            
            combined_tensors = pd.Series([
                torch.cat((
                    torch.from_numpy(q) if isinstance(q, np.ndarray) else q,
                    torch.from_numpy(a) if isinstance(a, np.ndarray) else a
                ), dim=0)
                for q, a in zip(batch[colname], batch['age_sex_tensor'])
            ], index=batch_indices)
            
            # Update using loc with proper indexing
            df.loc[batch_indices, colname] = combined_tensors
            
            if i % 10000 == 0:
                print(f"Processed {i}/{total_rows} rows")
                torch.cuda.empty_cache()
                gc.collect()

    #Filter out patients that have nan values in q_reps column
    if(not kwargs["infer_all"]):
        prepared_records = prepared_records.dropna(subset=['q_reps'])
    else:
        #prepared_records = prepared_records.dropna(subset=['q_reps_qwen', 'q_reps_llm2vec'])
        columns_to_include = ['q_reps_qwen', 'q_reps_qwen3', 'q_reps_llm2vec', 'q_reps_clmbr']
        if(config["useNVEmbed"]):
            columns_to_include.append('q_reps_nvembed')
        prepared_records = prepared_records.dropna(subset=columns_to_include)



    # Process tensors efficiently in batches
    if(config["addagesex"]):
        print("Combining demographics and initial embeddings")
        if(not kwargs["infer_all"]):
            combine_tensors_inplace(prepared_records, "q_reps")
        else:
            combine_tensors_inplace(prepared_records, "q_reps_qwen")
            combine_tensors_inplace(prepared_records, "q_reps_qwen3")
            combine_tensors_inplace(prepared_records, "q_reps_llm2vec")
            if(config["useNVEmbed"]):
                combine_tensors_inplace(prepared_records, "q_reps_nvembed")
            combine_tensors_inplace(prepared_records, "q_reps_clmbr")
        print("Finished combining demographics and initial embeddings")
    else:
        print("Did not append demographics to initial embeddings")
    



    if(config["balanced"]):
        #check target.sum() eids and eventually filter out eids not present anymore
        target = target[target.index.isin(prepared_records["eid"])]

        # in case number of positive samples is too low, only select smaller sample_size_steps
        if target.sum() < 75: #(test split = 32, train split = 32)
            max_range = 5
        elif target.sum()< 150: #(test split = 64, train split = 64)
            max_range = 6
        elif target.sum() < 300: #(test split = 128, train split = 128)
            max_range = 7
        elif target.sum() < 600: #(test split = 256, train split = 256)
            max_range = 8  # Default to full range
        else:
            max_range = 9  # Default to full range

        # Generate the sample size steps based on the determined range
        sample_size_steps = [2**i for i in range(0, max_range)] 
        
        # calculate number of positive and negative test samples for prediction
        num_test_samples = target.sum() - 2*(2**(max_range-1))
        num_test_samples = min(config["max_number_testsets"], num_test_samples-1) #limit number of test samples to 10.000
    
    else:
        sample_size_steps = [10**i for i in range(1, 6)] #1, 6 = 10 to 100.000

    if(kwargs["get_dataset_overview"]):
        # Save to file: name of disease, number of patients, number of test samples, number of positive patients
        with open("Patnums.txt", "a") as f:
            f.write(f"{config['selected_phecode']}, {config['disease']}, {len(prepared_records)}, {num_test_samples}, {target.sum()}\n")

    #check if images and Tables folders exist for results
    def ensure_dirs(base, subfolders):
        """Ensure base directory and all subfolders exist."""
        for sub in [""] + subfolders:
            os.makedirs(os.path.join(base, sub), exist_ok=True)

    # Check that images and Tables folders exist for results
    ensure_dirs("../images", ["death", "hospitalization", "disease_onset"])
    ensure_dirs("../tables", ["death", "hospitalization", "disease_onset"])

    scores = []
    testpatnum = -1
    for sample_size in reversed(sample_size_steps): 
        print(sample_size)

        for i in range(config["num_cv_rounds"]):

            testsplit = [] #testsplits[i]

            trainsplit, valsplit = llm2vec.split_train_val_test(
                #prepared_records["eid"], config, sample_size, target, testsplit, onlytest=False
                list({int(eid) for eid in prepared_records["eid"]}), config, sample_size, target, testsplit, config["balanced"], onlytest=False
            )

            if(config["balanced"]):
                testsplit = llm2vec.split_train_val_test_balanced(list({int(eid) for eid in prepared_records["eid"]}), target, trainsplit.tolist() + valsplit.tolist(), num_test_samples)
            


            print("calculating Counts")
            #run program for counts method
            auroc_counts, auprc_counts, trainpatnum, testpatnum = (
                    llm2vec.process_medical_records_counts(times, filtered_records, selected_covariates, trainsplit, valsplit, testsplit, min(sample_size, config["withPCA"]), config, target)
                )
            print(f"auroc counts in sample size {sample_size}: {auroc_counts}")   
    

            print("calculating LLM2Vec")
            #run program for llm2vec
            auroc_llm2vec, auprc_llm2vec = (
                    llm2vec.process_medical_records_llm(times, prepared_records, trainsplit, valsplit, testsplit, min(sample_size, config["withPCA"]), config, target, q_reps_name="q_reps_llm2vec")
                )
            print(f"auroc llm2vec in sample size {sample_size}: {auroc_llm2vec}") 
                    

            print("calculating Qwen")
            #run program for Qwen
            auroc_qwen, auprc_qwen = (
                    llm2vec.process_medical_records_llm(times, prepared_records, trainsplit, valsplit, testsplit, min(sample_size, config["withPCA"]), config, target, q_reps_name="q_reps_qwen")
                )
            print(f"auroc Qwen in sample size {sample_size}: {auroc_qwen}") 


            print("calculating Qwen3")
            #run program for Qwen3
            auroc_qwen3, auprc_qwen3 = (
                    llm2vec.process_medical_records_llm(times, prepared_records, trainsplit, valsplit, testsplit, min(sample_size, config["withPCA"]), config, target, q_reps_name="q_reps_qwen3")
                )
            print(f"auroc Qwen3 in sample size {sample_size}: {auroc_qwen3}")
   
            if(config["useNVEmbed"]):
                print("calculating NVEmbed")
                #run program for NVEmbed
                auroc_nvembed, auprc_nvembed = (
                        llm2vec.process_medical_records_llm(times, prepared_records, trainsplit, valsplit, testsplit, min(sample_size, config["withPCA"]), config, target,  q_reps_name="q_reps_nvembed")
                    )
                print(f"auroc NVEmbed in sample size {sample_size}: {auroc_nvembed}") 
                              

            print("calculating CLMBR")
            #run program for CLMBR
            auroc_clmbr, auprc_clmbr = (
                    llm2vec.process_medical_records_llm(times, prepared_records, trainsplit, valsplit, testsplit, min(sample_size, config["withPCA"]), config, target, q_reps_name="q_reps_clmbr")
                )
            print(f"auroc CLMBR in sample size {sample_size}: {auroc_clmbr}") 
            


            #run program for agesex method
            auroc_agesex, auprc_agesex = (
                    llm2vec.process_medical_records_agesex(times, prepared_records, trainsplit, valsplit, testsplit, min(sample_size, config["withPCA"]), config, target)
                )
            print(f"auroc agesex in sample size {sample_size}: {auroc_agesex}")

            

            score_entry = dict(
                sample_size=sample_size,
                crossval_iteration=i,
                patnum_train=trainpatnum,
                auroc_llm2vec=auroc_llm2vec,
                auroc_qwen=auroc_qwen,
                auroc_qwen3=auroc_qwen3,
                auroc_clmbr=auroc_clmbr,
                auroc_counts=auroc_counts,
                auroc_agesex=auroc_agesex,
                auprc_llm2vec=auprc_llm2vec,
                auprc_qwen=auprc_qwen,
                auprc_qwen3=auprc_qwen3,
                auprc_clmbr=auprc_clmbr,
                auprc_counts=auprc_counts,
                auprc_agesex=auprc_agesex,
                testpatnum=testpatnum,
            )
            

            if(config["useNVEmbed"]):
                score_entry["auroc_nvembed"]=auroc_nvembed,
                score_entry["auprc_nvembed"]=auprc_nvembed,
        
            scores.append(score_entry)

        scores_df = pd.DataFrame(scores)
        values_to_include = ["auroc_llm2vec", "auroc_qwen", "auroc_qwen3", "auroc_clmbr", "auroc_counts", "auroc_agesex", "auprc_llm2vec", "auprc_qwen", "auprc_qwen3", "auprc_clmbr", "auprc_counts", "auprc_agesex"]
        if(config["useNVEmbed"]):
            values_to_include.append("auroc_nvembed")
            values_to_include.append("auprc_nvembed")

        scores_pivot = scores_df.pivot_table(index="sample_size", values=values_to_include, aggfunc="mean")
    
        wandb.log({
            f"{kwargs['indication']}/auroc_llm2vec": scores_pivot.loc[sample_size, 'auroc_llm2vec'],
            f"{kwargs['indication']}/auroc_qwen": scores_pivot.loc[sample_size, 'auroc_qwen'],
            f"{kwargs['indication']}/auroc_qwen3": scores_pivot.loc[sample_size, 'auroc_qwen3'],
            f"{kwargs['indication']}/auroc_clmbr": scores_pivot.loc[sample_size, 'auroc_clmbr'],
            f"{kwargs['indication']}/auroc_agesex": scores_pivot.loc[sample_size, 'auroc_agesex'],
            f"{kwargs['indication']}/auroc_counts": scores_pivot.loc[sample_size, 'auroc_counts']
        }, step=sample_size)

        if (config["useNVEmbed"]):
            wandb.log({
                f"{kwargs['indication']}/auroc_nvembed": scores_pivot.loc[sample_size, 'auroc_nvembed']
            }, step=sample_size)

    results_df = pd.DataFrame(scores)

    modelname = kwargs["model"]


    num_cv_rounds = config["num_cv_rounds"]

    if(config["withPCA"] > 0):
        PCA_addition = "_PCA" + str(config["withPCA"]) + "_"
    else:
        PCA_addition = ""

    # save table to csv file 
    results_df.to_csv(f"tables/{task}/table_{path_components['agesexincluded']}{path_components['bigdataset']}{path_components['tokenlength']}{path_components['balanced']}{path_components['tabpfn_path']}{path_components['gbm_path']}{path_components['datespath']}_{path_components['querypath']}_{path_components['clmbrcodes']}" + kwargs["phecode"] + "_" + kwargs["indication"] + "_" + modelname + f"_{num_cv_rounds}_{path_components['diseasespecific']}_" + str(config["years_threshold_min"]) + "-" + str(config["years_threshold"]) + PCA_addition + f"_8192_balancedtestset_newserialization.csv", index=False)

    # Compute the mean or median of patnum_train for each sample_size
    aggregated_data = results_df.groupby("sample_size")["patnum_train"].agg(np.mean).reset_index()

    if(not config["balanced"]):

        # Set seaborn style
        sns.set(style="whitegrid")

        # Create the figure and axis
        fig, ax1 = plt.subplots(figsize=(8, 6))  # Adjust figure size if needed
        # Primary plot (AUROC vs. Sample Size)
        sns.lineplot(data=results_df, x="sample_size", y="auroc_qwen", label="Qwen", ax=ax1)
        sns.lineplot(data=results_df, x="sample_size", y="auroc_qwen3", label="Qwen3", ax=ax1)
        sns.lineplot(data=results_df, x="sample_size", y="auroc_llm2vec", label="LLM2Vec", ax=ax1)
        if(config["useNVEmbed"]):
            sns.lineplot(data=results_df, x="sample_size", y="auroc_nvembed", label="NVEmbed", ax=ax1)
        sns.lineplot(data=results_df, x="sample_size", y="auroc_clmbr", label="CLMBR", ax=ax1)
        sns.lineplot(data=results_df, x="sample_size", y="auroc_counts", label="Counts", ax=ax1)
        sns.lineplot(data=results_df, x="sample_size", y="auroc_agesex", label="AgeSex", ax=ax1)

        # Set log scale for x-axis
        ax1.set_xscale('log', base=10)
        ax1.set_xlabel("Sample Size train set (Total number of diseased Patients)", labelpad=-5)  # Add padding to primary x-axis label
        ax1.set_ylabel("AUROC")
        ax1.set_title(f"AUROC vs. Sample Size in " + kwargs["indication"] + f"\nNumber of test pats: {testpatnum}")

        # Create the secondary x-axis
        ax2 = ax1.twiny()  # Create a twin x-axis
        ax2.spines["top"].set_position(("axes", -0.15))  # Move the secondary axis farther down
        ax2.spines["top"].set_visible(True)  # Hide the top spine of the secondary axis

        # Set ticks and labels for the secondary x-axis
        ax2.set_xscale('log', base=10)
        ax2.set_xlim(ax1.get_xlim())  # Ensure the scale matches the primary x-axis
        ax2.set_xticks(aggregated_data["sample_size"])
        ax2.set_xticklabels([f"{int(diseased)}" for diseased in aggregated_data["patnum_train"]])
        ax2.set_xlabel("Number of Diseased Patients", labelpad=-40)  # Add padding to secondary x-axis label
        
        # Adjust layout to avoid cutting off the second x-axis
        fig.subplots_adjust(bottom=0.3)  # Increase space at the bottom of the plot
        plt.tight_layout()
    
    else:
        # Set seaborn style
        sns.set(style="whitegrid")

        #only keep one x-axis in case we use balanced set since num positive samples = num samples
        plt.figure(figsize=(8, 5))
        sns.lineplot(data=results_df, x="sample_size", y="auroc_qwen", label="Qwen",marker="o")
        sns.lineplot(data=results_df, x="sample_size", y="auroc_qwen3", label="Qwen3",marker="o")
        sns.lineplot(data=results_df, x="sample_size", y="auroc_llm2vec", label="LLM2Vec", marker="o")
        if(config["useNVEmbed"]):
            sns.lineplot(data=results_df, x="sample_size", y="auroc_nvembed", label="NVEmbed", marker="o")
        sns.lineplot(data=results_df, x="sample_size", y="auroc_clmbr", label="CLMBR", marker="o")
        sns.lineplot(data=results_df, x="sample_size", y="auroc_counts", label="Counts", marker="o")
        sns.lineplot(data=results_df, x="sample_size", y="auroc_agesex", label="AgeSex", marker="o")
        plt.xscale('log', base=2)
        plt.xlabel("Sample Size (log2 scale)")
        plt.ylabel("AUROC")
        plt.title(f"AUROC vs. Sample Size in " + kwargs["indication"] + f"\nNumber of test pats: {testpatnum}")

    #save image to file
    plt.savefig(f"images/{task}/image_{path_components['agesexincluded']}{path_components['bigdataset']}{path_components['tokenlength']}{path_components['balanced']}{path_components['tabpfn_path']}{path_components['gbm_path']}{path_components['datespath']}_{path_components['querypath']}_{path_components['clmbrcodes']}" + kwargs["phecode"] + "_" + kwargs["indication"] + "_" + modelname + f"_{num_cv_rounds}_{path_components['diseasespecific']}_" + str(config["years_threshold_min"]) + "-" + str(config["years_threshold"]) + PCA_addition + "_8192_balancedtestset_newserialization.png")


    return





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Your script description")
    parser.add_argument("--model", type=str, help="Model - choose from NVEmbed, LLM2Vec, Qwen(2) or Qwen3. For inference no model needs to be specified (but --infer_all).")
    parser.add_argument("--infer_all", type=lambda x: x.lower() == "true", nargs='?', const=True, default=False, help="Predict performance for all models at once (currently: Counts, Qwen, Qwen3, LLM2Vec (to do: potentially add NVEmbed)).")
    parser.add_argument("--indication", type=str, help="Diseasename (if task = disease_onset), otherwise choose from [death, hospitalization]")
    parser.add_argument("--phecode", type=str, help="Phecode (if task = disease_onset)")
    parser.add_argument("--minyears", type=float, help="Time in years from recruitment date to consider patients as diseased. Also float can be provided (will be calculated in months).", default=1.0)
    parser.add_argument("--maxyears", type=int, help="Time in years to consider patients diseased in the future for logistic regression", default=10)
    parser.add_argument("--tokenlength", type=int, help="Number of tokens used for embedding creation", default=4096)

    parser.add_argument("--include_dates", type=lambda x: x.lower() == "true", nargs='?', const=True, default=False)
    parser.add_argument("--includequeries", type=lambda x: x.lower() == "true", nargs='?', const=True, default=False)

    parser.add_argument("--tabpfn", type=lambda x: x.lower() == "true", nargs='?', const=True, default=False, help="Use tabpfn for prediction instead of logistic regression")
    parser.add_argument("--gbm", type=lambda x: x.lower() == "true", nargs='?', const=True, default=False, help="Use gradient boosting for Counts baseline instead of logistic regression")
    parser.add_argument("--use_big_dataset", type=lambda x: x.lower() == "true", nargs='?', const=True, default=False, help="Use the bigger dataset including the gp_clinical data")
    parser.add_argument("--use_raw_dataset", type=lambda x: x.lower() == "true", nargs='?', const=True, default=False, help="Use the raw data")
    parser.add_argument("--calculate_embeddings", action="store_true", help="calculate embeddings - e.g. to include disease name in query")
    parser.add_argument("--save_embeddings", action="store_true", help="Set to true if embeddings of run shall be stored.")
    parser.add_argument("--cv_rounds", type=int, help="Number of cross validation rounds for different patients (currently max 100)", default=10)
    parser.add_argument("--batch_size", type=int, help="Batchsize to calculate embeddings", default=10)
    parser.add_argument("--pat_test_size", type=int, help="Number of patients in test set", default=20000)
    #parser.add_argument("--num_samples", type=int, help="Number of sample patients used in run", default=269664)
    parser.add_argument("--ehr_format", type=lambda x: x.lower() == "true", nargs='?', const=True, help="Include if data should be in EHR format", default=True)
    #parser.add_argument("--calculate_testsplits", action="store_true", help="Include if only testsplits should be calculated. Also remember to include min- and maxyear + to create folder.")
    parser.add_argument("--balanced", action="store_true", help="Set to true if calculation train and test split should be balanced.")
    parser.add_argument("--diseaseunspecific", action="store_true", help="Perform calculation disease/ indication unspecific.")
    parser.add_argument("--add_agesex_tensor", type=lambda x: x.lower() == "true", nargs='?', const=True, default=False, help="Set to false to exclude age and sex tensor.")
    parser.add_argument("--clmbrcodes", type=lambda x: x.lower() == "true", nargs='?', const=True, default=False, help="Only use codes also used in clmbr.")
    parser.add_argument("--withPCA", type=int, help="Select if model should be caclulated with PCA and if yes, story the number of PCA components. Default: no PCA (0)", default=0)
    parser.add_argument("--disable_wandb", action="store_true", help="Set to true if wandb should be disabled - e.g. for debugging.")
    parser.add_argument("--get_dataset_overview", action="store_true", help="Set to true if information about task (e.g. dataset size, number of positive samples, .. should be saved to file Patnums.txt).")

    ## remove
    parser.add_argument("--start", type=int)
    parser.add_argument("--end", type=int)
    
    args = parser.parse_args()

    wandbname = "Run_" + args.model + "_" + args.indication
    wandb.login()

    if(args.disable_wandb):
        #wandb.init(project="LLM2Vec", entity="georg-von-arnim-berlin-institute-of-health", mode="disabled")
        wandb.init(project="LLM2Vec", entity="cardiors", mode="disabled")
    else:
        #wandb.init(project="LLM2Vec", entity="georg-von-arnim-berlin-institute-of-health", name=wandbname)
        wandb.init(project="LLM2Vec", entity="cardiors", name=wandbname)
        #wandb.init(project="LLM2Vec", entity="georg-von-arnim-berlin-institute-of-health", mode="disabled")



    # Log hyperparameters
    wandb.config.update(vars(args))

    #instruction = "Classify if the following patient is either likely or unlikely to develop " + args.indication + "in the next 10 years based on the following electronic healthcare record."
    instruction_prefix = "Given a patient's electronic healthcare record (EHR) in Markdown format, retrieve relevant passages that answer the query: "


    if(args.indication == "death"):
        instruction = f"Classify if the following patient is either likely or unlikely to die in the next {args.maxyears} years based on the following electronic healthcare record."
        instruction_task = f"will the patient die in the next {args.maxyears} year"
        task = "death"
        #args.phecode = "admin_death"
    elif(args.indication == "hospitalization"):
        instruction = f"Classify if the following patient is either likely or unlikely to be hospitalized in the next {args.maxyears} years based on the following electronic healthcare record."
        instruction_task = f"will the patient be hospitalized in the next {args.maxyears} year"
        task = "hospitalization"
        #args.phecode = "admin_hospital"
    else:    
        if(args.diseaseunspecific == True):
            instruction = f"Classify if the following patient is either likely or unlikely to develop/ have a medical condition in the next {args.maxyears} years based on the following electronic healthcare record."
            #instruction_task = f"Will the patient develop a medical condition in the next {args.maxyears} years?"
            instruction_task = f"has the patient a medical condition?"
        else:
            instruction = f"Classify if the following patient is either likely or unlikely to develop {args.indication} in the next {args.maxyears} years based on the following electronic healthcare record."
            instruction_task = f"Will the patient develop {args.indication} in the next {args.maxyears} years"
        task = "disease_onset"
    
    print(args.indication)

    #if includequeries not present in args or False -> change instruction to empty
    if(args.includequeries == False):
        instruction = " "

    instruction = instruction_prefix + instruction_task

    main(instruction, task, **vars(args))

    wandb.finish()