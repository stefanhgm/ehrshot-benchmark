import gc
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pathlib
from functools import reduce

from evaluation import main_evaluation

import scipy.sparse as sp
import wandb

import argparse
import random
import torch

import torch.nn.functional as F

import polars as pl
import sys

sys.path.append('./LLM2Vec_project/')  # Path where LLM2Vec.py - main script is located
from my_utils import TIMEBINS, TIMEBIN_LABELS

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

    # Path components based on configuration
    path_components = {
        'tokenlength': f"{kwargs['tokenlength']}_" if kwargs["tokenlength"] != 4096 else "",
        'clmbrcodes': "clmbrcodes_" if kwargs["clmbrcodes"] else "",
        'uniquecodes': "keep_all_codes_" if kwargs["keep_all_codes"] else "",
    }

    if(not kwargs["infer_all"]):
        embeddingfile = f"{embedding_path}embeddings_{task}_bigdataset_{path_components['tokenlength']}dates_query_{path_components['clmbrcodes']}{path_components['uniquecodes']}" + kwargs["model"] + str(int(12*kwargs["minyears"])) + "-" + str(kwargs["maxyears"]) + ".feather"
        print(embeddingfile)
    else:
        embeddingfile_qwen = f"{embedding_path}embeddings_{task}_bigdataset_{path_components['tokenlength']}dates_query_{path_components['clmbrcodes']}{path_components['uniquecodes']}_Qwen" + str(int(12*kwargs["minyears"])) + "-" + str(kwargs["maxyears"]) + ".feather"
        embeddingfile_qwen3 = f"{embedding_path}embeddings_{task}_bigdataset_{path_components['tokenlength']}dates_query_{path_components['clmbrcodes']}{path_components['uniquecodes']}_Qwen3" + str(int(12*kwargs["minyears"])) + "-" + str(kwargs["maxyears"]) + ".feather"
        embeddingfile_llm2vec = f"{embedding_path}embeddings_{task}_bigdataset_{path_components['tokenlength']}dates_query_{path_components['clmbrcodes']}{path_components['uniquecodes']}_LLM2Vec" + str(int(12*kwargs["minyears"])) + "-" + str(kwargs["maxyears"]) + ".feather"
        embeddingfile_clmbr = f"{embedding_path}embeddings_clmbr_5000-1.feather"


    config = dict(
        # Model settings
        batch_size=kwargs["batch_size"],
        #max_number_testsets = 10000,
        device="cuda" if torch.cuda.is_available() else "cpu",
        years_threshold=kwargs["maxyears"], #number of years to consider in the future for logistic regression
        years_threshold_min = int(12*kwargs["minyears"]), #months from recruitment date to consider patients as diseased
        selected_phecode=kwargs["phecode"], 
        disease=kwargs["indication"], #disease of interest in full name
        instruction=instruction,
        #lr= True, #kwargs["lr"], # use logistic regression for classification
        clmbrethnicities = True, #kwargs["clmbrethnicities"], # use CLMBR embeddings
        clmbrcodes = kwargs["clmbrcodes"], #kwargs["clmbrcodes"], # use CLMBR embeddings
        calculate_embeddings = kwargs["calculate_embeddings"], # whether to calculate embeddings or load them from file (if already calculated)

        # Feature settings
        random_seed=42,
        num_codes_in_records=50,

        # Data paths
        covariates_path = data_path / "baseline_covariates_231016.feather",
        big_dataset_path = f"{UKB_data_path}dataportal_final_records_omop_240625_mapped_eids_inpatient_updated.feather", #from Inpatient_mapping.py file
        big_dataset_path_clmbr = f"{UKB_data_path}filtered_records_mapped_clmbrwithnames.feather",

        embeddingfile_qwen = embeddingfile_qwen if "embeddingfile_qwen" in locals() else None,
        embeddingfile_qwen3 = embeddingfile_qwen3 if "embeddingfile_qwen3" in locals() else None,
        embeddingfile_llm2vec = embeddingfile_llm2vec if "embeddingfile_llm2vec" in locals() else None,
        embeddingfile_clmbr = embeddingfile_clmbr if "embeddingfile_clmbr" in locals() else None,
        embeddingfile = embeddingfile if "embeddingfile" in locals() else None,
    )

    recruitmentdatename = "recruitment_date" 


    # read in records and covariates data
    # records, selected_covariates, patients_future = load_process_data.load_data(config)
    records, selected_covariates = load_process_data.load_data(config)


    # select only necessary columns from records dataframe
    if(not kwargs["clmbrcodes"]):
        records = pl.from_pandas(records[["eid", "date", "concept_name", "concept_id", "recruitment_date", "code", "vocabulary_id", "vocabulary"]])
    else:
        records = pl.from_pandas(records[["eid", "date", "concept_name", "concept_id", "recruitment_date", "code"]])

    # Continue with the rest of the processing - check that date, recruitment_date and concept_name are not null (all required for embedding creation/ inference)
    records = records.filter(
        pl.col("date").is_not_null()
        & pl.col(recruitmentdatename).is_not_null()
        & pl.col("concept_name").is_not_null()
    )

    #filter out patients with disease prior recruitment - relevant for prediction
    if(not kwargs["calculate_embeddings"]):
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

    if(not kwargs["clmbrcodes"]):
        records = records.filter(pl.col("vocabulary") != "phecode")
    

    if(type(records["date"][0]) == str):
        records = records.with_columns(
            pl.col("date").str.to_datetime("%Y-%m-%d"),  # Adjust format as needed
            pl.col(recruitmentdatename).str.to_datetime("%Y-%m-%d")
        )


    # Ensure correct data types
    filtered_records = records.with_columns(
        [
            pl.col("date").cast(pl.Date),
            pl.col(recruitmentdatename).cast(pl.Date),
            pl.col("concept_name").cast(pl.Utf8),
            pl.col("eid").cast(pl.Utf8),
        ]
    )

    # Calculate days ago if specified in the config
    filtered_records = filtered_records.with_columns(
        [((pl.col(recruitmentdatename) - pl.col("date")).dt.total_days()).alias("days_ago")]
    )




    if(kwargs["calculate_embeddings"]):
        ## Convert to simple list serialization and sort
        df = filtered_records.sort(["eid", "date"], descending=[False, True])

        # Apply uniqueness only if needed
        if not kwargs["keep_all_codes"]:
            subset_col = "code" if kwargs["clmbrcodes"] else "concept_id"
            df = df.unique(subset=["eid", subset_col], keep="first")

        df = df.sort(["eid", "date"], descending=[False, True])  # Ensure sorting after uniqueness if applied

        # Group and aggregate
        prepared_records = (
            df
            .group_by("eid", maintain_order=True)
            .agg(
                pl.col("concept_name").alias("code_list")
            )
        )

        # # Convert selected_covariates to a Polars DataFrame
        selected_covariates_polars = pl.from_pandas(selected_covariates)


        # convert gender to uppercase to keep similar to format in EHRSHOT benchmark
        selected_covariates_polars = selected_covariates_polars.with_columns(
            pl.col("sex")
            .cast(pl.Utf8)
            .str.to_uppercase()
            .cast(pl.Categorical)
        )

        # Ensure the data types match for the join
        prepared_records = prepared_records.with_columns(pl.col("eid").cast(pl.Int32))
        selected_covariates_polars = selected_covariates_polars.with_columns(pl.col("eid").cast(pl.Int32))

        prepared_records_filtered = (
                prepared_records
                # ensure matching dtype for join
                .with_columns(pl.col("eid").cast(pl.Int32))
                .join(
                    selected_covariates_polars.with_columns([
                        pl.col("eid").cast(pl.Int32),
                        pl.col("sex").cast(pl.Utf8),
                        pl.col("ethnicity_name").cast(pl.Utf8),
                    ]),
                    on="eid",
                    how="left",
                )
                .with_columns(
                    pl.col("code_list").list.concat(
                        pl.concat_list([
                            pl.col("sex"),
                            pl.col("ethnicity_name"),
                            pl.lit("Birth"),
                        ]).list.drop_nulls()
                    ).alias("code_list")
                )
                .drop(["sex", "ethnicity_name"])
            )

        ## convert simple list to string serialization for LLM input
        prepared_records = prepared_records_filtered.with_columns(
            pl.col("code_list").list.join("\n").alias("queries")
        )


        prepared_records = prepared_records.to_pandas()


        # Calculate or load embeddings
        if(kwargs["calculate_embeddings"]):
            prepared_records.sort_values(by="eid", inplace=True)
            _ = process_embeddings(prepared_records, config, instruction, **kwargs)
            
            return

        del prepared_records

    

    # from filtered_records, filter out the entries that only occur less than 50 times (need to check if some patients do not have any remaining entries)
    # also before add ontology extension if wanted
    # Count occurrences of each concept code combined with vocabulary_id
    if(not kwargs["clmbrcodes"]):
        filtered_records = filtered_records.with_columns(
            (filtered_records["vocabulary_id"].cast(str) + "/" + filtered_records["code"].cast(str)).alias("codes")
        )
    else:
        filtered_records = filtered_records.with_columns(
            (filtered_records["concept_id"].cast(str)).alias("codes")
        )


    # remove columns in filtered_records that are not required anymore
    filtered_records = filtered_records[["eid", "codes", "days_ago"]]
    filtered_records = filtered_records.to_pandas()
    

    # Add ontology extension for all codes in filtered_records for big_dataset
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




    filtered_records["eid"] = filtered_records["eid"].astype(int)

    selected_covariates['sex'] = selected_covariates['sex'].cat.rename_categories({'Male': 0, 'Female': 1})
    
    selected_covariates["age"] = (
        selected_covariates["age"] - selected_covariates["age"].mean()
    ) / selected_covariates["age"].std()
    

    # # create time bins based on "years_ago" column
    filtered_records["time_bin"] = pd.cut(
        filtered_records["days_ago"],
        bins=TIMEBINS,
        labels=TIMEBIN_LABELS,
        right=False  # 0–30 means [0,30)
    )


    def build_counts_matrix(codes_df, selected_covariates):

        # combined column key
        codes_df["code_time"] = codes_df["codes"].astype(str) + "_" + codes_df["time_bin"].astype(str)

        # categorical encoding
        eid_cat = pd.Categorical(codes_df["eid"])
        col_cat = pd.Categorical(codes_df["code_time"])

        rows = eid_cat.codes
        cols = col_cat.codes

        counts_matrix = sp.coo_matrix(
            (np.ones(len(codes_df), dtype=np.int32), (rows, cols)),
            shape=(len(eid_cat.categories), len(col_cat.categories))
        ).tocsr()

        eid_index = pd.Index(eid_cat.categories)

        # ---- add covariates ----
        cov = (
            selected_covariates
            .set_index("eid")
            .loc[eid_index][["age","sex", "ethnicity"]]
        )

        # in cov - replace nan values of ethnicity column to -1
        cov["ethnicity"] = cov["ethnicity"].fillna(-1)

        # convert ethnicity to categorical and map to integers, with nan values mapped to -1
        cov["ethnicity"] = cov["ethnicity"].map({
            -1: 0,
            2: 1,
            3: 2,
            5: 3
        })
        
        cov_sparse = sp.csr_matrix(cov.values)

        counts_matrix = sp.hstack([counts_matrix, cov_sparse]).tocsr()

        return counts_matrix, eid_index
    
    counts_matrix, eid_index = build_counts_matrix(filtered_records, selected_covariates)
    
    del filtered_records
    del selected_covariates
    gc.collect()

    main_evaluation(config['disease'], config['selected_phecode'], clmbrcodes=kwargs["clmbrcodes"], counts_df=counts_matrix, counts_eids=eid_index)

    return





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Your script description")
    parser.add_argument("--model", type=str, help="Model - choose from LLM2Vec, Qwen(2), Qwen3 or BioClinicalBERT. For inference no model needs to be specified (but --infer_all).")
    parser.add_argument("--infer_all", type=lambda x: x.lower() == "true", nargs='?', const=True, default=False, help="Predict performance for all models at once (currently: Counts, Qwen, Qwen3, LLM2Vec).")
    parser.add_argument("--indication", type=str, help="Diseasename (if task = disease_onset), otherwise choose from [death, hospitalization]")
    parser.add_argument("--phecode", type=str, help="Phecode (if task = disease_onset)")
    parser.add_argument("--minyears", type=float, help="Time in years from recruitment date to consider patients as diseased. Also float can be provided (will be calculated in months).", default=1.0)
    parser.add_argument("--maxyears", type=int, help="Time in years to consider patients diseased in the future for logistic regression", default=10)
    parser.add_argument("--tokenlength", type=int, help="Number of tokens used for embedding creation", default=4096)
    parser.add_argument("--calculate_embeddings", action="store_true", help="calculate embeddings - e.g. to include disease name in query")
    parser.add_argument("--clmbrcodes", type=lambda x: x.lower() == "true", nargs='?', const=True, default=False, help="Only use codes also used in clmbr.")
    parser.add_argument("--disable_wandb", action="store_true", help="Set to true if wandb should be disabled - e.g. for debugging.")
    parser.add_argument("--batch_size", type=int, help="Batchsize to calculate embeddings", default=10)

    parser.add_argument("--keep_all_codes", action="store_true", help="Use all codes instead of unqiue codes in correct order for patient serialization")

    
    args = parser.parse_args()

    wandbname = "Run_" + args.model + "_" + args.indication
    wandb.login()

    if(args.disable_wandb):
        #wandb.init(project="LLM2Vec", entity="georg-von-arnim-berlin-institute-of-health", mode="disabled")
        wandb.init(project="LLM2Vec", entity="cardiors", mode="disabled")
    else:
        wandb.init(project="LLM2Vec", entity="cardiors", name=wandbname)
        #wandb.init(project="LLM2Vec", entity="georg-von-arnim-berlin-institute-of-health", mode="disabled")



    # Log hyperparameters
    wandb.config.update(vars(args))

    instruction_prefix = "Given a patient's electronic healthcare record (EHR) as a newline separated list, retrieve relevant passages that answer the query:"


    if(args.indication == "death"):
        instruction_task = f"will the patient die within {'one' if args.maxyears == 1 else f'{args.maxyears}'} year"
        task = "death"
    elif(args.indication == "hospitalization"):
        instruction_task = f"will the patient be admitted to the hospital within {'one' if args.maxyears == 1 else f'{args.maxyears}'} year"
        task = "hospitalization"
    else:    
        instruction_task = f"has the patient {args.indication.lower()}"
        task = args.phecode
    
    print(args.indication)


    instruction = instruction_prefix + " " + instruction_task

    main(instruction, task, **vars(args))

    wandb.finish()