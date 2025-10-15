## Script for mapping UKB codes to CLMBR codes using intermediate mapping
#Steps:
#1. map CLMBR codes to omop codes using ATHENA mapping
#2. map concept codes from clmbr to concept_id's from UKB codes

import pandas as pd
from collections import Counter
import pathlib
import seaborn as sns
import matplotlib.pyplot as plt
import polars as pl
import numpy as np




# Read in available clmbr codes
clmbr_mapping = pd.read_csv("code_lookup.csv", index_col=0)

# create two new columns called code and source, containing the split values of the code_string column
clmbr_mapping[['source', 'code']] = clmbr_mapping['code_string'].str.split('/', n=1, expand=True)
clmbr_mapping = clmbr_mapping[['source', 'code']]

clmbr_mapping["source"] = clmbr_mapping["source"].astype(str)
clmbr_mapping["code"] = clmbr_mapping["code"].astype(str)

clmbr_mapping_unique = clmbr_mapping.drop_duplicates(subset=["source", "code"], keep=False)

# Read in ATHENA dataset
path = "/sc-projects/sc-proj-ukb-cvd/data/mapping/athena_250220" #change path to your athena folder
vocab = {
    "concept": pd.read_csv(f"{path}/CONCEPT.csv", sep="\t"),
    "concept_cpt4": pd.read_csv(f"{path}/CONCEPT_CPT4.csv", sep="\t"),
    "domain": pd.read_csv(f"{path}/DOMAIN.csv", sep="\t"),
    "class": pd.read_csv(f"{path}/CONCEPT_CLASS.csv", sep="\t"),
    "relationship": pd.read_csv(f"{path}/RELATIONSHIP.csv", sep="\t"),
    "drug_strength": pd.read_csv(f"{path}/DRUG_STRENGTH.csv", sep="\t"),
    "vocabulary": pd.read_csv(f"{path}/VOCABULARY.csv", sep="\t"),
    "concept_synonym": pd.read_csv(f"{path}/CONCEPT_SYNONYM.csv", sep="\t"),
    "concept_ancestor": pd.read_csv(f"{path}/CONCEPT_ANCESTOR.csv", sep="\t"),
    "concept_relationship": pd.read_csv(f"{path}/CONCEPT_RELATIONSHIP.csv", sep="\t"),
}

# save concept and concept_cpt4 in separate dataframes and combine them
concept_df = vocab["concept"]
concept_df_cpt4 = vocab["concept_cpt4"]

concept_df = pd.concat([concept_df, concept_df_cpt4], axis=0)

concept_df["concept_code"] = concept_df["concept_code"].astype(str)


def prepare_ancestors(vocab_ancestor, kg_nodes_list):
    kg_anc_raw = vocab_ancestor.query(
        f"ancestor_concept_id == @kg_nodes_list & descendant_concept_id == @kg_nodes_list"
    ).query("min_levels_of_separation!=0")
    kg_anc_raw = pd.concat(
        [
            kg_anc_raw.assign(relationship_id="Subsumes").rename(
                columns={
                    "ancestor_concept_id": "concept_id_1",
                    "descendant_concept_id": "concept_id_2",
                }
            ),
            kg_anc_raw[
                [
                    "descendant_concept_id",
                    "ancestor_concept_id",
                    "min_levels_of_separation",
                    "max_levels_of_separation",
                ]
            ]
            .assign(relationship_id="Is a")
            .rename(
                columns={
                    "descendant_concept_id": "concept_id_1",
                    "ancestor_concept_id": "concept_id_2",
                }
            ),
        ]
    )[
        [
            "concept_id_1",
            "concept_id_2",
            "relationship_id",
            "min_levels_of_separation",
            "max_levels_of_separation",
        ]
    ]
    return kg_anc_raw


# initial settings
vocabularies_to = ["SNOMED", "RxNorm", "ATC", "CVX"]
domains = ["Condition", "Observation", "Procedure", "Drug", "Device"]


kg_nodes = vocab["concept"]  # .query("domain_id==@domains")
kg_nodes_list = kg_nodes.concept_id.to_list()
kg_edges_raw = (
    vocab["concept_relationship"]
    .query("concept_id_1 == @kg_nodes_list & concept_id_2 == @kg_nodes_list")
    .query("concept_id_1!=concept_id_2")[["concept_id_1", "concept_id_2", "relationship_id"]]
    .assign(k_hops=1)
)

# shortcuts
kg_anc_raw = prepare_ancestors(vocab["concept_ancestor"], kg_nodes_list).assign(
    k_hops=lambda x: x.min_levels_of_separation
)[["concept_id_1", "concept_id_2", "relationship_id", "k_hops"]]
kg_edges_raw = pd.concat([kg_edges_raw, kg_anc_raw])
kg_edges_raw = kg_edges_raw.drop_duplicates().reset_index(drop=False)



cols = [
    "concept_id",
    "concept_name",
    "concept_code",
    "domain_id",
    "vocabulary_id",
    "concept_class_id",
    "standard_concept",
]
merge_1 = kg_nodes[cols].rename(columns={col: f"{col}_1" for col in cols})
merge_2 = kg_nodes[cols].rename(columns={col: f"{col}_2" for col in cols})
kg_edges = (
    kg_edges_raw[["concept_id_1", "concept_id_2", "relationship_id", "k_hops"]]
    .merge(merge_1, on="concept_id_1", how="left")
    .merge(merge_2, on="concept_id_2", how="left")
)


valid_mapping_relations = [
    #"Is a",
    #"Subsumes",
    "Maps to",
    # "RxNorm has ing", #evtl ab hier rausnehmen - testen wie viele
    # "Brand name of",
    # "RxNorm - CVX",
    # "SNOMED - RxNorm eq",
    # "RxNorm - SNOMED eq",
    # "CVX - RxNorm",
]
kg_edges_mapping = kg_edges[kg_edges["relationship_id"].isin(valid_mapping_relations)] #[['concept_id_1', 'concept_id_2']]
#kg_edges_mapping = kg_edges_mapping[["concept_id_2", "concept_code_1", "domain_id_1", "vocabulary_id_1", "vocabulary_id_2"]]

#filter out columns not required
kg_edges_mapping = kg_edges_mapping[["concept_code_1", "concept_id_2", "relationship_id", "k_hops", "vocabulary_id_1", "vocabulary_id_2"]]

kg_edges_mapping["concept_code_1"] = kg_edges_mapping["concept_code_1"].astype(str)


# map clmbr codes to omop codes of ATHENA mapping
clmbr_athena_mapped = kg_edges_mapping.merge(
    clmbr_mapping_unique,
    left_on='concept_code_1',
    right_on='code',
    how='inner'
)

## Map clmbr codes to concept_df on concept_code
clmbr_athena_mapped_direct = clmbr_mapping_unique.merge(
    concept_df,
    left_on='code',
    right_on='concept_code',
    how='inner'
)


## now read in UKB data and map to clmbr codes 1. without vocabulary, 2. with second vocabulary
#records_path_big = "/sc-projects/sc-proj-ukb-cvd/projects/llm2vec/data/dataportal_final_records_omop_240625_mapped_eids.feather"
records_path_big = "/sc-projects/sc-proj-ukb-cvd/projects/llm2vec/data/dataportal_final_records_omop_240625_mapped_eids_inpatient_updated.feather"
records = pd.read_feather(records_path_big)

# remove all entries after recruitment date
filtered_records = records[records['date'] <= records['recruitment_date']]

## remove all phecode entries - only keep OMOP codes (no other codes like mapped phecodes)
filtered_records = filtered_records[filtered_records["vocabulary"] != "phecode"]
filtered_records = filtered_records[["date", "eid", "concept_id", "vocabulary", "vocabulary_id", "concept_name", "recruitment_date", "domain_id"]]

#filtered_records_nohospital_unique = filtered_records.drop_duplicates(ignore_index=True, subset=["code", "concept_id"])

# only keep omop code (structure before: OMOP_xxx)
filtered_records['code'] = filtered_records['concept_id'].str.split('_').str[1]

clmbr_athena_mapped_direct["concept_id"] = clmbr_athena_mapped_direct["concept_id"].astype(str)
clmbr_athena_mapped["concept_id_2"] = clmbr_athena_mapped["concept_id_2"].astype(str)
filtered_records["code"] = filtered_records["code"].astype(str)

# map UKB to direct codes 
clmbr_UKB_mapping_direct = clmbr_athena_mapped_direct.merge(
    filtered_records,
    left_on='concept_id',
    right_on='code',
    how='right'
)

# for mapping to relationship only use remaining patients that were not mappable before
filtered_records_rest = clmbr_UKB_mapping_direct[clmbr_UKB_mapping_direct['concept_id_x'].isna()]
filtered_records_rest = filtered_records_rest[['date', 'eid', 'concept_id_y', 'vocabulary', 'vocabulary_id_y', 'concept_name_y', 'recruitment_date', 'domain_id_y', 'code_y']].rename(columns={
    'concept_id_y': 'concept_id',
    'vocabulary_id_y': 'vocabulary_id',
    'concept_name_y': 'concept_name',
    'domain_id_y': 'domain_id',
    'code_y': 'code'
})

clmbr_UKB_mapping_direct = clmbr_UKB_mapping_direct[~clmbr_UKB_mapping_direct['code_x'].isna()]


# check for relations between UKB and clmbr codes through vocabulary
clmbr_UKB_mapping = clmbr_athena_mapped.merge(
    filtered_records_rest,
    left_on='concept_id_2',
    right_on='code',
    how='inner'
)

print(filtered_records["code"].nunique(), clmbr_UKB_mapping["code_y"].nunique(), clmbr_mapping_unique["code"].nunique(), clmbr_UKB_mapping["code_x"].nunique())

print(clmbr_UKB_mapping.columns)
clmbr_UKB_mapping = clmbr_UKB_mapping[["eid", "date", "code_x", "source", "concept_name", "concept_id", "recruitment_date", "domain_id"]]
clmbr_UKB_mapping.rename(columns={"code_x": "code"}, inplace=True)

print(clmbr_UKB_mapping_direct.columns)
clmbr_UKB_mapping_direct = clmbr_UKB_mapping_direct[["eid", "date", "code_x", "source", "concept_name_y", "concept_id_y", "recruitment_date", "domain_id_y"]]
clmbr_UKB_mapping_direct.rename(columns={"code_x": "code", "concept_name_y": "concept_name", "domain_id_y": "domain_id"}, inplace=True)


# Also add Inpatient visits and ICU visits bsed on EHRShot/ CLMBR format
# Inpatient visits are mapped to OMOP_9201 
filtered_records_inpatient = filtered_records[filtered_records["concept_id"] == "OMOP_9201"] 
filtered_records_inpatient_clmbr = filtered_records_inpatient[["eid", "date", "recruitment_date", "concept_id", "domain_id"]]
filtered_records_inpatient_clmbr["code"] = "IP"
filtered_records_inpatient_clmbr["source"] = "Visit"
filtered_records_inpatient_clmbr["concept_name"] = "IP Visit"

# ICU visits are mapped to OMOP_4138933
filtered_records_icu = filtered_records[filtered_records["concept_id"] == "OMOP_4138933"]
filtered_records_icu_clmbr = filtered_records_icu[["eid", "date", "recruitment_date", "concept_id", "domain_id"]]
filtered_records_icu_clmbr["code"] = "ER"
filtered_records_icu_clmbr["source"] = "Visit"
filtered_records_icu_clmbr["concept_name"] = "ER Visit"

visit_df = pd.concat([filtered_records_inpatient_clmbr, filtered_records_icu_clmbr], axis=0)


# combine the two df's
clmbr_UKB_mapping_complete = pd.concat([clmbr_UKB_mapping, clmbr_UKB_mapping_direct, visit_df])

# check that all codes are in clmbr_mapping_df
clmbr_mapping_df = pd.read_csv("code_lookup.csv", index_col=0)

clmbr_UKB_mapping_complete["source"] = clmbr_UKB_mapping_complete["source"].astype(str)
clmbr_UKB_mapping_complete["code"] = clmbr_UKB_mapping_complete["code"].astype(str)

clmbr_UKB_mapping_complete["clmbr_code"] = clmbr_UKB_mapping_complete["source"] + "/" + clmbr_UKB_mapping_complete["code"]
## check if all codes are in the code lookup table
clmbr_UKB_mapping_complete = clmbr_UKB_mapping_complete[clmbr_UKB_mapping_complete["clmbr_code"].isin(clmbr_mapping_df["code_string"])]


clmbr_UKB_mapping_complete = clmbr_UKB_mapping_complete[["eid", "date", "code", "source", "concept_name", "concept_id", "recruitment_date", "domain_id"]]
clmbr_UKB_mapping_complete.to_feather("/sc-projects/sc-proj-ukb-cvd/projects/llm2vec/data/filtered_records_mapped_clmbrwithnames.feather")

# ## for the LLMs, adapt the df a bit and save separately

# clmbr_UKB_mapping_complete = clmbr_UKB_mapping_complete[["eid", "date", "concept_name", "concept_id", "recruitment_date", "domain_id"]]
# clmbr_UKB_mapping_complete.to_feather("/sc-projects/sc-proj-ukb-cvd/projects/llm2vec/data/clmbr_ukb_mapping.feather")





# #Just for the overview
# clmbr_UKB_mapping_complete["ehr_class"] = clmbr_UKB_mapping_complete["domain_id"].apply(lambda x:  "Medications" if x in ["Drug"] else "Procedures" if x in ["Procedure", "Measurement"] else "Diagnoses")


# clmbr_UKB_mapping_complete["date"] = clmbr_UKB_mapping_complete["date"].astype(str)
# clmbr_UKB_mapping_complete["recruitment_date"] = clmbr_UKB_mapping_complete["recruitment_date"].astype(str)
# clmbr_UKB_mapping_complete["date"] = pd.to_datetime(clmbr_UKB_mapping_complete["date"])
# clmbr_UKB_mapping_complete["recruitment_date"] = pd.to_datetime(clmbr_UKB_mapping_complete["recruitment_date"])
# # filter out records appearing after recruitment_date
# records_df = clmbr_UKB_mapping_complete[clmbr_UKB_mapping_complete["date"] <= clmbr_UKB_mapping_complete["recruitment_date"]]

# print(clmbr_UKB_mapping_complete["ehr_class"].value_counts())