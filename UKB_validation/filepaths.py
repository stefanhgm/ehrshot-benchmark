### creation of records file
## mapping_eids_records.py
# raw UKB data
UKB_RAW_DATA_PATH = "/sc-projects/sc-proj-ukb-cvd/data/3_datasets_post/240625_ukb_preprocessing/ukb_data_portal/2_final/"

# UKB raw data records
UKB_RAW_DATA_RECORDS = UKB_RAW_DATA_PATH +"dataportal_final_records_omop_240625.feather"

# UKB raw data covariates
UKB_RAW_DATA_COVARIATES = UKB_RAW_DATA_PATH + "baseline_covariates_240625.feather"

# mapping tsv file containing eid mappings between data versions - might not be required
UKB_EID_MAPPING = "/sc-resources/ukb/data/shared/open/fam_files/link_file.tsv"

# path containing most data
UKB_DATA_PATH = "/sc-projects/sc-proj-ukb-cvd/projects/llm2vec/data/"

# Records file with mapped patient ids
UKB_RECORDS_FILE_MAPPED_EIDS = UKB_DATA_PATH + "dataportal_final_records_omop_240625_mapped_eids.feather"

## map_admission_ids.py
#
UKB_RAW_ADMISSION_FILE = "/sc-projects/sc-proj-ukb-cvd/data/3_datasets_post/240625_ukb_preprocessing/ukb_data_portal/1_processed/codes_admin_231012.feather"

# File created
ADMISSION_RECRUITMENT_FILE = UKB_DATA_PATH + "codes_admin_231012_mapped_eids_filtered.feather"

## Inpatient_mapping.py
# path to records file used in main LLM2Vec.py script - created with
records_path_big = UKB_DATA_PATH + "dataportal_final_records_omop_240625_mapped_eids_inpatient_updated.feather"



### clmbr_baseline
## clmbr_baseline_create_embeddings.py
# folder containing hf models
shared_cache_dir = "/sc-projects/sc-proj-dh-ag-eils-ml/shared_hf_cache/hub"
# folder containing all UKB data files
UKB_covariates_folder = "/sc-projects/sc-proj-ukb-cvd/data/3_datasets_post/231012_ukb_preprocessing/ukb_data_portal/2_final"

# file containing covariates including age, gender and birth date for all patients in UKB
UKB_covariates_filename = "baseline_covariates_231016.feather"


# path to clmbr mapping file with all available clmbr codes mapped through ATHENA
# created with clmbr_baseline.py script
clmbr_UKB_mapping_complete_path = UKB_DATA_PATH +"filtered_records_mapped_clmbrwithnames.feather"

# file to save ukb patients in MEDS format in
clmbr_meds_pickle = UKB_DATA_PATH + "clmbr_ukb_meds.pkl"

# output path of feather file containing clmbr embeddings
EMBEDDING_PATH = UKB_DATA_PATH + "embeddings/"

## clmbr_baseline
# path to ATHENA dataset
athena_dataset_path = "/sc-projects/sc-proj-ukb-cvd/data/mapping/athena_250220"


### evaluation.py
PATH_TO_SPLIT_CSV = UKB_DATA_PATH + "splits.json"
PATH_TO_LABELED_PATIENTS_FOLDER=f"~/Documents/LLM2Vec_project/Splits/generate_labels_2/"
# base path - will contain all results for all diseases and all methods for UKB
BASE_PATH_RESULTS = "/sc-projects/sc-proj-ukb-cvd/projects/llm2vec/"
# PATH to csv files containing task group results for ehrshot - for plotting
BASE_PATH_FIGS_EHRSHOT = "/home/sthe14/ehrshot-benchmark/EHRSHOT_ASSETS/figures/task_groups/"

## LLM2Vec.py
# csv file containing code to parent mapping for all codes in the dataset - used for ontology extension
ontology_extension = "Data_preprocessing/mappings/code_to_parent_mapping.csv"
