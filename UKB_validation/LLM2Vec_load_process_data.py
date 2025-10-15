import pandas as pd
import numpy as np
import pathlib


def load_data(config):
    """
    Load data from specified paths.
    
    Args:
        config (dict): Configuration with data paths and options
    
    Returns:
        tuple: (records_df, demographics_df)
    """

    # Load demographic data
    demographics_list = ['eid', 'age_at_recruitment_f21022_0_0', 'sex_f31_0_0', 'ethnic_background_f21000_0_0']
    demographics_df = pd.read_feather(config['covariates_path'], columns=demographics_list)
    demographics_df = demographics_df.rename(columns={
        'age_at_recruitment_f21022_0_0': 'age', 
        'sex_f31_0_0': 'sex',
        'ethnic_background_f21000_0_0': 'ethnicity'
    })

    if config['clmbrethnicities']:
        # Convert ethnicity based on values from clmbr if wanted
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
        clmbr_thnicity_to_name = {
            1: "American Indian",
            2: "Asian",
            3: "Black",
            4: "Pacific Islander",
            5: "White"
        }
        demographics_df["ethnicity"] = demographics_df["ethnicity"].map(data_ethnicity_to_clmbr)
        demographics_df["ethnicity_name"] = demographics_df["ethnicity"].map(clmbr_thnicity_to_name)
    else:
        data_ethnicity_to_name = {
            "-3":  "Prefer not to answer",
            "-1":  "Do not know",
            "1":  "White",
            "2":  "Mixed", #-> No direct equivalent
            "3": "Asian or Asian British", # -> Asian
            "4": "Black or Black British", # -> Black or African American
            "5": "Chinese", # -> Considered part of Asian
            "6": "Other ethnic group", # -> No direct match
            "1001": "British", # -> White
            "1002": "Irish", # -> White
            "1003": "Any other white background", # -> White
            "2001": "White and Black Caribbean", # -> Mixed (No direct equivalent)
            "2002": "White and Black African", # -> Mixed (No direct equivalent)
            "2003": "White and Asian", # -> Mixed (No direct equivalent)
            "2004": "Any other mixed background", # -> Mixed (No direct equivalent)
            "3001": "Indian", # -> Asian
            "3002": "Pakistani", # -> Asian
            "3003": "Bangladeshi", # -> Asian
            "3004": "Any other Asian background", # -> Asian
            "4001": "Caribbean", # -> Black or African American
            "4002": "African", # -> Black or African American
            "4003": "Any other Black background", # -> Black or African American
        }
        demographics_df["ethnicity_name"] = demographics_df["ethnicity"].map(data_ethnicity_to_name)
  
    
    # Load records data based on configuration
    if config['use_big_dataset']:
        if(config['clmbrcodes']):
            records_path = config['big_dataset_path_clmbr']
        else:
            records_path = config['big_dataset_path']
        records_path_indications = config['standard_dataset_path'] #to get indications from future
    elif config['use_raw_dataset']:
        records_path = config['raw_dataset_path']
        records_path_indications = config['standard_dataset_path'] #to get indications from future
    else:
        records_path = config['standard_dataset_path']
    
    # Read data with appropriate transformations
    if not config['use_raw_dataset']:
        records_df = pd.read_feather(records_path) #.drop(columns=["Unnamed: 0"]).reset_index(drop=True)
        
        ## TODO: remove - just to speed up embedding calculation
        #df = pd.read_feather(speed_filepath)
        #records_df = records_df[~records_df["eid"].isin(df["eid"])]
        
        records_df["date"] = records_df["date"].astype(str)
        records_df["recruitment_date"] = records_df["recruitment_date"].astype(str)
        records_df["date"] = pd.to_datetime(records_df["date"])
        records_df["recruitment_date"] = pd.to_datetime(records_df["recruitment_date"])


        ## add endpoints - from before and after
        records_indications = records_df.copy() #[records_df["date"] > records_df["recruitment_date"]]
        #records_indications["date"] = records_indications["date"].astype(str)
        #records_indications["recruitment_date"] = records_indications["recruitment_date"].astype(str)
        #records_indications["ehr_class"] = np.nan

        # filter out records appearing after recruitment_date
        records_df = records_df[records_df["date"] <= records_df["recruitment_date"]]

        records_df["date"] = records_df["date"].astype(str)
        records_df["recruitment_date"] = records_df["recruitment_date"].astype(str)
    else:
        records_df = pd.read_feather(records_path) #already checked that only information from before recruitment date is present
        records_df["date"] = records_df["date"].astype(str)
        records_df["recruitment_date"] = records_df["recruitment_date"].astype(str)

    # Filter out entries that have phecode as vocabulary
    if(not config["clmbrcodes"] and not config["use_raw_dataset"]):
        records_df = records_df[records_df.vocabulary != "phecode"]

    if(config["use_big_dataset"]):
        # add new column for different EHR classes - Conditions, Medications, Procedures
        # first, similar to ehrshot approach, change CVX codes to Procedures
        records_df.loc[records_df['vocabulary_id'] == 'CVX', 'domain_id'] = 'Observation'
        records_df["ehr_class"] = records_df["domain_id"].apply(lambda x:  "Medications" if x in ["Drug"] else "Procedures" if x in ["Procedure", "Measurement"] else "Conditions")
        #records_df["ehr_class"].value_counts()

        # records_indications = pd.read_feather(records_path_indications, columns=["eid", "date", "concept_name", "concept_id", "recruitment_date"])
        # # keep information about patients that have disease before recruitment date


        patients_future = records_indications[records_indications["date"] >= records_indications["recruitment_date"]]["eid"].unique() #patients with information after recruitment date
        #records_indications = patients_future[patients_future["concept_id"] == config["selected_phecode"]]
        records_indications = records_indications[records_indications["concept_id"] == config["selected_phecode"]]
        records_indications["date"] = records_indications["date"].astype(str)
        records_indications["recruitment_date"] = records_indications["recruitment_date"].astype(str)
        records_indications["ehr_class"] = np.nan
        records_df = pd.concat([records_df, records_indications], axis=0)
    else:
        if(config["use_raw_dataset"]):
            records_indications = pd.read_feather(records_path_indications, columns=["eid", "date", "concept_name", "concept_id", "recruitment_date", "code", "vocabulary_id"])
            patients_future = records_indications[records_indications["date"] >= records_indications["recruitment_date"]]["eid"].unique() #patients with information after recruitment date
            #records_indications = records_indications[(records_indications["date"] >= records_indications["recruitment_date"]) & (records_indications["concept_id"] == config["selected_phecode"])]
            records_indications = records_indications[records_indications["concept_id"] == config["selected_phecode"]] #also keep information about patients that have disease before recruitment date
            records_indications["date"] = records_indications["date"].astype(str)
            records_indications["recruitment_date"] = records_indications["recruitment_date"].astype(str)
            records_indications["ehr_class"] = np.nan
            records_df = pd.concat([records_df, records_indications], axis=0)
        else:
            patients_future = []

    length_of_stay_df = pd.read_feather(config['length_of_stay_path'])
    length_of_stay_df["date"] = pd.to_datetime(length_of_stay_df["date"])
    length_of_stay_df["date"] = length_of_stay_df["date"].dt.date

    # Filter records to match demographic data
    records_df = records_df[records_df["eid"].isin(demographics_df["eid"])]
    
    return records_df, demographics_df, length_of_stay_df, patients_future