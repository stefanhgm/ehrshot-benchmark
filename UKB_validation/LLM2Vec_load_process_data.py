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
    if(config['clmbrcodes']):
        records_path = config['big_dataset_path_clmbr']
    else:
        records_path = config['big_dataset_path']
    
    # Read data with appropriate transformations
    records_df = pd.read_feather(records_path) 
    
    # filter out records appearing after recruitment_date
    records_df = records_df[records_df["date"] <= records_df["recruitment_date"]]

    records_df["date"] = records_df["date"].dt.strftime("%Y-%m-%d")
    records_df["recruitment_date"] = records_df["recruitment_date"].dt.strftime("%Y-%m-%d")

    # Filter records to match demographic data
    records_df = records_df[records_df["eid"].isin(demographics_df["eid"])]
    
    return records_df, demographics_df 