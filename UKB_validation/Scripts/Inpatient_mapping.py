import pandas as pd
import numpy as np
import time


import sys
sys.path.append("../../UKB_validation/")
#import python file containing filenames
import filepaths as fp


#Check that no date information about admissions is going over the time of the admission
## Combine medical codes from patient stays into one date - makes serialization easier
def update_dates_efficiently(admission_df, big_df):
    """
    Efficiently update dates in big_df that fall within windows defined in admission_df.
    
    Parameters:
    -----------
    admission_df : DataFrame
        DataFrame containing columns 'eid', 'date', and 'date_end' defining windows
    big_df : DataFrame
        DataFrame containing 'eid' and 'date' to be checked/updated
    
    Returns:
    --------
    DataFrame
        A copy of big_df with dates updated where applicable
    """
    # Create a copy of big_df to avoid modifying the original
    #result_df = big_df.copy()
    
    # Ensure all date columns are datetime objects
    #admission_df = admission_df.copy()
    admission_df['date'] = pd.to_datetime(admission_df['date'])
    admission_df['date_end'] = pd.to_datetime(admission_df['date_end'])
    big_df['date'] = pd.to_datetime(big_df['date'])
    
    # Create a unique identifier for each row in the result_df
    big_df['original_index'] = big_df.index
    
    # Rename columns in admission_df to avoid conflicts in the merge
    admission_df = admission_df.rename(columns={
        'date': 'admission_date',
        'date_end': 'admission_date_end'
    })
    
    # Perform a cross join between the two dataframes, keeping only rows with matching eid
    merged_df = pd.merge(
        big_df,
        admission_df,
        on='eid',
        how='left'
    )
    
    # Create a mask for rows where the date falls within the window
    in_window_mask = (
        (merged_df['date'] >= merged_df['admission_date']) & 
        (merged_df['date'] <= merged_df['admission_date_end'])
    )
    
    # Filter to only rows where the date is within the window
    matches = merged_df[in_window_mask]
    
    # If there are no matches, return the original dataframe
    if matches.empty:
        return big_df
    
    # Group by the original index and take the first admission date for each
    # This handles cases where a date might fall within multiple admission windows
    updates = matches.groupby('original_index')['admission_date'].first()
    
    # Apply the updates to the result dataframe
    big_df.loc[updates.index, 'date'] = updates.values
    
    # Remove the temporary column
    big_df.drop('original_index', axis=1, inplace=True)
    
    return big_df


# Read in data
admission_recruitment = pd.read_feather(fp.ADMISSION_RECRUITMENT_FILE) #created with script map_admission_ids

big_dataset_path = fp.UKB_RECORDS_FILE_MAPPED_EIDS #created with script mapping_eids_records
big_dataset = pd.read_feather(big_dataset_path)

big_dataset["date"] = pd.to_datetime(big_dataset["date"])
big_dataset["date"] = big_dataset["date"].dt.date



updated_df = update_dates_efficiently(admission_recruitment, big_dataset)

#Save updated df to feather
updated_df.to_feather(fp.records_path_big)
