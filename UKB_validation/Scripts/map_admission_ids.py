import pandas as pd

import sys
sys.path.append("../../UKB_validation/")
#import python file containing filenames
import filepaths as fp


# Read in eid mapping file
eid_mapping = pd.read_csv(fp.UKB_EID_MAPPING, sep='\t', usecols=['EID.49966', 'EID.44448'])
eid_mapping

# Read in df containing length of stays
admin_df = pd.read_feather(fp.UKB_RAW_ADMISSION_FILE)
admin_df

# Map eids of admin_df to the new eids
admin_df = admin_df.merge(eid_mapping, left_on='eid', right_on='EID.44448', how='inner')
admin_df = admin_df.drop(columns=['eid', 'EID.44448'])
admin_df = admin_df.rename(columns={'EID.49966': 'eid'})
admin_df = admin_df[['eid', 'date', 'date_end', 'duration_days']]
admin_df

# Read in data
big_dataset_path = fp.UKB_RECORDS_FILE_MAPPED_EIDS
big_dataset = pd.read_feather(big_dataset_path)

## Check that between no start and end date a recruitment date is present
#-> so no information leakage is present

big_dataset["date"] = pd.to_datetime(big_dataset["date"])
big_dataset["date"] = big_dataset["date"].dt.date

big_dataset_recruitmentdate = big_dataset[['eid', 'date', 'recruitment_date']]
big_dataset_recruitmentdate.drop_duplicates(inplace=True)
big_dataset_recruitmentdate

# Merge the two dataframes based on matching eid and date (end in test_df and date in big_df_subset)
merged_df = pd.merge(admin_df, big_dataset_recruitmentdate, left_on=['date', 'eid'], right_on=['date', 'eid'], how='left')
merged_df

# filter out entries where date is smaller than value in column recruitment_date
admission_recruitment = merged_df[merged_df["date_end"] <= merged_df["recruitment_date"]]
print(admission_recruitment.shape)
admission_recruitment.head()

# check for NaN values in duration_days
admission_recruitment = admission_recruitment[~admission_recruitment["duration_days"].isna()]

admission_recruitment[['eid', 'date', 'date_end', 'duration_days']].to_feather(fp.ADMISSION_RECRUITMENT_FILE)