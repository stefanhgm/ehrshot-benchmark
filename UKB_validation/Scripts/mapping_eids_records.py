import pathlib
import pandas as pd
import polars as pl


import sys
sys.path.append("../../UKB_validation/")
#import python file containing filenames
import filepaths as fp



records_path = fp.UKB_RAW_DATA_RECORDS
covariates_path = fp.UKB_RAW_DATA_COVARIATES

records = pd.read_feather(records_path)
records


eid_mapping = pl.read_csv(fp.UKB_EID_MAPPING, separator='\t', columns=['EID.44448', 'EID.49966'])
eid_mapping

records = pl.from_pandas(records[["eid", "date", "concept_name", "concept_id", "recruitment_date", "domain_id", "code", "vocabulary_id", "vocabulary", "origin"]])
records

# append eid_mapping to records based on EID.44448 and eid columns
records = records.join(
    eid_mapping, 
    left_on='eid', 
    right_on='EID.44448', 
    how='inner'
)
records.head()


# remove col 'EID.44448' and rename 'EID.49966' to 'eid'
records = records.drop('eid')
records = records.rename({'EID.49966': 'eid'})
records = records.filter(records["eid"].is_not_null())
records.head(), records.shape


print(records.shape)
#drop duplicates in polars df

records = records.unique() 
print(records.shape)
print(records.head())

outfile = fp.UKB_RECORDS_FILE_MAPPED_EIDS
records.to_pandas().to_feather(outfile)
