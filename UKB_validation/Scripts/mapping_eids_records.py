import pathlib
import pandas as pd
import polars as pl


data_path = pathlib.Path(
    #"/sc-projects/sc-proj-ukb-cvd/data/3_datasets_post/231012_ukb_preprocessing/ukb_data_portal/2_final"
    "/sc-projects/sc-proj-ukb-cvd/data/3_datasets_post/240625_ukb_preprocessing/ukb_data_portal/2_final/"
)

records_path = data_path / "dataportal_final_records_omop_240625.feather"
#endpoints_path = data_path / "dataportal_final_endpoints_240625.feather"
#phecodes_path = data_path / "dataportal_final_phecodes_240625.feather"
covariates_path = data_path / "baseline_covariates_240625.feather"

records = pd.read_feather(records_path)
records


eid_mapping = pl.read_csv('/sc-resources/ukb/data/shared/open/fam_files/link_file.tsv', separator='\t', columns=['EID.44448', 'EID.49966'])
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

outfile = "/sc-projects/sc-proj-ukb-cvd/projects/llm2vec/data/dataportal_final_records_omop_240625_mapped_eids.feather" #this file is used by clmbr and embedding creation code
records.to_pandas().to_feather(outfile)
