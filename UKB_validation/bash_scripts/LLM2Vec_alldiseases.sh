#!/bin/bash


CODE_DIR=~/Documents/ehrshot-benchmark/UKB_validation/bash_scripts

cd $CODE_DIR

source ~/.bashrc

conda activate ~/.conda/envs/LLM

# List of diseases and corresponding phecodes
diseases=(
    "Endometriosis,phecode_GU_615"
    "Hypertension,phecode_CV_401"    
    "Diabetes mellitus,phecode_EM_202"  #might have to calculate tesplits!
    "Pneumonia,phecode_RE_468"          #might have to calculate tesplits!
    "Chronic obstructive pulmonary disease [COPD],phecode_RE_474"
    "Chronic kidney disease,phecode_GU_582.2"
    "Ischemic heart disease,phecode_CV_404"
    "Myocardial infarction [Heart attack],phecode_CV_404.1"
    "Cerebral infarction [Ischemic stroke],phecode_CV_431.11"
    "Heart failure,phecode_CV_424"
    "Cardiac arrest,phecode_CV_420"
    "Abdominal aortic aneurysm,phecode_CV_438.11"
    "Pulmonary embolism,phecode_CV_440.3"
    "Aortic stenosis,phecode_CV_413.21"
    "Mitral valve insufficiency,phecode_CV_413.11"
    "Endocarditis,phecode_CV_410.2"
    "Rheumatic fever and chronic rheumatic heart diseases,phecode_CV_400"
    "Anemia,phecode_BI_164"
    "Back pain,phecode_MS_718"
    "Parkinson's disease (Primary),phecode_NS_324.11"
    "Rheumatoid arthritis,phecode_MS_705.1"
    "Psoriasis,phecode_DE_664.4"
    "Suicide ideation and attempt or self harm,phecode_MB_284"
    "Crohn's disease,phecode_GI_522.11"
    "Alzheimer's disease,phecode_NS_328.11"
    "Rheumatoid arthritis and other inflammatory polyarthropathies,phecode_MS_705"
    "Multiple sclerosis,phecode_NS_326.1"
    "Systemic lupus erythematosus [SLE],phecode_MS_700.11"
    "Spinal muscular atrophy,phecode_GE_972.41"
    "Acute lymphoid leukemia,phecode_CA_121.11"
    "Acute myeloid leukemia,phecode_CA_121.12"
    "Human immunodeficiency virus,phecode_ID_057.1"
    "Sicca syndrome [Sjögren],phecode_MS_700.2"
    "Systemic sclerosis,phecode_MS_700.3"
    "Polymyalgia rheumatica,phecode_MS_705.3"
    "Vascular dementia,phecode_NS_328.12"
    "Kidney stone disease,phecode_GU_585"
    "Urinary tract infection [UTI],phecode_GU_591"
)

# Loop through each disease and phecode pair
for entry in "${diseases[@]}"; do
    # Split each entry into disease and phecode
    IFS=',' read -r disease phecode <<< "$entry"
    
    # Submit job with sbatch, passing both disease and phecode to run_program.sh
    echo "Submitting job for disease: $disease with phecode: $phecode"
    sbatch --export=ALL run_LLM.sh "$disease" "$phecode"
done