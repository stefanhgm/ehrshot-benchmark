#!/bin/bash


CODE_DIR=/home/gear11/Documents/LLM2Vec_project/bash_scripts

cd $CODE_DIR


# List of diseases from medical history paper
diseases=(
    ### Hospitalization
    "hospitalization,OMOP_9201"   

    ### Death
    "death,OMOP_4306655"

    # ### Medical hisotry diseases
    "Hypertension,phecode_CV_401"
    "Diabetes mellitus,phecode_EM_202"
    "Atrial fibrillation,phecode_CV_416.21", 
    "Pneumonia,phecode_RE_468"
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
)


minyears=0
maxyears=1

# Define the models to use
models=(
    #"LLM2Vec"          # uncomment for calculating LLM2Vec embeddings
    #"Qwen"             # uncomment for calculating Qwen2 embeddings
    "Qwen3"             # uncomment for calculating Qwen3 embeddings
    #"BioClinicalBERT"  # uncomment for calculating BioClinicalBERT embeddings (Similar to MEME approach)
    #"runall"           # uncomment to perform downstream prediction - need to calculate embeddings first
)


# Loop through each disease and phecode pair
for entry in "${diseases[@]}"; do
    # Split each entry into disease and phecode
    IFS=',' read -r disease phecode <<< "$entry"
    for model in "${models[@]}"; do
        echo "Submitting job for disease: $disease with phecode: $phecode and model: $model"
        if [ "$model" == "LLM2Vec" ]; then
            sbatch --export=ALL run_LLM.sh "$disease" "$phecode" "$minyears" "$maxyears" "$model" 
        elif [ "$model" == "Qwen" ]; then
            sbatch --export=ALL run_Qwen.sh "$disease" "$phecode" "$minyears" "$maxyears" "$model" 
        elif [ "$model" == "Qwen3" ]; then
            sbatch --export=ALL run_Qwen3.sh "$disease" "$phecode" "$minyears" "$maxyears" "$model" 
        elif [ "$model" == "BioClinicalBERT" ]; then
            sbatch --export=ALL run_BERT.sh "$disease" "$phecode" "$minyears" "$maxyears" "$model" 
        elif [ "$model" == "runall" ]; then
            sbatch --export=ALL run_inference.sh "$disease" "$phecode" "$minyears" "$maxyears" "$model" 
        fi
    done
done
