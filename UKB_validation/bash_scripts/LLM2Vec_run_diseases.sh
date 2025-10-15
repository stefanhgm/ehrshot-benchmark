#!/bin/bash


CODE_DIR=/home/gear11/ehrshot-benchmark/UKB_validation/LLM2Vec_project/bash_scripts

cd $CODE_DIR



#source ~/.bashrc
#conda init

#conda activate ~/.conda/envs/LLM
#conda activate /sc-projects/sc-proj-ukb-cvd/environments/LLM

# List of diseases from medical history paper
diseases=(
    ### Hospitalization
    #"hospitalization,admin_hospital"   
    "hospitalization,OMOP_9201"   

    ### Death
    #"death,admin_death"
    "death,OMOP_4306655"

    # ### Medical hisotry diseases
    "Hypertension,phecode_CV_401"
    "Diabetes mellitus,phecode_EM_202"
    "Atrial fibrillation,phecode_CV_416.21", #  "Atrial fibrillation"
    "Pneumonia,phecode_RE_468"
    "Chronic obstructive pulmonary disease [COPD],phecode_RE_474"
    "Chronic kidney disease,phecode_GU_582.2"

    "Ischemic heart disease,phecode_CV_404"
    "Myocardial infarction [Heart attack],phecode_CV_404.1"
    "Cerebral infarction [Ischemic stroke],phecode_CV_431.11"
    "Heart failure,phecode_CV_424"
    "Cardiac arrest,phecode_CV_420"
    #'OMOP_4306655', #  "All-Cause Death", # intervention

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

# CLMBR_diseases=(
#     "Hyperlipidemia, "
#     "Pancreatic cancer, "
#     "Celiac, "
#     "Lupus, "
# )

# only if calculating embeddings for diseaseunspecific version
# diseases=(
#     # "Rheumatoid arthritis and other inflammatory polyarthropathies,phecode_MS_705.1"
#     "hospitalization,OMOP_9201" #"hospitalization,admin_hospital"
#     #"death,OMOP_4306655" #"death,admin_death"
# )

start_end=(
    # "0,80000"
    # "80000,170000"
    # "170000,260000"
    # "260000,400000"
    "0,400000"
    # "40000,80000"
    # "80000,130000"
    # "130000,170000"
    # "170000,210000"
    # "210000,260000"
    # "260000,300000"
    # "300000,400000"
    # "270000,280000"
    # "280000,290000"
    # "290000,300000"
    # "260000,400000"
    # "260000,300000"
    # "300000,330000"
    # "330000,400000"
    # "0,130000"
    # "130000,260000"
    #"0,400000"
    # "0,1700000"
    # "1700000,4000000"
)

dataset=(
    "--use_big_dataset"
    #"--use_raw_dataset"
)

num_PCA=0 #set to 0 if you don't want to use PCA
minyears=0
maxyears=1

# Define the models to use
models=(
    #"LLM2Vec"
    # "Qwen"
    #"Qwen3"
    # "NVEmbed"
    "runall"
)
balanced_inference=True # only works for runall model option

queryinclusion_options=("True") #("False" "True")
dateinclusion_options=("True") #("False" "True")

# Loop through each parameter option
for dataset_to_use in "${dataset[@]}"; do
    # Loop through each parameter option
    for dateinclusion in "${dateinclusion_options[@]}"; do
        # Loop through each parameter option
        for queryinclusion in "${queryinclusion_options[@]}"; do
            # Loop through each disease and phecode pair
            for entry in "${diseases[@]}"; do
                # Split each entry into disease and phecode
                IFS=',' read -r disease phecode <<< "$entry"
                
                for model in "${models[@]}"; do
                    echo "Submitting job for disease: $disease with phecode: $phecode and model: $model"
                    if [ "$model" == "LLM2Vec" ]; then
                        for start_end_val in "${start_end[@]}"; do
                            IFS=',' read -r start end <<< "$start_end_val"
                            sbatch --export=ALL run_LLM_pgpu.sh "$disease" "$phecode" "$minyears" "$maxyears" "$model" "$num_PCA" "$queryinclusion" "$dateinclusion" "$dataset_to_use" "$start" "$end"
                        done
                    elif [ "$model" == "NVEmbed" ]; then
                        sbatch --export=ALL run_NV.sh "$disease" "$phecode" "$minyears" "$maxyears" "$model" "$num_PCA" "$queryinclusion" "$dateinclusion" "$dataset_to_use"
                    elif [ "$model" == "Qwen" ] || [ "$model" == "Qwen3" ]; then
                        for start_end_val in "${start_end[@]}"; do
                            IFS=',' read -r start end <<< "$start_end_val"
                            sbatch --export=ALL run_Qwen.sh "$disease" "$phecode" "$minyears" "$maxyears" "$model" "$num_PCA" "$queryinclusion" "$dateinclusion" "$dataset_to_use" "$start" "$end"
                        done
                        #sbatch --export=ALL run_Qwen.sh "$disease" "$phecode" "$minyears" "$maxyears" "$model" "$num_PCA" "$queryinclusion" "$dateinclusion" "$dataset_to_use"
                    elif [ "$model" == "runall" ]; then
                        if [ "$balanced_inference" == "True" ]; then
                            sbatch --export=ALL run_inference_balanced.sh "$disease" "$phecode" "$minyears" "$maxyears" "$model" "$num_PCA" "$queryinclusion" "$dateinclusion" "$dataset_to_use"
                        else
                            sbatch --export=ALL run_inference.sh "$disease" "$phecode" "$minyears" "$maxyears" "$model" "$num_PCA" "$queryinclusion" "$dateinclusion" "$dataset_to_use"
                        fi
                    fi
                done
            done
        done
    done
done
