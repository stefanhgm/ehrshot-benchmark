import argparse
import pickle
import os
from loguru import logger
from utils import check_file_existence_and_handle_force_refresh
from typing import Dict, List, Tuple
import numpy as np
from serialization.text_encoder import TextEncoder, LLM2VecLlama3_7B_InstructSupervisedEncoder, LLM2VecLlama3_1_7B_InstructSupervisedEncoder, GTEQwen2_7B_InstructEncoder, STGTELargeENv15Encoder, BioClinicalBert, LongformerLargeEncoder
from datetime import datetime
from llm_featurizer import preprocess_llm_featurizer, featurize_llm_featurizer, load_labeled_patients_with_tasks

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text-based featurizations for LLM models (for all tasks at once)")
    parser.add_argument("--path_to_database", required=True, type=str, help="Path to FEMR patient database")
    parser.add_argument("--path_to_labels_dir", required=True, type=str, help="Path to directory containing saved labels")
    parser.add_argument("--path_to_features_dir", required=True, type=str, help="Path to directory where features will be saved")
    parser.add_argument("--num_threads", type=int, help="Number of threads to use")
    parser.add_argument("--is_force_refresh", action='store_true', default=False, help="If set, then overwrite all outputs")
    parser.add_argument("--text_encoder", type=str, help="Text encoder to use")
    return parser.parse_args()
    
if __name__ == "__main__":
    args = parse_args()
    NUM_THREADS: int = args.num_threads
    IS_FORCE_REFRESH = args.is_force_refresh
    PATH_TO_PATIENT_DATABASE = args.path_to_database
    PATH_TO_LABELS_DIR = args.path_to_labels_dir
    PATH_TO_FEATURES_DIR = args.path_to_features_dir
    PATH_TO_LABELS_FILE: str = os.path.join(PATH_TO_LABELS_DIR, 'all_labels_tasks.csv')
    
    # LLM text encoder
    # Debug: Use specific text encoder
    # args.text_encoder = 'llm2vec_llama3_1_7b_instruct_supervised'
    if args.text_encoder == 'llm2vec_llama3_7b_instruct_supervised':
        text_encoder = TextEncoder(LLM2VecLlama3_7B_InstructSupervisedEncoder())
    elif args.text_encoder == 'llm2vec_llama3_1_7b_instruct_supervised':
        text_encoder = TextEncoder(LLM2VecLlama3_1_7B_InstructSupervisedEncoder())
    elif args.text_encoder == 'gteqwen2_7b_instruct':
        text_encoder = TextEncoder(GTEQwen2_7B_InstructEncoder())
    elif args.text_encoder == 'st_gte_large_en_v15':
        text_encoder = TextEncoder(STGTELargeENv15Encoder())
    elif args.text_encoder == 'bioclinicalbert-fl':
        text_encoder = TextEncoder(BioClinicalBert())
    elif args.text_encoder == 'bioclinicalbert-fl-average-chunks':
        text_encoder = TextEncoder(BioClinicalBert(handle_long_texts='average_chunks'))
    elif args.text_encoder == 'longformerlarge-fl':
        text_encoder = TextEncoder(LongformerLargeEncoder())
    elif args.text_encoder == 'biomedicallongformerlarge-fl':
        text_encoder = TextEncoder(LongformerLargeEncoder(biomedical=True))
    else:
        raise ValueError(f"Text encoder `{args.text_encoder}` not recognized")
    # Add date and time (hh-mm-ss) to name
    output_file_name = f'llm_features_{args.text_encoder}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.pkl'
    PATH_TO_OUTPUT_FILE = os.path.join(PATH_TO_FEATURES_DIR, output_file_name)

    # Force refresh
    check_file_existence_and_handle_force_refresh(PATH_TO_OUTPUT_FILE, IS_FORCE_REFRESH)

    # Load consolidated labels across all patients for all tasks
    logger.info(f"Loading LabeledPatients from `{PATH_TO_LABELS_FILE}`")
    patients_to_labels: Dict[int, List[Tuple[datetime, str]]] = load_labeled_patients_with_tasks(PATH_TO_LABELS_FILE)
    # Debug: Consider subset of patients
    # patients_to_labels = {k: v for k, v in list(patients_to_labels.items())[:10]}
    logger.info(f"Loaded {len(patients_to_labels)} patients with {sum([len(v) for v in patients_to_labels.values()])} labels")

    # Combine two featurizations of each patient: one for the patient's age, and one for the text of every code
    # they've had in their record up to the prediction timepoint for each label
    logger.info("Start | Preprocess featurizers")
    llm_featurizer = preprocess_llm_featurizer(PATH_TO_PATIENT_DATABASE, patients_to_labels, text_encoder.encoder.embedding_size, NUM_THREADS)

    logger.info("Finish | Preprocess featurizers")
    
    # Run text encoding on serializations of patients - must be done separately to prevent multiprocessing issue with CUDA
    llm_featurizer.encode_serializations(text_encoder)
    # Encoder not necessary anymore
    del text_encoder

    # Featurization only performs serial copying of embeddings.
    # Hence, one thread faster than multiple threads.
    logger.info("Start | Featurize patients")
    results = featurize_llm_featurizer(PATH_TO_PATIENT_DATABASE, patients_to_labels, llm_featurizer, num_threads=1)
    feature_matrix, patient_ids, label_values, label_times, label_tasks = (
        results[0],
        results[1],
        results[2],
        results[3],
        results[4],
    )
    logger.info("Finish | Featurize patients")
    
    # Ensure that all final features sum up to the same value as the generated embeddings
    assert np.allclose(llm_featurizer.embeddings, feature_matrix)

    # Save results
    logger.info(f"Saving results to `{PATH_TO_OUTPUT_FILE}`")
    with open(PATH_TO_OUTPUT_FILE, 'wb') as f:
        pickle.dump(results, f)

    # Logging
    logger.info("FeaturizedPatient stats:\n"
                f"feature_matrix={repr(feature_matrix)}\n"
                f"patient_ids={repr(patient_ids)}\n"
                f"label_values={repr(label_values)}\n"
                f"label_times={repr(label_times)}")
    logger.success("Done!")
    