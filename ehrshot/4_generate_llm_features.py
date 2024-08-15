import argparse
import pickle
import os
from loguru import logger
from femr.featurizers import FeaturizerList
from femr.labelers import LabeledPatients, load_labeled_patients
from ehrshot.llm_featurizer import LLMFeaturizer
from utils import check_file_existence_and_handle_force_refresh
import numpy as np
from serialization.text_encoder import TextEncoder, LLM2VecLlama3_7B_InstructSupervisedEncoder, LLM2VecLlama3_1_7B_InstructSupervisedEncoder, GTEQwen2_7B_InstructEncoder

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text-based featurizations for LLM models (for all tasks at once)")
    parser.add_argument("--path_to_database", required=True, type=str, help="Path to FEMR patient database")
    parser.add_argument("--path_to_labels_dir", required=True, type=str, help="Path to directory containing saved labels")
    parser.add_argument("--path_to_features_dir", required=True, type=str, help="Path to directory where features will be saved")
    parser.add_argument("--num_threads", type=int, help="Number of threads to use")
    parser.add_argument("--is_force_refresh", action='store_true', default=False, help="If set, then overwrite all outputs")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    NUM_THREADS: int = args.num_threads
    IS_FORCE_REFRESH = args.is_force_refresh
    PATH_TO_PATIENT_DATABASE = args.path_to_database
    PATH_TO_LABELS_DIR = args.path_to_labels_dir
    PATH_TO_FEATURES_DIR = args.path_to_features_dir
    PATH_TO_LABELS_FILE: str = os.path.join(PATH_TO_LABELS_DIR, 'all_labels.csv')
    # TODO
    PATH_TO_OUTPUT_FILE: str = os.path.join(PATH_TO_FEATURES_DIR, 'llm_features_out.pkl')
    
    # LLM text encoder
    # text_encoder = TextEncoder(LLM2VecLlama3_7B_InstructSupervisedEncoder())
    text_encoder = TextEncoder(LLM2VecLlama3_1_7B_InstructSupervisedEncoder())
    # text_encoder = TextEncoder(GTEQwen2_7B_InstructEncoder())

    # Force refresh
    check_file_existence_and_handle_force_refresh(PATH_TO_OUTPUT_FILE, IS_FORCE_REFRESH)

    # Load consolidated labels across all patients for all tasks
    logger.info(f"Loading LabeledPatients from `{PATH_TO_LABELS_FILE}`")
    labeled_patients: LabeledPatients = load_labeled_patients(PATH_TO_LABELS_FILE)
    # Debug: Only consider first 10 patients with at most 20 labels
    # labeled_patients.patients_to_labels = {k: v[:20] for k, v in list(labeled_patients.patients_to_labels.items())[:10]}

    # Combine two featurizations of each patient: one for the patient's age, and one for the text of every code
    # they've had in their record up to the prediction timepoint for each label
    llm_featurizer = LLMFeaturizer(text_encoder.encoder.embedding_size)
    featurizer_text = FeaturizerList([llm_featurizer])

    # Preprocessing the featurizers -- this includes processes such as normalizing age
    logger.info("Start | Preprocess featurizers")
    featurizer_text.preprocess_featurizers(PATH_TO_PATIENT_DATABASE, labeled_patients, NUM_THREADS)
    logger.info("Finish | Preprocess featurizers")
    
    # Run text encoding on serializations of patients - must be done separately to prevent multiprocessing issue with CUDA
    featurizer_text.featurizers[0].encode_serializations(text_encoder)

    # Run actual featurization for each patient
    logger.info("Start | Featurize patients")
    results = featurizer_text.featurize(PATH_TO_PATIENT_DATABASE, labeled_patients, NUM_THREADS)
    feature_matrix, patient_ids, label_values, label_times = (
        results[0],
        results[1],
        results[2],
        results[3],
    )
    logger.info("Finish | Featurize patients")
    
    # Ensure that all final features sum up to the same value as the generated embeddings
    assert np.allclose(featurizer_text.featurizers[0].embeddings, feature_matrix.toarray())

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
    