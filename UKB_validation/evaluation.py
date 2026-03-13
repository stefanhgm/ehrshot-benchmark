from LLM2Vec_embeddingscreation import process_embeddings

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import collections
import pandas as pd
import sklearn
import torch
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from loguru import logger
from sklearn.preprocessing import MaxAbsScaler
from my_utils import (
    LR_PARAMS,
    XGB_PARAMS,
    SHOT_STRATS,
    MODEL_2_INFO,
    MODELS
)


from sklearn.model_selection import GridSearchCV, PredefinedSplit
from scipy.sparse import issparse
import scipy.sparse as sp

import scipy
import lightgbm as lgb




"""Create a file at `PATH_TO_LABELS_AND_FEATS_DIR/LABELING_FUNCTION/{SHOT_STRAT}_results.csv` containing:
    Output is a CSV with headers:
        sub_task, model, head, replicate, score_name, score_value, k

        
        """


## read in embeddings
def load_embeddings(embedding_path, phecode):
    kwargs={}
    config={}
    kwargs["calculate_embeddings"] = False
    kwargs["infer_all"] = True #change to your own path - here, all embeddings should be stored
    #task="hospitalization" #phecode

    # just for quicker verification before all embeddings are calculated
    if phecode == "OMOP_9201":
        task = "hospitalization"
    elif phecode == "OMOP_4306655":
        task = "death"
    else:
        #task = "disease_onset" #
        task = phecode

    config["embeddingfile_qwen"] = f"{embedding_path}embeddings_{task}_bigdataset_8192_dates_query_Qwen0-1.feather"
    config["embeddingfile_qwen3"] = f"{embedding_path}embeddings_{task}_bigdataset_8192_dates_query_Qwen30-1.feather"
    config["embeddingfile_llm2vec"] = f"{embedding_path}embeddings_{task}_bigdataset_8192_dates_query_LLM2Vec0-1.feather"
    config["embeddingfile_bioclinicalbert"] = f"{embedding_path}embeddings_{task}_bigdataset_8192_dates_query_BioClinicalBERT0-1.feather"
    config["embeddingfile_qwenclmbrcodes"] = f"{embedding_path}embeddings_{task}_bigdataset_8192_dates_query_clmbrcodes_Qwen30-1.feather"
    config["embeddingfile_clmbr"] = f"{embedding_path}embeddings_clmbr_5000-1.feather"
    config["useNVEmbed"] = False

    embedding_df = process_embeddings(pd.DataFrame(), config, " ", **kwargs)
    return embedding_df


#models = ["qwen", "qwen3", "llm2vec", "clmbr"]
models = ["qwen3", "clmbr", "bioclinicalbert", "qwenclmbrcodes"]


def get_labels_and_features(PATH_TO_LABELLED_PATIENTS, embedding_path, task, counts_df=None, counts_eids=None):
    """Given a path to a directory containing labels and features as well as a LabeledPatients object, returns
        the labels and features for each patient. Note that this function is more complex b/c we need to align
        the labels with their corresponding features based on their prediction times."""
    df = pd.read_csv(PATH_TO_LABELLED_PATIENTS)
    label_patient_ids = df['patient_id'].tolist()
    #label_values = df['value'].tolist()
    label_values = (
        df
        .set_index("patient_id")["value"]
    )

    embedding_df = load_embeddings(embedding_path, task)

    # sort embedding_df based on sort_order
    embedding_df = (
        embedding_df
            .set_index("eid")
            .reindex(label_patient_ids)
            #.reindex(label_patient_ids)
        )

    if(counts_df is not None):
        # counts_df = counts_df.loc[label_patient_ids]
        # counts_matrix = sp.csr_matrix(counts_df.values)
        eid_to_row = {eid: i for i, eid in enumerate(counts_eids)}
        #rows = [eid_to_row[eid] for eid in label_patient_ids]
        # keep only patients present in counts_eids
        valid_eids = [eid for eid in label_patient_ids if eid in eid_to_row]

        rows = [eid_to_row[eid] for eid in valid_eids]

        counts_matrix = counts_df[rows]
        
        # also filter labels to keep alignment
        label_values = label_values.loc[valid_eids]
        embedding_df = embedding_df.loc[valid_eids]


    # in embedding_df column names: remove q_reps_ suffix
    embedding_df.columns = [col.replace("q_reps_", "") for col in embedding_df.columns]

    # Go through every featurization we've created (e.g. count, clmbr, motor)
    # and align the label times with the featurization times
    featurizations: Dict[str, np.ndarray] = {}

    for model in models:
        #featurizations[model] = embedding_df[model]
        featurizations[model] = np.stack(embedding_df[model].values)
    featurizations["count"] = counts_matrix if counts_df is not None else None
    
    return label_patient_ids, label_values, featurizations


# PATH_TO_LABELLED_PATIENTS = "/home/gear11/Documents/LLM2Vec_project/Splits/generate_labels_2/labeled_patients_hospitalization.csv"
# label_patient_ids, label_values, feature_matrixes = get_labels_and_features(PATH_TO_LABELLED_PATIENTS)
# feature_matrixes




def tune_hyperparams(X_train: np.ndarray, X_val: np.ndarray, y_train: np.ndarray, y_val: np.ndarray, model, param_grid: Dict[str, List], n_jobs: int = 1):
    """Use GridSearchCV to do hyperparam tuning, but we want to explicitly specify the train/val split.
        Thus, we ned to use `PredefinedSplit` to force the proper splits."""
    # First, concatenate train/val sets (NOTE: need to do concatenation slightly diff for sparse arrays)
    X: np.ndarray = scipy.sparse.vstack([X_train, X_val]) if issparse(X_train) else np.concatenate((X_train, X_val), axis=0)
    y: np.ndarray = np.concatenate((y_train, y_val), axis=0)
    # In PredefinedSplit, -1 = training example, and 0 = validation example
    test_fold: np.ndarray = -np.ones(X.shape[0])
    test_fold[X_train.shape[0]:] = 0
    # Fit model
    clf = GridSearchCV(model, param_grid, scoring='roc_auc', n_jobs=n_jobs, verbose=0, cv=PredefinedSplit(test_fold), refit=False)
    clf.fit(X, y)
    best_model = model.__class__(**clf.best_params_)
    best_model.fit(X_train, y_train) # refit on only training data so that we are truly do `k`-shot learning
    return best_model

def run_evaluation(X_train: np.ndarray, 
                    X_val: np.ndarray, 
                    X_test: np.ndarray, 
                    y_train: np.ndarray, 
                    y_val: np.ndarray, 
                    y_test: np.ndarray, 
                    model_head: str, 
                    n_jobs: int = 1,
                    test_patient_ids: np.ndarray = None) -> Tuple[Any, Dict[str, float], np.ndarray]:

    # Shuffle training set
    np.random.seed(X_train.shape[0])
    # train_shuffle_idx = np.arange(X_train.shape[0])
    # np.random.shuffle(train_shuffle_idx)
    # X_train = X_train.iloc[train_shuffle_idx]
    # y_train = y_train[train_shuffle_idx]
    # X_train = torch.stack(X_train.tolist()).numpy()
    # X_val = torch.stack(X_val.tolist()).numpy()
    # X_test = torch.stack(X_test.tolist()).numpy()
    train_shuffle_idx = np.random.permutation(X_train.shape[0])
    X_train = X_train[train_shuffle_idx]
    y_train = y_train[train_shuffle_idx]

    logger.critical(f"Start | Fitting {model_head}...")
    model_head_parts: List[str] = model_head.split("_")
    model_head_base: str = model_head_parts[0]
    if model_head_base == "gbm":
        # XGBoost
        model = lgb.LGBMClassifier(random_state=0)
        # NOTE: Need to set `min_child_samples = 1`, which specifies the minimum number of samples required in a leaf (terminal node).
        # This is necessary for few-shot learning, since we may have very few samples in a leaf node.
        # Otherwise the GBM model will refuse to learn anything
        XGB_PARAMS['min_child_samples'] = [ 1 ]
        model = tune_hyperparams(X_train, X_val, y_train, y_val, model, XGB_PARAMS, n_jobs=n_jobs)
        logger.info(f"Best hparams: {model.get_params()}")
    elif model_head_base == "lr":
        # Logistic Regresion
        solver: str = model_head_parts[1] # "newton-cg" or "lbfgs" etc.
        # Use built-in SKLearn solver
        scaler = MaxAbsScaler().fit(X_train)
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
        model = LogisticRegression(n_jobs=1, penalty="l2", tol=0.0001, solver=solver, max_iter=1000, random_state=0)
        model = tune_hyperparams(X_train, X_val, y_train, y_val, model, LR_PARAMS, n_jobs=n_jobs)
        logger.info(f"Best hparams: {model.get_params()}")
    else:
        raise ValueError(f"Model head `{model_head}` not supported.")
    logger.critical(f"Finish | Fitting {model_head}...")
    
    # Calculate probabilistic preds
    y_train_proba = model.predict_proba(X_train)[::, 1]
    y_val_proba = model.predict_proba(X_val)[::, 1]
    y_test_proba = model.predict_proba(X_test)[::, 1]
    
    metric_dict = {
        'auroc': metrics.roc_auc_score,
        'brier': metrics.brier_score_loss,
        'auprc': metrics.average_precision_score,
    }
    
    # Calculate metrics
    scores = {}
    for metric, func in metric_dict.items():
        scores[metric] = {}
        train_score = func(y_train, y_train_proba)
        val_score = func(y_val, y_val_proba)
        test_score = func(y_test, y_test_proba)

        logger.info(f"Train {metric} score: {train_score}")
        logger.info(f"Val {metric} score:   {val_score}")
        logger.info(f"Test {metric} score:  {test_score}")

        #test_set = sorted(list(set(test_patient_ids)))
        test_indices = np.arange(len(y_test))
        rng = np.random.default_rng(42)
        n = len(test_indices)


        score_list = []
        for i in range(1000): # 1k bootstrap replicates
            # sample = sklearn.utils.resample(test_set, random_state=i)
            # counts = collections.Counter(sample)
            # weights = np.zeros_like(test_patient_ids)

            # for i, p in enumerate(test_patient_ids):
            #     weights[i] = counts[p]

            # score_val = func(y_test, y_test_proba, sample_weight=weights)
            # score_list.append(score_val)
            sample_idx = rng.integers(0, n, size=n, dtype=int)
            score_val = func(y_test[sample_idx], y_test_proba[sample_idx])
            score_list.append(score_val)


        # 95% CI
        lower, upper = np.percentile(score_list, [2.5, 97.5])

        # Std
        std = np.std(score_list, ddof=1)

        scores[metric]['score'] = test_score
        scores[metric]['std'] = std
        scores[metric]['lower'] = lower
        scores[metric]['mean'] = np.mean(score_list)
        scores[metric]['upper'] = upper

    return model, scores, y_test_proba


#if __name__ == "__main__":
def main_evaluation(disease, phecode, clmbrcodes=False, counts_df=None, counts_eids=None):    
    SHOT_STRAT = "all"
    PATH_TO_SHOTS = f"./Splits/kshots/{SHOT_STRAT}_{disease}_shots_data.json"
    PATH_TO_SPLIT_CSV = "/sc-projects/sc-proj-ukb-cvd/projects/llm2vec/data/splits.json"
    PATH_TO_LABELED_PATIENTS=f"/home/gear11/Documents/LLM2Vec_project/Splits/generate_labels_2/labeled_patients_{disease}.csv"
    #PATH_TO_FEATURES_DIR = 
    clmbraddition = "_clmbr" if clmbrcodes else ""
    EMBEDDING_PATH = "/sc-projects/sc-proj-ukb-cvd/projects/llm2vec/data/embeddings/"
    LABELING_FUNCTION = disease
    IS_FORCE_REFRESH=False
    NUM_THREADS = 20
    PATH_TO_OUTPUT_DIR = "./Results_new/"
    #PATH_TO_OUTPUT_DIR = "/sc-projects/sc-proj-ukb-cvd/projects/llm2vec/Results_Revision2/"
    PATH_TO_OUTPUT_FILE = os.path.join(PATH_TO_OUTPUT_DIR, f'{LABELING_FUNCTION}_{SHOT_STRAT}_results{clmbraddition}.csv')
    PATH_TO_PROBA_OUTPUT_FILE: str = os.path.join(PATH_TO_OUTPUT_DIR, f'{LABELING_FUNCTION}_{SHOT_STRAT}_probas{clmbraddition}.csv')

    os.makedirs(os.path.dirname(PATH_TO_OUTPUT_FILE), exist_ok=True)

 
    # If results already exist, then append new results to existing file
    df_existing: Optional[pd.DataFrame] = None
    if os.path.exists(PATH_TO_OUTPUT_FILE):
        logger.warning(f"Results already exist @ `{PATH_TO_OUTPUT_FILE}`.")
        print(f"Results already exist @ `{PATH_TO_OUTPUT_FILE}`.")
        df_existing = pd.read_csv(PATH_TO_OUTPUT_FILE)

    # Load FEMR Patient Database
    #database = femr.datasets.PatientDatabase(PATH_TO_DATABASE)

    # Load labels for this task
    #labeled_patients = load_labeled_patients(PATH_TO_LABELED_PATIENTS)
    patient_ids, label_values, feature_matrixes = get_labels_and_features(PATH_TO_LABELED_PATIENTS, EMBEDDING_PATH, phecode, counts_df=counts_df, counts_eids=counts_eids)
    #train_pids_idx, val_pids_idx, test_pids_idx = get_patient_splits_by_idx(PATH_TO_SPLIT_CSV, patient_ids)
    
    # Create patient_id -> row index mapping
    patient_id_to_idx = {pid: i for i, pid in enumerate(patient_ids)}

    with open(PATH_TO_SPLIT_CSV, "r") as f:
        loaded_splits = json.load(f)

    # train_pids_idx = loaded_splits["train"]
    # val_pids_idx = loaded_splits["val"]
    test_pids_idx = loaded_splits["test"]
    # # filter the train, test and val pids
    # train_pids_idx = np.array(train_pids_idx)[np.isin(train_pids_idx, patient_ids)]
    # val_pids_idx = np.array(val_pids_idx)[np.isin(val_pids_idx, patient_ids)]
    test_pids_idx = np.array(test_pids_idx)[np.isin(test_pids_idx, patient_ids)]
    
    # train_indices = np.array([patient_id_to_idx[pid] for pid in train_pids_idx])
    # val_indices   = np.array([patient_id_to_idx[pid] for pid in val_pids_idx])
    test_indices  = np.array([patient_id_to_idx[pid] for pid in test_pids_idx])


    # Load shot assignments for this task
    with open(PATH_TO_SHOTS) as f:
        few_shots_dict: Dict[str, Dict] = json.load(f)

    sub_tasks: List[str] = [LABELING_FUNCTION]
        
    # Results will be stored as a CSV with columns:
    #   sub_task, model, head, replicate, score_name, score_value, k
    results: List[Dict[str, Any]] = []
    probas: List[Dict[str, Any]] = []

    logger.info(f"Starting evaluation, using {NUM_THREADS} threads for parameter tuning.")
    
    # For each base model we are evaluating...
    for model in MODEL_2_INFO.keys():
        model_heads: List[str] = MODEL_2_INFO[model]['heads']
        # For each head we can add to the top of this model...
        for head in model_heads:
            # Unpack each individual featurization we want to test
            assert model in feature_matrixes, f"Feature matrix not found for `{model}`. Are you sure you have generated features for this model? If not, you'll need to rerun `generate_features.py` or `generate_clmbr_representations.py`."
            # X_train: np.ndarray = feature_matrixes[model][train_pids_idx]
            # X_val: np.ndarray = feature_matrixes[model][val_pids_idx]
            # X_test: np.ndarray = feature_matrixes[model][test_pids_idx]
            # X_train = feature_matrixes[model][train_indices]
            # X_val   = feature_matrixes[model][val_indices]
            X_test  = feature_matrixes[model][test_indices]
            y_test: np.ndarray = label_values[test_pids_idx].values
            
            test_patient_ids = test_pids_idx #patient_ids[test_pids_idx]
            
            # For each subtask in this task... 
            # NOTE: The "subtask" is just the same thing as LABELING_FUNCTION for all binary tasks.
            # But for Chexpert, there are multiple subtasks, which of each represents a binary subtask
            for sub_task_idx, sub_task in enumerate(sub_tasks):
                # Check if results already exist for this model/head/shot_strat in `results.csv`
                if df_existing is not None:
                    existing_rows: pd.DataFrame = df_existing[
                        (df_existing['labeling_function'] == LABELING_FUNCTION) 
                        & (df_existing['sub_task'] == sub_task) 
                        & (df_existing['model'] == model) 
                        & (df_existing['head'] == head)
                    ]
                    if existing_rows.shape[0] > 0:
                        # Overwrite
                        if IS_FORCE_REFRESH:
                            logger.warning(f"Results ALREADY exist for {model}/{head}:{LABELING_FUNCTION}/{sub_task} in `results.csv`. Overwriting these rows because `is_force_refresh` is TRUE.")
                        else:
                            logger.warning(f"Results ALREADY exist for {model}/{head}:{LABELING_FUNCTION}/{sub_task} in `results.csv`. Skipping this combination because `is_force_refresh` is FALSE.")
                            results += existing_rows.to_dict(orient='records')
                            continue
                    else:
                        # Append
                        logger.warning(f"Results DO NOT exist for {model}/{head}:{LABELING_FUNCTION}/{sub_task} in `results.csv`. Appending to this CSV.")
        
                ks: List[int] = sorted([ int(x) for x in few_shots_dict[sub_task].keys() ])
                
                # For each k-shot sample we are evaluating...
                for k in ks:
                    replicates: List[int] = sorted([ int(x) for x in few_shots_dict[sub_task][str(k)].keys() ])

                    # For each replicate of this k-shot sample...
                    for replicate in replicates:
                        logger.success(f"Model: {model} | Head: {head} | Task: {sub_task} | k: {k} | replicate: {replicate}")
                        
                        # Get X/Y train/val for this k-shot sample     
                        shot_dict: Dict[str, List[int]] = few_shots_dict[sub_task][str(k)][str(replicate)]      
                        train_k_indices = np.array([patient_id_to_idx[pid] 
                            for pid in shot_dict["patient_ids_train_k"]])

                        val_k_indices = np.array([patient_id_to_idx[pid] 
                          for pid in shot_dict["patient_ids_val_k"]])         
                        #X_train_k: np.ndarray = X_train[shot_dict["train_idxs"]]
                        #X_val_k: np.ndarray = X_val[shot_dict["val_idxs"]]
                        # X_train_k: np.ndarray = X_train[shot_dict["patient_ids_train_k"]]
                        # X_val_k: np.ndarray = X_val[shot_dict["patient_ids_val_k"]]
                        X_train_k: np.ndarray = feature_matrixes[model][train_k_indices]
                        X_val_k: np.ndarray = feature_matrixes[model][val_k_indices]
                        y_train_k: np.ndarray = np.array(shot_dict['label_values_train_k'])
                        y_val_k: np.ndarray = np.array(shot_dict['label_values_val_k'])
                        y_test_k: np.ndarray = np.array(y_test)


                        # Fit model with hyperparameter tuning
                        best_model, scores, y_test_proba = run_evaluation(
                            X_train_k, X_val_k, X_test, y_train_k, y_val_k, y_test_k,
                            model_head=head, n_jobs=NUM_THREADS, test_patient_ids=test_patient_ids
                        )

                        # Save probabilities (test set)
                        for pid, lbl, proba in zip(test_patient_ids, y_test_k, y_test_proba):
                            probas.append({
                                'patient_id': pid,
                                'sub_task': sub_task,
                                'model': model,
                                'head': head,
                                'replicate': replicate,
                                'k': k,
                                'label': lbl,
                                'proba': proba,
                            })

                        # Save results
                        for score_name, score_value in scores.items():
                            results.append({
                                'labeling_function' : LABELING_FUNCTION,
                                'sub_task' : sub_task,
                                'model' : model,
                                'head' : head,
                                'replicate' : replicate,
                                'k' : k,
                                'score' : score_name,
                                'value' : score_value['score'],
                                'std' : score_value['std'],
                                'lower' : score_value['lower'],
                                'mean' : score_value['mean'],
                                'upper' : score_value['upper'],
                            })

    logger.info(f"Saving results to: {PATH_TO_OUTPUT_FILE}")
    df: pd.DataFrame = pd.DataFrame(results)
    logger.info(f"Added {df.shape[0] - (df_existing.shape[0] if df_existing is not None else 0)} rows")
    df.to_csv(PATH_TO_OUTPUT_FILE)
    logger.success("Done!")

    logger.info(f"Saving probabilities to: {PATH_TO_PROBA_OUTPUT_FILE}")
    df_proba: pd.DataFrame = pd.DataFrame(probas, columns=['patient_id','sub_task','model','head','replicate','k','label','proba'])
    df_proba.to_csv(PATH_TO_PROBA_OUTPUT_FILE, index=False)
    logger.success("Done saving probabilities!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Your script description")
    parser.add_argument("--indication", type=str, required=True, help="Disease to evaluate on (e.g. `hospitalization` or `phecode_411`)")
    parser.add_argument("--phecode", type=str, required=True, help="Phecode to evaluate on (e.g. `411`)")
    args = parser.parse_args()
    main_evaluation(args.indication, args.phecode)