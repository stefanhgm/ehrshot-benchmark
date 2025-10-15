# LLM2Vec_main.py
import pandas as pd
import numpy as np
import time
from datetime import datetime
import re

try:
    from cuml import PCA
    from cuml.decomposition import PCA
    from cuml.linear_model import LogisticRegression
    from cuml.metrics import accuracy_score
except ImportError:
    print("Not running with cuml")

#from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, average_precision_score
import scipy
from scipy import sparse
from scipy.sparse import coo_matrix

from sklearn.model_selection import BaseShuffleSplit
from sklearn.utils import check_random_state
from sklearn.utils.validation import _num_samples, check_array
from sklearn.model_selection._split import _validate_shuffle_split  # Import this function
import warnings

# For tabpfn insted as LR alternative
from tabpfn import TabPFNClassifier

#for gradient boosting machines
from sklearn.model_selection import GridSearchCV, PredefinedSplit
import lightgbm as lgb
from scipy.sparse import issparse
from typing import Dict, List



def datetime_date_to_markdown(dt):
    display_date = dt.strftime("%Y-%m-%d")
    iso_date = dt.strftime("%Y-%m-%d")
    return f"[{display_date}]({iso_date})"


def convert_years_to_years_months(time_float):
    years, months = divmod(round(time_float * 12), 12)
    return f"{years} years and {months} months"

def format_medical_history_single(events):
    """
    Convert a list of timestamped medical events into a formatted medical history.
    
    Args:
        events (list): List of strings in format "X.XX years ago: Condition"
    
    Returns:
        str: Formatted medical history with markdown headers and grouped events
    """
    # Handle empty lists or None values
    if not events or len(events) == 0:
        return "## Medical History\n\nNo medical history available."
        
    # Dictionary to group events by timestamp
    grouped_events = {}
    
    for event in events:
        # Split the event into timestamp and condition
        time_part, condition = event.split(': ', 1)
        
        # Add to grouped_events dictionary
        if time_part not in grouped_events:
            grouped_events[time_part] = []
        grouped_events[time_part].append(condition)
    
    # Start building the output string
    output = ["## Medical History"]
    
    # Sort timestamps in reverse chronological order
    for timestamp in sorted(grouped_events.keys(), 
                          key=lambda x: float(x.split()[0]), 
                          reverse=False):
        # Add section header
        years = timestamp.split()[0]
        formatted_time = convert_years_to_years_months(float(years))
        output.append(f"\n### Outpatient Visit ({formatted_time} before prediction time)")
        
        # Add conditions as bullet points
        for condition in grouped_events[timestamp]:
            output.append(f"* {condition}")
        
    return "\n".join(output)


def format_medical_history(procedures, conditions, medications, proceduresorlabval, recruitmentdate_string, eid, hospital_stay_length):
    """
    Convert three lists of timestamped medical events into a formatted medical history.
    
    Args:
        procedures (list): List of strings in format "X.XX years ago: Procedure"
        conditions (list): List of strings in format "X.XX years ago: Condition"
        medications (list): List of strings in format "X.XX years ago: Medication"
        proceduresorlabval (str): "procedures" or "lab_values", written in EHR
        recruitmentdate_string (str): Date of recruitment in format "YYYY-MM-DD"
        eid (int): EHR ID of the patient
        hospital_stay_length (pd.DataFrame): DataFrame with columns "eid", "date", and "duration_days"
        
    Returns:
        str: Formatted medical history with markdown headers and grouped events
    """
    # Handle empty lists or None values
    if not any([procedures, conditions, medications]):
        return "## Medical History\n\nNo medical history available."
    
    # Dictionary to group events by timestamp
    grouped_events = {}

    hospital_stay_length = hospital_stay_length[hospital_stay_length["eid"] == eid]

    # Extract date using regex
    match = re.search(r'\d{4}-\d{2}-\d{2}', recruitmentdate_string)

    if match:
        date_str = match.group()
        recruitment_date = datetime.strptime(date_str, '%Y-%m-%d').date()
    else:
        print("No date found")
    
    # Process all three types of events
    for event_list, category in [(procedures, proceduresorlabval), 
                                (conditions, "Conditions"), 
                                (medications, "Medications")]:
        if event_list:
            for event in event_list:
                # Split the event into timestamp and description
                time_part, description = event.split(': ', 1)
                # Initialize nested dictionary if timestamp doesn't exist
                if time_part not in grouped_events:
                    grouped_events[time_part] = {
                        proceduresorlabval: [],
                        "Conditions": [],
                        "Medications": []
                    }
                # Add to appropriate category
                grouped_events[time_part][category].append(description)
    
    output = ["## Past Medical Visits \n\n"]
    for timestamp in sorted(grouped_events.keys(),
                        key=lambda x: datetime.strptime(x.split()[0], "%Y-%m-%d"),
                        reverse=True):

        # get date
        date = timestamp.split()[0]
        date_dt = datetime.strptime(date, '%Y-%m-%d').date()
        #date = datetime_date_to_markdown(date)
        days = (recruitment_date - date_dt).days #/ 365.25
        
        if(date_dt in set(hospital_stay_length["date"])):
            add_length = True
            #length = hospital_stay_length[hospital_stay_length["date"].astype(str) == str(date_dt)]["duration_days"].values[0]
            length = hospital_stay_length.loc[hospital_stay_length["date"] == date_dt, "duration_days"].iloc[0]
            
            if("Admission to intensive care unit" in grouped_events[date]["Conditions"]):
                visittype = "Intensive Care Unit Stay"
            elif("Inpatient Visit" in grouped_events[date]["Conditions"]):
                visittype = "Inpatient Visit"
            else:
                visittype = "Outpatient Visit"

            # check if prediction date and length go over current date
            if(date_dt + pd.Timedelta(days=length) >= recruitment_date):
                output.append(f"\n### {visittype} on [{date}]({date}) ({days} days before prediction time, current visit)")
            else:
                output.append(f"\n### {visittype} on [{date}]({date}) ({days} days before prediction time)")


    # Start building the output string
    output = output + ["\n## Detailed Past Medical Visits (most recent first)"]

    # Sort timestamps in reverse chronological order
    for timestamp in sorted(grouped_events.keys(),
                          key=lambda x: datetime.strptime(x.split()[0], "%Y-%m-%d"),
                          reverse=True):
        # Add section header
        # variable to know if length of stay should be added or not
        add_length = False

        # get date
        date = timestamp.split()[0]
        date_dt = datetime.strptime(date, '%Y-%m-%d').date()
        #date = datetime_date_to_markdown(date)

        
        if(date_dt in set(hospital_stay_length["date"])):
            add_length = True
            if("Admission to intensive care unit" in grouped_events[date]["Conditions"]):
                visittype = "Intensive Care Unit Stay"
                length = hospital_stay_length[hospital_stay_length["date"] == date_dt].iloc[0]["duration_days"]
            elif("Inpatient Visit" in grouped_events[date]["Conditions"]):
                visittype = "Inpatient Visit"
                length = hospital_stay_length[hospital_stay_length["date"] == date_dt].iloc[0]["duration_days"]
            else:
                add_length = False
                visittype = "Outpatient Visit"
        else:
            visittype = "Outpatient Visit"

        # also get "time ago"
        #subtract recruitment date from date
        #years = (recruitment_date - datetime.strptime(date, '%Y-%m-%d')).days / 365.25
        days = (recruitment_date - date_dt).days #/ 365.25
        #years = timestamp.split()[0]
        #formatted_time = convert_years_to_years_months(float(years))
        if add_length:
            output.append(f"\n### {visittype} on [{date}]({date}) ({days} days ago, duration: {int(length)} days)")
        else:
            output.append(f"\n### {visittype} on [{date}]({date}) ({days} days ago)")
        
        # Add each category's events as bullet points
        for category in [proceduresorlabval, "Conditions", "Medications"]:
            if grouped_events[timestamp][category]:
                output.append(f"\n#### {category}")
                for item in grouped_events[timestamp][category]:
                    output.append(f"- {item}")
    
    return "\n".join(output)


def format_medical_history_rawdata(labvals, conditions, medications, procedures, recruitmentdate_string, eid, hospital_stay_length):
    """
    Convert three lists of timestamped medical events into a formatted medical history.
    
    Args:
        procedures (list): List of strings in format "X.XX years ago: Procedure"
        conditions (list): List of strings in format "X.XX years ago: Condition"
        medications (list): List of strings in format "X.XX years ago: Medication"
        recruitmentdate_string (str): Date of recruitment in format "YYYY-MM-DD"
        eid (int): EHR ID of the patient
        hospital_stay_length (pd.DataFrame): DataFrame with columns "eid", "date", and "duration_days"
        
    Returns:
        str: Formatted medical history with markdown headers and grouped events
    """
    # Handle empty lists or None values
    if not any([procedures, conditions, medications, labvals]):
        return "## Medical History\n\nNo medical history available."
    
    # Dictionary to group events by timestamp
    grouped_events = {}

    hospital_stay_length = hospital_stay_length[hospital_stay_length["eid"] == eid]

    # Extract date using regex
    match = re.search(r'\d{4}-\d{2}-\d{2}', recruitmentdate_string)

    if match:
        date_str = match.group()
        recruitment_date = datetime.strptime(date_str, '%Y-%m-%d').date()
    else:
        print("No date found")
    
    # Process all three types of events
    for event_list, category in [(procedures, "Procedures"), 
                                (conditions, "Conditions"), 
                                (medications, "Medications"),
                                (labvals, "Lab values") ]:
        if event_list:
            for event in event_list:
                # Split the event into timestamp and description
                time_part, description = event.split(': ', 1)
                # Initialize nested dictionary if timestamp doesn't exist
                if time_part not in grouped_events:
                    grouped_events[time_part] = {
                        "Procedures": [],
                        "Conditions": [],
                        "Medications": [],
                        "Lab values": []
                    }
                # Add to appropriate category
                grouped_events[time_part][category].append(description)
    
    # Start building the output string
    output = ["## General Medical Events \n\n## Detailed Past Medical Visits (most recent first)"]
    
    # Sort timestamps in reverse chronological order
    for timestamp in sorted(grouped_events.keys(),
                          key=lambda x: datetime.strptime(x.split()[0], "%Y-%m-%d"),
                          reverse=True):
        # Add section header
        # variable to know if length of stay should be added or not
        add_length = False

        # get date
        date = timestamp.split()[0]
        date_dt = datetime.strptime(date, '%Y-%m-%d').date()
        #date = datetime_date_to_markdown(date)

        
        if(date_dt in set(hospital_stay_length["date"])):
            add_length = True
            if("Admission to intensive care unit" in grouped_events[date]["Conditions"]):
                visittype = "Intensive Care Unit Stay"
                length = hospital_stay_length[hospital_stay_length["date"] == date_dt].iloc[0]["duration_days"]
            elif("Inpatient Visit" in grouped_events[date]["Conditions"]):
                visittype = "Inpatient Visit"
                length = hospital_stay_length[hospital_stay_length["date"] == date_dt].iloc[0]["duration_days"]
            else:
                add_length = False
                visittype = "Outpatient Visit"
        else:
            visittype = "Outpatient Visit"

        # also get "time ago"
        #subtract recruitment date from date
        #years = (recruitment_date - datetime.strptime(date, '%Y-%m-%d')).days / 365.25
        days = (recruitment_date - date_dt).days #/ 365.25
        #years = timestamp.split()[0]
        #formatted_time = convert_years_to_years_months(float(years))
        if add_length:
            output.append(f"\n### {visittype} on [{date}]({date}) ({days} days ago, duration: {int(length)} days)")
        else:
            output.append(f"\n### {visittype} on [{date}]({date}) ({days} days ago)")
        
        # Add each category's events as bullet points
        for category in ["Procedures", "Conditions", "Medications", "Lab values"]:
            if grouped_events[timestamp][category]:
                output.append(f"\n#### {category}")
                for item in grouped_events[timestamp][category]:
                    output.append(f"- {item}")
    
    return "\n".join(output)




def process_and_encode(prepared_records, model, config):
    """Generates queries, encodes them, and performs PCA."""
    queries = [[config["instruction"], "\n".join(q)] for q in prepared_records.record_list.tolist()]
    q_reps = model.encode(queries, batch_size=config["batch_size"], device=config["device"])
    return q_reps



def get_phenotypes(records, phecode, censoring_times):
    endpoint_records = records[records["concept_id"] == phecode]
    endpoint_prior = endpoint_records[
        endpoint_records["date"] <= endpoint_records["recruitment_date"]
    ]
    endpoint_incident = endpoint_records[
        endpoint_records["date"] > endpoint_records["recruitment_date"]
    ].copy()

    # Ensure sorting by eid and date so first() gets earliest incident date
    endpoint_incident = endpoint_incident.sort_values(["eid", "date"]).copy()

    endpoint_incident[f"{phecode}_time"] = (
        endpoint_incident["date"] - endpoint_incident["recruitment_date"]
    ).dt.days / 365.25

    outcomes = pd.merge(
        endpoint_incident.groupby("eid")
        .first()[[f"{phecode}_time"]]
        .assign(**{f"{phecode}_event": True}),
        endpoint_prior.groupby("eid").first()[[]].assign(**{f"{phecode}_prior": True}),
        left_index=True,
        right_index=True,
        how="outer"
    ).fillna(False).infer_objects(copy=False)

    non_endpoint_outcomes = censoring_times[["censoring_time"]].rename(
        columns={"censoring_time": f"{phecode}_time"}
    )
    non_endpoint_outcomes = non_endpoint_outcomes.loc[
        ~non_endpoint_outcomes.index.isin(outcomes.index)
    ]
    non_endpoint_outcomes[f"{phecode}_event"] = False
    non_endpoint_outcomes[f"{phecode}_prior"] = False

    outcomes = pd.concat([outcomes, non_endpoint_outcomes], axis=0).fillna(False)
    outcomes.sort_index(inplace=True)

    return outcomes



# for gradient boosting machines
def tune_hyperparams(X_train: np.ndarray, X_val: np.ndarray, y_train: np.ndarray, y_val: np.ndarray, model, param_grid: Dict[str, List], n_jobs: int = 1):
    """Use GridSearchCV to do hyperparameter tuning with a predefined train/val split."""
    X = scipy.sparse.vstack([X_train, X_val]) if issparse(X_train) else np.concatenate((X_train, X_val), axis=0)
    y = np.concatenate((y_train, y_val), axis=0)

    test_fold = -np.ones(X.shape[0])
    test_fold[X_train.shape[0]:] = 0

    clf = GridSearchCV(model, param_grid, scoring='roc_auc', n_jobs=n_jobs, verbose=0, cv=PredefinedSplit(test_fold), refit=False)
    clf.fit(X, y)

    best_model = model.__class__(**clf.best_params_)
    best_model.fit(X_train, y_train) # refit on only training data so that we are truly do `k`-shot learning
    return best_model



# Apply PCA to reduce dimensionality of the array
def process_and_encode(prepared_records, pca_components, train_eids, valid_eids, test_eids, q_reps_name):
    """Generates queries, encodes them, and performs PCA."""

    prepared_records_train = prepared_records[prepared_records["eid"].isin(train_eids)]
    prepared_records_valid = prepared_records[prepared_records["eid"].isin(valid_eids)]
    prepared_records_test = prepared_records[prepared_records["eid"].isin(test_eids)]
    q_reps_train = np.stack(prepared_records_train[q_reps_name].to_list())
    q_reps_valid = np.stack(prepared_records_valid[q_reps_name].to_list())
    q_reps_test = np.stack(prepared_records_test[q_reps_name].to_list())
    """embedding_df = pd.DataFrame()
    embedding_df["eid"] = pd.concat([prepared_records_train, prepared_records_valid])["eid"].astype(int)
    embedding_df["q_reps"] = pd.concat([prepared_records_train, prepared_records_valid])["q_reps"].tolist()
    embedding_df["q_reps"] = embedding_df["q_reps"].apply(lambda x: x.tolist())
    embedding_df.to_feather("/sc-projects/sc-proj-ukb-cvd/projects/llm2vec/data_ben/Sinusitis/nvembed_sinusitis_withinfo_embeddingmatrix.feather")"""
    if pca_components == 0: #in case of no pca
        q_reps_pca = np.concatenate([q_reps_train, q_reps_valid, q_reps_test])
        return q_reps_pca, pd.concat([prepared_records_train, prepared_records_valid, prepared_records_test])[["eid"]]
    
    pca = PCA(n_components=pca_components)

    q_reps_train_pca = pca.fit_transform(q_reps_train)
    q_reps_valid_pca = pca.transform(q_reps_valid)
    q_reps_test_pca = pca.transform(q_reps_test)
    q_reps_pca = np.concatenate([q_reps_train_pca, q_reps_valid_pca, q_reps_test_pca]) #in case of pca
    #q_reps_pca = np.concatenate([q_reps_train, q_reps_valid]) #in case of no pca
    return q_reps_pca, pd.concat([prepared_records_train, prepared_records_valid, prepared_records_test])[["eid"]]



# Prepare data for model training and evaluation by merging the feature matrix with survival times and event indicators
def prepare_data(feature_matrix, prepared_records, times, selected_phecode, train_eids, valid_eids, test_eids):
    """Prepares data for model training and evaluation."""
    X = pd.DataFrame(feature_matrix, index=prepared_records.eid)
    #X = pd.DataFrame(feature_matrix, index=np.concatenate([train_eids, valid_eids]))
    #event_times = times[selected_phecode].loc[X.index].copy().to_frame("event_time")
    #event_indicators = events[selected_phecode].loc[X.index].copy().to_frame("event")
    #X = X.join(event_times).join(event_indicators)
    #if config["lr"]:
    #    X = X.drop(columns=['event_time', 'event'])
    return X

def train_and_evaluate_model(X, train_eids, test_eids, cph_penalizer):
    """Trains and evaluates the Cox Proportional Hazards model."""
    cph = CoxPHFitter(penalizer=cph_penalizer)
    cph.fit(X.loc[train_eids], duration_col="event_time", event_col="event")
    test_loghs = cph.predict_partial_hazard(X.loc[test_eids])
    c_index = concordance_index(
        X.loc[test_eids]["event_time"], -test_loghs, X.loc[test_eids]["event"]
    )
    return c_index


def train_and_evaluate_model_lr(X, times, train_eids, val_eids, test_eids, config, target):
    """Trains and evaluates a logistic regression model."""
    #target = (times[config["selected_phecode"]] <= config["years_threshold"]).astype(int)

    # Convert X DataFrame to sparse matrix format
    X_train_sparse = sparse.csr_matrix(X.loc[train_eids].values)
    X_val_sparse = sparse.csr_matrix(X.loc[val_eids].values)

    #TODO: move to config
    LR_PARAMS = {
        "C": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e2, 1e3, 1e4, 1e5, 1e6], 
        "penalty": ['l2']
}

    # Fit the logistic regression model using PCA compressed embeddings
    #logistic_model = logistic_regression_model(X.loc[train_eids], target.loc[train_eids], C_value)
    if(config["tabpfn"]):
        clf = TabPFNClassifier(ignore_pretraining_limits=True)
        clf.fit(X.loc[train_eids], target.loc[train_eids])
        prediction_probabilities = clf.predict_proba(X.loc[test_eids])
        auroc = roc_auc_score(target.loc[test_eids], prediction_probabilities[:, 1])

        # Compute Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(target.loc[test_eids], prediction_probabilities[:, 1])
        auprc = auc(recall, precision)  # Trapezoidal integration

        predictions = clf.predict(X.loc[test_eids])
        accuracy = accuracy_score(target.loc[test_eids], predictions)

    else:
        #model = LogisticRegression(class_weight='balanced', max_iter=1000, C=C_value)
        model = LogisticRegression(max_iter=1000, penalty="l2", random_state=0)
        #model = LogisticRegression(solver='saga', class_weight='balanced', max_iter=1000, C=C_value)
        best_model = tune_hyperparams(X_train_sparse, X_val_sparse, target.loc[train_eids], target.loc[val_eids], model, LR_PARAMS, n_jobs=1)

        #model.fit(X.loc[train_eids], target.loc[train_eids])
        # Predict on the same dataset (or a test set if available)
        predictions = best_model.predict_proba(X.loc[test_eids]).iloc[:,1] #[:, 1]
        # Calculate accuracy (optional)
        accuracy = accuracy_score(target.loc[test_eids], predictions)
        auroc = roc_auc_score(target.loc[test_eids], predictions)

        # Compute Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(target.loc[test_eids], predictions)
        auprc = auc(recall, precision)  # Trapezoidal integration

    #print(f"Model AUROC: {auroc:.4f}")
    return accuracy, auroc, auprc


def train_and_evaluate_model_lr_counts(X, times, train_eids, val_eids, test_eids, config, target):
    """Trains and evaluates a logistic regression model with sparse matrix input."""
    #target = (times[config["selected_phecode"]] <= config["years_threshold"]).astype(int)

    # Convert X DataFrame to sparse matrix format
    X_train_sparse = sparse.csr_matrix(X.loc[train_eids].values)
    X_val_sparse = sparse.csr_matrix(X.loc[val_eids].values)
    X_train_sparse = X_train_sparse.astype(np.float32)
    X_val_sparse = X_val_sparse.astype(np.float32)



    if(config["gbm"]):
        # Instantiate model
        model = lgb.LGBMClassifier(random_state=0)    
        
        #TODO: move to config
        XGB_PARAMS = {
            'max_depth': [3, 6, -1],
            'learning_rate': [0.02, 0.1, 0.5],
            'num_leaves': [10, 25, 100],
            'min_child_samples': [1]  # Necessary for few-shot learning
        }
        LR_PARAMS = {
            "C": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e2, 1e3, 1e4, 1e5, 1e6], 
            "penalty": ['l2']
        }

        # Tune hyperparameters
        best_model = tune_hyperparams(X_train_sparse, X_val_sparse, target.loc[train_eids], target.loc[val_eids], model, XGB_PARAMS, n_jobs=1)
        
        # Predict on the test set
        y_pred = best_model.predict(X.loc[test_eids])
        y_pred_proba = best_model.predict_proba(X.loc[test_eids])[:, 1]  # Get probability for positive class
        
        # Compute metrics
        accuracy = accuracy_score(target.loc[test_eids], y_pred)
        auroc = roc_auc_score(target.loc[test_eids], y_pred_proba)

        # Compute Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(target.loc[test_eids], y_pred_proba)
        auprc = auc(recall, precision)  # Trapezoidal integration



    elif(config["tabpfn"]):
        clf = TabPFNClassifier(ignore_pretraining_limits=True)
        clf.fit(X.loc[train_eids], target.loc[train_eids])
        prediction_probabilities = clf.predict_proba(X.loc[test_eids])
        auroc = roc_auc_score(target.loc[test_eids], prediction_probabilities[:, 1])

        predictions = clf.predict(X.loc[test_eids])
        accuracy = accuracy_score(target.loc[test_eids], predictions)


        # Compute Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(target.loc[test_eids], y_pred_proba)
        auprc = auc(recall, precision)  # Trapezoidal integration

    else:
        # Fit the logistic regression model using sparse matrix
        #model = LogisticRegression(class_weight='balanced', max_iter=1000, C=C_value) #previous version (before 13.05.25)
        model = LogisticRegression(max_iter=1000, penalty="l2", random_state=0)
        best_model = tune_hyperparams(X_train_sparse, X_val_sparse, target.loc[train_eids], target.loc[val_eids], model, LR_PARAMS, n_jobs=1)


        # Predict using sparse matrix
        predictions = best_model.predict_proba(X.loc[test_eids])[:, 1]  # Note: predict_proba returns numpy array, not DataFrame
        #print(target.loc[train_eids][target.loc[train_eids] == 1])

        # Calculate accuracy (optional)
        accuracy = accuracy_score(target.loc[test_eids], predictions)
        auroc = roc_auc_score(target.loc[test_eids], predictions)

        # Compute Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(target.loc[test_eids], y_pred_proba)
        auprc = auc(recall, precision)  # Trapezoidal integration


    return accuracy, auroc, auprc, target.loc[train_eids].sum(), target.loc[test_eids].sum()

#Define the logistic regression model
def logistic_regression_model(X, y, C_value):
    model = LogisticRegression(class_weight='balanced', C=C_value)
    model.fit(X, y)
    return model

def prepare_counts_data(records, available_eids, pca_components_counts, train_eids, valid_eids, test_eids, demographics_df, agesex):
    """Prepares count-based data for the alternative model."""
    filtered_records = records.query("eid in @available_eids")
    counts_df = (
        filtered_records.pivot_table(index="eid", columns="codes", aggfunc="size", fill_value=0, observed=True)
    )

    # Do not add age and sex if not wanted
    #if agesex:
    #Add age and gender columns to counts_df
    counts_df = counts_df.join(
        demographics_df.set_index('eid')[['age', 'sex']],
        on='eid',
        how='left'
    )
    #else:
    #    print("Did not append demographics information to counts df")

    """counts_df.to_csv("/sc-projects/sc-proj-ukb-cvd/projects/llm2vec/data_ben/Sinusitis/counts_df.csv")"""
    counts_df_train = counts_df.loc[train_eids]
    counts_df_val = counts_df.loc[valid_eids]
    counts_df_test = counts_df.loc[test_eids]
    if pca_components_counts == 0: #in case of no pca:
            return pd.concat([counts_df_train, counts_df_val, counts_df_test]), pd.DataFrame(pd.concat([counts_df_train, counts_df_val, counts_df_test]).index, columns=['eid']) #without pca

    counts_df_train_coo = coo_matrix(counts_df_train.values)
    counts_df_val_coo = coo_matrix(counts_df_val.values)
    counts_df_test_coo = coo_matrix(counts_df_test.values)
    pca = PCA(n_components=pca_components_counts)
    counts_df_pca_train = pca.fit_transform(counts_df_train_coo)
    counts_df_pca_val = pca.transform(counts_df_val_coo)
    counts_df_pca_test = pca.transform(counts_df_test_coo)
    counts_df_pca = np.concatenate([counts_df_pca_train, counts_df_pca_val, counts_df_pca_test])
    return counts_df_pca, pd.DataFrame(pd.concat([counts_df_train, counts_df_val, counts_df_test]).index, columns=['eid']) #with pca


def process_medical_records_llm(times, prepared_records, train_eids, valid_eids, test_eids, pca_components, config, target, q_reps_name):

    # LLM approach
    q_reps_pca, prepared_records_filtered = process_and_encode(prepared_records, pca_components, train_eids, valid_eids, test_eids, q_reps_name)

    X_model = prepare_data(
        q_reps_pca, prepared_records_filtered, times, config["selected_phecode"], train_eids, valid_eids, test_eids
    )

    if(config["lr"]):
        accuracy, auroc_model, auprc_model = train_and_evaluate_model_lr(
            X_model, times, train_eids, valid_eids, test_eids, config, target
        )
    else:
        c_index_model = train_and_evaluate_model(
            X_model, train_eids, valid_eids, cph_penalizer
        )

    return auroc_model, auprc_model


def process_medical_records_counts(times, filtered_records, demographics_df, train_eids, valid_eids, test_eids, pca_components_counts, config, target):

    # Counts-based approach
    counts_df_pca, counts_df = prepare_counts_data(
        filtered_records, np.concatenate([train_eids, valid_eids, test_eids]).tolist(), pca_components_counts, train_eids, valid_eids, test_eids, demographics_df, config["addagesex"]
    )

    X_counts = prepare_data(
        counts_df_pca, counts_df, times, config["selected_phecode"], train_eids, valid_eids, test_eids
    )

    if(config["lr"]):
        accuracy, auroc_counts, auprc_counts, trainpatnum, testpatnum = train_and_evaluate_model_lr_counts(
            X_counts, times, train_eids, valid_eids, test_eids, config, target
        )
    else:
        c_index_counts = train_and_evaluate_model_lr_counts(
            X_counts, train_eids, valid_eids, cph_penalizer
        )

    return auroc_counts, auprc_counts, trainpatnum, testpatnum


def process_medical_records_agesex(times, prepared_records, train_eids, valid_eids, test_eids, pca_components, config, target):
    
    prepared_records_agesex = prepared_records.copy()
    prepared_records_agesex['q_reps'] = prepared_records_agesex['age_sex_tensor']

    # LLM approach
    q_reps_pca, prepared_records_agesex_filtered = process_and_encode(prepared_records_agesex, pca_components, train_eids, valid_eids, test_eids, "q_reps")

    X_agesex = prepare_data(
        q_reps_pca, prepared_records_agesex_filtered, times, config["selected_phecode"], train_eids, valid_eids, test_eids
    )

    if(config["lr"]):
        accuracy, auroc_agesex, auprc_agesex = train_and_evaluate_model_lr(
            X_agesex, times, train_eids, valid_eids, test_eids, config, target
        )
    else:
        c_index_agesex = train_and_evaluate_model(
            X_agesex, train_eids, valid_eids, cph_penalizer
        )

    return auroc_agesex, auprc_agesex


def split_train_val_test_balanced(eids, target, used_eids, num_samples):
    np.random.seed(None)
        
    # Ensure eids is a Series with eid as values
    if isinstance(eids, pd.DataFrame):
        eids = eids['eid']
    elif isinstance(eids, np.ndarray):
        eids = pd.Series(eids, name='eid')

    eids = np.setdiff1d(eids, used_eids)

    # Create a DataFrame with eid as index and target as a column
    combined = pd.DataFrame({'target': target})
    #combined = combined.loc[eids]  # Align with the provided eids
    combined = combined.reindex(eids)
    combined = combined.dropna()  # Remove any rows with NaN values
    #print(combined["target"].index.nunique())

    # Separate positive and negative samples
    positive_eids = combined[combined['target'] == 1].index.values
    negative_eids = combined[combined['target'] == 0].index.values

    # Select positive and negative samples in number of num_samples
    test_pos = np.random.choice(positive_eids, num_samples, replace=False)
    test_neg = np.random.choice(negative_eids, num_samples, replace=False)

    # Combine positive and negative samples for each split
    sample_test_eids = np.concatenate([test_pos, test_neg])

    np.random.shuffle(sample_test_eids)

    return sample_test_eids



def split_train_val_test(eids, config, sample_size, target, test_eids, balanced, onlytest):
    #np.random.seed(config["random_seed"])  # For reproducibility
    np.random.seed(None)
    sample_size_test = config["num_patients_test"]


    #if not onlytest, remove test_eids from eids
    # potentially include again if testsplits should be used
    # if not onlytest:
    #     eids = np.setdiff1d(eids, test_eids)
    #     sample_size_test = min(sample_size, 20000)
    
    # Ensure eids is a Series with eid as values
    if isinstance(eids, pd.DataFrame):
        eids = eids['eid']
    elif isinstance(eids, np.ndarray):
        eids = pd.Series(eids, name='eid')
    
    # Create a DataFrame with eid as index and target as a column
    combined = pd.DataFrame({'target': target})
    #combined = combined.loc[eids]  # Align with the provided eids
    combined = combined.reindex(eids)
    combined = combined.dropna()  # Remove any rows with NaN values
    #print(combined["target"].index.nunique())
    
    # Separate positive and negative samples
    positive_eids = combined[combined['target'] == 1].index.values
    negative_eids = combined[combined['target'] == 0].index.values
    
    # Ensure we have at least one healthy sample
    if len(positive_eids) == 0:
        raise ValueError("No healthy samples in the dataset")

    # Ensure we have at least one diseased sample
    if len(positive_eids) == 0:
        raise ValueError("No diseased samples in the dataset")
    
    # Calculate the number of positive samples needed for each split
    if(not balanced):
        n_train_pos = max(1, int(sample_size * len(positive_eids) / len(combined)))
        n_valid_pos = max(1, int(sample_size * len(positive_eids) / len(combined)))
        if(onlytest):
            n_test_pos = max(1, int(sample_size * len(positive_eids) / len(combined)))
    else:
        n_train_pos = n_valid_pos = sample_size
        if(n_train_pos > len(positive_eids)):
            print("Not enough positive examples")
            return
    
    # Randomly select positive samples for each split
    train_pos = np.random.choice(positive_eids, min(n_train_pos, len(positive_eids)), replace=False)
    # remove eids selected for train split
    remaining_pos = np.setdiff1d(positive_eids, train_pos)
    valid_pos = np.random.choice(remaining_pos, min(n_valid_pos, len(remaining_pos)), replace=False)
    if(onlytest):
        test_pos = np.random.choice(np.setdiff1d(remaining_pos, valid_pos), min(n_test_pos, len(np.setdiff1d(remaining_pos, valid_pos))), replace=False)
    
    # Fill the rest with negative samples
    if(not balanced):
        train_neg = np.random.choice(negative_eids, sample_size - len(train_pos), replace=False)
        remaining_neg = np.setdiff1d(negative_eids, train_neg)
        valid_neg = np.random.choice(remaining_neg, sample_size - len(valid_pos), replace=False)
        if(onlytest):
            test_neg = np.random.choice(np.setdiff1d(remaining_neg, valid_neg), sample_size_test - len(test_pos), replace=False)
    else:
        train_neg = np.random.choice(negative_eids, sample_size, replace=False)
        remaining_neg = np.setdiff1d(negative_eids, train_neg)
        valid_neg = np.random.choice(remaining_neg, sample_size, replace=False)

    
    
    if(not onlytest):
        # Combine positive and negative samples for each split
        sample_train_eids = np.concatenate([train_pos, train_neg])
        sample_valid_eids = np.concatenate([valid_pos, valid_neg])
        
        # Shuffle the combined arrays
        np.random.shuffle(sample_train_eids)
        np.random.shuffle(sample_valid_eids)

    if(onlytest):
        sample_test_eids = np.concatenate([test_pos, test_neg])
        np.random.shuffle(sample_test_eids)
    
    if(onlytest):
        return sample_test_eids
    else:
        return sample_train_eids, sample_valid_eids








class StratifiedShuffleSplitWithMinClass(BaseShuffleSplit):
    """
    Stratified ShuffleSplit cross-validator that ensures each train and test split
    has at least a configurable minimum number of samples from each class.

    Parameters
    ----------
    n_splits : int, default=10
        Number of re-shuffling & splitting iterations.

    test_size : float or int, default=None
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the test split. If int, represents the
        absolute number of test samples. If None, the value is set to the
        complement of the train size. If ``train_size`` is also None, it will
        be set to 0.1.

    train_size : float or int, default=None
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the train split. If
        int, represents the absolute number of train samples. If None,
        the value is automatically set to the complement of the test size.

    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the training and testing indices produced.
        Pass an int for reproducible output across multiple function calls.

    min_train_per_class : int, default=1
        Minimum number of samples per class in the training set.

    min_test_per_class : int, default=1
        Minimum number of samples per class in the testing set.
    """

    def __init__(
        self,
        n_splits=10,
        *,
        test_size=None,
        train_size=None,
        random_state=None,
        min_train_per_class=1,
        min_test_per_class=1,
    ):
        super().__init__(
            n_splits=n_splits,
            test_size=test_size,
            train_size=train_size,
            random_state=random_state,
        )
        self._default_test_size = 0.1
        self.min_train_per_class = min_train_per_class
        self.min_test_per_class = min_test_per_class

    def _iter_indices(self, X, y, groups=None):
        n_samples = _num_samples(X)
        y = check_array(y, ensure_2d=False, dtype=None)
        n_train, n_test = _validate_shuffle_split(  # Use the imported function
            n_samples,
            self.test_size,
            self.train_size,
            default_test_size=self._default_test_size,
        )

        classes, y_indices = np.unique(y, return_inverse=True)
        n_classes = classes.shape[0]

        class_counts = np.bincount(y_indices)
        min_total_per_class = self.min_train_per_class + self.min_test_per_class

        if np.any(class_counts < min_total_per_class):
            raise ValueError(
                "Some classes have too few samples to satisfy the minimum per-class requirements."
            )

        if n_train < n_classes * self.min_train_per_class:
            raise ValueError(
                f"The train_size = {n_train} should be at least "
                f"{n_classes * self.min_train_per_class} to have "
                f"{self.min_train_per_class} samples per class."
            )
        if n_test < n_classes * self.min_test_per_class:
            raise ValueError(
                f"The test_size = {n_test} should be at least "
                f"{n_classes * self.min_test_per_class} to have "
                f"{self.min_test_per_class} samples per class."
            )

        class_indices = np.split(
            np.argsort(y_indices, kind="mergesort"), np.cumsum(class_counts)[:-1]
        )

        rng = check_random_state(self.random_state)

        for _ in range(self.n_splits):
            train_indices = []
            test_indices = []
            n_train_per_class = np.full(n_classes, self.min_train_per_class, dtype=int)
            n_test_per_class = np.full(n_classes, self.min_test_per_class, dtype=int)

            # Allocate remaining samples proportionally
            remaining_train = n_train - n_classes * self.min_train_per_class
            remaining_test = n_test - n_classes * self.min_test_per_class

            class_proportions = class_counts / class_counts.sum()

            # Distribute remaining train samples
            train_allocations = np.floor(remaining_train * class_proportions).astype(int)
            n_train_per_class += train_allocations
            remaining_train -= train_allocations.sum()

            # Distribute any remaining train samples
            for i in np.argsort(-class_proportions):
                if remaining_train <= 0:
                    break
                additional = min(
                    remaining_train, class_counts[i] - n_train_per_class[i] - n_test_per_class[i]
                )
                if additional > 0:
                    n_train_per_class[i] += additional
                    remaining_train -= additional

            # Distribute remaining test samples
            test_allocations = np.floor(remaining_test * class_proportions).astype(int)
            n_test_per_class += test_allocations
            remaining_test -= test_allocations.sum()

            # Distribute any remaining test samples
            for i in np.argsort(-class_proportions):
                if remaining_test <= 0:
                    break
                additional = min(
                    remaining_test, class_counts[i] - n_train_per_class[i] - n_test_per_class[i]
                )
                if additional > 0:
                    n_test_per_class[i] += additional
                    remaining_test -= additional

            # Ensure allocations do not exceed class counts
            for i in range(n_classes):
                total_allocated = n_train_per_class[i] + n_test_per_class[i]
                n_samples_in_class = class_counts[i]
                if total_allocated > n_samples_in_class:
                    excess = total_allocated - n_samples_in_class
                    # Reduce from test first, then from train
                    reduce_test = min(excess, n_test_per_class[i] - self.min_test_per_class)
                    n_test_per_class[i] -= reduce_test
                    excess -= reduce_test
                    reduce_train = min(excess, n_train_per_class[i] - self.min_train_per_class)
                    n_train_per_class[i] -= reduce_train
                    excess -= reduce_train
                    if excess > 0:
                        raise ValueError(
                            f"Not enough samples in class {classes[i]} to allocate "
                            f"requested train and test samples."
                        )

            # Collect indices for train and test
            for i in range(n_classes):
                n_samples_in_class = class_counts[i]
                permutation = rng.permutation(n_samples_in_class)
                permuted_class_indices = class_indices[i][permutation]

                n_train_i = n_train_per_class[i]
                n_test_i = n_test_per_class[i]

                train_class_indices = permuted_class_indices[:n_train_i]
                test_class_indices = permuted_class_indices[n_train_i : n_train_i + n_test_i]

                train_indices.extend(train_class_indices)
                test_indices.extend(test_class_indices)

            rng.shuffle(train_indices)
            rng.shuffle(test_indices)

            yield np.array(train_indices), np.array(test_indices)

    def split(self, X, y, groups=None):
        """Generate indices to split data into training and test set."""
        if groups is not None:
            warnings.warn(
                f"The groups parameter is ignored by {self.__class__.__name__}",
                UserWarning,
            )
        y = check_array(y, ensure_2d=False, dtype=None)
        return super().split(X, y, groups)