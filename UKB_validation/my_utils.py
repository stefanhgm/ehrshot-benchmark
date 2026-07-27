from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
import ast


## file containing patient splits (built from UKB_PROJECT_DIR in .env; see filepaths.py)
from filepaths import PATH_TO_SPLIT_CSV as path_to_splits



## hyperparameter grids for tuning
XGB_PARAMS = {
    'max_depth': [3, 6, -1],
    'learning_rate': [0.02, 0.1, 0.5],
    'num_leaves': [10, 25, 100]
}
LR_PARAMS = {
    "C": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e2, 1e3, 1e4, 1e5, 1e6], 
    "penalty": ['l2']
}


# Few shot settings
SHOT_STRATS = {
    'few' : [1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 128],
    'long' : [-1],
    'all' : [1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 128, -1],
    'debug' : [10],
}


MODELS = ["qwen3", "clmbr", "bioclinicalbert", "qwenclmbrcodes"] #potentially add qwen and llm2vec. Counts is used in addition

TIMEBINS = [0, 30, 180, 365, np.inf]
TIMEBIN_LABELS = ["0-30", "30-180", "180-365", "365_plus"]

# Types of base models to test
MODEL_2_INFO: Dict[str, Dict[str, Any]] = {
# Removed count based model for debugging
    'count' : {
        'label' : 'Count-based',
        # 'heads' : ['gbm', 'lr_lbfgs', 'rf', ],
        'heads' : ['gbm', ],
    },
    'clmbr' : {
        'label' : 'CLMBR-T-Base',
        'heads' : ['lr_lbfgs', ],
    },
    # 'qwen' : {
    #     'label' : 'Qwen2-7B-instruct', # 'LLM',
    #     # 'heads' : ['gbm', ],
    #     'heads' : ['lr_lbfgs', ],
    # },
    'qwen3' : {
        'label' : 'Qwen3-Emb-8B', # 'LLM',
        # 'heads' : ['gbm', ],
        'heads' : ['lr_lbfgs', ],
    },
    # 'llm2vec' : {
    #     'label' : 'LLM2Vec-Llama-3.1-8B',
    #     'heads' : ['lr_lbfgs', ],
    # },
    'bioclinicalbert' : {
        'label' : 'BioClinicalBERT', # 'BioClinicalBERT',
        'heads' : ['lr_lbfgs', ],
    },
    'qwenclmbrcodes' : {
        'label' : 'Qwen3-Emb-8B CLMBR-T-Base codes', # 'Qwen3-Emb-8B + CLMBR Codes',
        'heads' : ['lr_lbfgs', ],
    }
}


TASK_GROUP_2_LABELING_FUNCTION = {
    "Mortality Prediction": [
        "death"
    ],
    "Operational Outcomes": [
        "hospitalization"
    ],
    "Assignment of New Diagnoses": [
    "Pneumonia",
    "Myocardial infarction [Heart attack]",
    "Rheumatoid arthritis",
    "Back pain",
    "Chronic obstructive pulmonary disease [COPD]",
    "Cerebral infarction [Ischemic stroke]",
    "Pulmonary embolism",
    "Diabetes mellitus",
    "Chronic kidney disease",
    "Suicide ideation and attempt or self harm",
    "Endocarditis",
    "Mitral valve insufficiency",
    "Abdominal aortic aneurysm",
    "Psoriasis",
    "Ischemic heart disease",
    "Parkinson's disease (Primary)",
    "Rheumatic fever and chronic rheumatic heart diseases",
    "Aortic stenosis",
    "Atrial fibrillation",
    "Cardiac arrest",
    "Anemia",
    "Hypertension",
    "Heart failure",
    ]
}

TASK_GROUP_2_PAPER_NAME = {
    "Mortality Prediction": "Mortality Prediction",
    "Operational Outcomes": "Operational Outcomes",
    "Assignment of New Diagnoses": "Assignment of New Diagnoses",
    "mean_all_task_groups": "Mean across all task groups",
    "UKB": "UK Biobank",
    "EHRSHOT": "EHRSHOT"
}

LABELING_FUNCTION_2_PAPER_NAME = {
    "hospitalization" : "phecode_9201",
    "death" : "phecode_4306655",
    "Pneumonia": "phecode_RE_468",
    "Myocardial infarction [Heart attack]": "phecode_CV_404.1",
    "Rheumatoid arthritis": "phecode_MS_705.1",
    "Back pain": "phecode_MS_718",
    "Chronic obstructive pulmonary disease [COPD]": "phecode_RE_474",
    "Cerebral infarction [Ischemic stroke]": "phecode_CV_431.11",
    "Pulmonary embolism": "phecode_CV_440.3",
    "Diabetes mellitus": "phecode_EM_202",
    "Chronic kidney disease": "phecode_GU_582.2",
    "Suicide ideation and attempt or self harm": "phecode_MB_284",
    "Endocarditis": "phecode_CV_410.2",
    "Mitral valve insufficiency": "phecode_CV_413.11",
    "Abdominal aortic aneurysm": "phecode_CV_438.11",
    "Psoriasis": "phecode_DE_664.4",
    "Ischemic heart disease": "phecode_CV_404",
    "Parkinson's disease (Primary)": "phecode_NS_324.11",
    "Rheumatic fever and chronic rheumatic heart diseases": "phecode_CV_400",
    "Aortic stenosis": "phecode_CV_413.21",
    "Atrial fibrillation": "phecode_CV_416.21",
    "Cardiac arrest": "phecode_CV_420",
    "Anemia": "phecode_BI_164",
    "Hypertension": "phecode_CV_401",
    "Heart failure": "phecode_CV_424",
}


HEAD_2_INFO: Dict[str, Dict[str, str]] = {
    'gbm' : {
        'label' : 'GBM',
    },
    'lr_lbfgs' : {
        'label' : 'LR',
    },
    'lr_newton-cg' : {
        'label' : 'LR',
    },
    'protonet' : {
        'label' : 'ProtoNet',
    },
    'rf' : {
        'label' : 'Random Forest',
    },
}



# Plotting
SCORE_MODEL_HEAD_2_COLOR = {
    'auroc' : {
        'clmbr' : {
            'lr_lbfgs' : 'tab:orange',
        },
        'count' : {
            'gbm' : 'tab:green',
            'lr_lbfgs' : 'tab:green',
            'rf' : 'tab:orange',
        },
        # 'qwen' : {
        #     'lr_lbfgs' : 'tab:purple',
        # },
        'qwen3' : {
            'lr_lbfgs' : 'tab:blue',
        },
        # 'llm2vec' : {
        #     'lr_lbfgs' : 'tab:green',
        # },
        'agr' : {
            'lr_lbfgs' : 'tab:brown',
        },
        'bioclinicalbert' : {
            'lr_lbfgs' : 'tab:purple',
        },
        'qwenclmbrcodes' : {
            'lr_lbfgs' : 'tab:purple',
        }
    },
    'auprc' : {
        'clmbr' : {
            'lr_lbfgs' : 'tab:orange',
        },
        'count' : {
            'gbm' : 'tab:green',
            'lr_lbfgs' : 'tab:green',
            'rf' : 'tab:orange',
        },
        'llm2vec' : {
            'lr_lbfgs' : 'tab:orange',
        },
        'qwen' : {
            'lr_lbfgs' : 'tab:purple',
        },
        'qwen3' : {
            'lr_lbfgs' : 'tab:blue',
        },
        'agr' : {
            'lr_lbfgs' : 'tab:brown',
        },
        'bioclinicalbert' : {
            'lr_lbfgs' : 'tab:purple',
        },
        'qwenclmbrcodes' : {
            'lr_lbfgs' : 'tab:purple',
        }
    },
    'brier' : {
        'clmbr' : {
            'lr_lbfgs' : 'tab:orange',
        },
        'count' : {
            'gbm' : 'tab:green',
            'lr_lbfgs' : 'tab:green',
            'rf' : 'tab:orange',
        },
        'llm2vec' : {
            'lr_lbfgs' : 'tab:orange',
        },
        'qwen' : {
            'lr_lbfgs' : 'tab:purple',
        },
        'qwen3' : {
            'lr_lbfgs' : 'tab:blue',
        },
        'agr' : {
            'lr_lbfgs' : 'tab:brown',
        },
        'bioclinicalbert' : {
            'lr_lbfgs' : 'tab:purple',
        },
        'qwenclmbrcodes' : {
            'lr_lbfgs' : 'tab:purple',
        }
    },
}



def filter_df(df: pd.DataFrame, 
            score: Optional[str] = None, 
            labeling_function: Optional[str] = None, 
            task_group: Optional[str] = None,
            sub_tasks: Optional[List[str]] = None,
            model_heads: Optional[List[Tuple[str, str]]] = None) -> pd.DataFrame:
    """Filters results df based on various criteria."""
    df = df.copy()
    if score:
        df = df[df['score'] == score]
    if labeling_function:
        df = df[df['labeling_function'] == labeling_function]
    if task_group:
        labeling_functions: List[str] = TASK_GROUP_2_LABELING_FUNCTION[task_group]
        df = df[df['labeling_function'].isin(labeling_functions)]
    if sub_tasks:
        df = df[df['sub_task'].isin(sub_tasks)]
    if model_heads:
        mask = [ False ] * df.shape[0]
        for model_head in model_heads:
            mask = mask | ((df['model'] == model_head[0]) & (df['head'] == model_head[1]))
        df = df[mask]
    return df



def type_tuple_list(s):
    """For parsing List[Tuple] from command line using `argparse`"""
    try:
        # Convert the string representation of list of tuples into actual list of tuples
        val = ast.literal_eval(s)
        if not isinstance(val, list):
            raise ValueError("Argument should be a list of tuples")
        for item in val:
            if not isinstance(item, tuple) or not all(isinstance(i, str) for i in item):
                raise ValueError("Argument items should be tuples of strings")
        return val
    except ValueError:
        raise ValueError("Argument should be a list of tuples of strings")