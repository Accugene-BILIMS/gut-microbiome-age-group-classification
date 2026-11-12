"""Shared data loaders, cross-validation splitter and metrics for the classification analyses.

Input
  S1_data_processing/results/discovery_genus_relab_cut0.01.csv   genus relative abundance (%) of 3,659 participants
  S1_data_processing/results/discovery_covariates.csv            Diet, Drug, Sex, Disease_pr per participant
  S1_data_processing/results/split_sample_ids.csv                training set (n = 420) and hold-out set (n = 60)
  S1_data_processing/results/external_genus_relab_cut0.01.csv    external cohort genus relative abundance (%)
  metadata/metadata_external.csv                                 Age_group of the external cohort

Process
  - Features: the 783 genus columns (names starting with 'd__'); 'Unassigned' is not a genus and is excluded.
  - Labels: 10s = 0, 20s_40s = 1, 50s_plus = 2.
  - Cross-validation: repeated stratified five-fold, 10 repeats (50 splits), random_state 42.
  - Metrics for a three-class probability matrix: macro one-vs-rest AUC, accuracy, macro sensitivity,
    macro specificity, macro F1-score, and one-vs-rest AUC per age group.
"""
import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

PKG = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
DATA = os.path.join(PKG, 'S1_data_processing', 'results')
GENUS_TABLE = os.path.join(DATA, 'discovery_genus_relab_cut0.01.csv')
COVARIATES = os.path.join(DATA, 'discovery_covariates.csv')
SPLIT_IDS = os.path.join(DATA, 'split_sample_ids.csv')
EXTERNAL_TABLE = os.path.join(DATA, 'external_genus_relab_cut0.01.csv')
EXTERNAL_META = os.path.join(PKG, 'metadata', 'metadata_external.csv')

TARGET = 'Age_group3'
CLASSES = ['10s', '20s_40s', '50s_plus']
COVARIATE_COLS = ['Sex', 'Diet', 'Drug', 'Disease_pr']

SEED = 42
N_SPLITS = 5
N_REPEATS = 10


def encode(labels):
    return np.array([CLASSES.index(v) for v in labels])


def load_genus():
    """Genus table indexed by Sample_name, and the list of 783 genus columns."""
    df = pd.read_csv(GENUS_TABLE, index_col='Sample_name')
    features = [c for c in df.columns if c.startswith('d__')]
    return df, features


def load_split(split):
    """split: 'train' or 'holdout'. Returns X (participants x genera), y (labels), sample ids."""
    df, features = load_genus()
    ids = pd.read_csv(SPLIT_IDS)
    ids = ids[ids['split'] == split]
    sub = df.loc[ids['Sample_name']]
    return sub[features], encode(sub[TARGET].values), ids['Sample_name'].tolist()


def load_covariates(sample_ids):
    cov = pd.read_csv(COVARIATES, index_col='Sample_name')
    return cov.loc[sample_ids, COVARIATE_COLS]


def load_external(features):
    """External cohort in the given genus order (genera not detected = 0) and its labels."""
    X = pd.read_csv(EXTERNAL_TABLE, index_col=0).reindex(columns=features, fill_value=0)
    meta = pd.read_csv(EXTERNAL_META, index_col='Sample_name').loc[X.index]
    return X, encode(meta['Age_group'].values)


def cv_splitter():
    return RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)


def specificity_macro(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASSES)))
    specs = []
    for i in range(len(CLASSES)):
        tp = cm[i, i]
        fn = cm[i].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        specs.append(tn / (tn + fp) if tn + fp else 0.0)
    return float(np.mean(specs))


def classification_metrics(y_true, prob):
    """Metrics for a three-class probability matrix (columns in CLASSES order); the predicted class is the argmax."""
    pred = prob.argmax(axis=1)
    out = {
        'auc_macro': roc_auc_score(y_true, prob, multi_class='ovr', average='macro'),
        'accuracy': accuracy_score(y_true, pred),
        'sensitivity_macro': recall_score(y_true, pred, average='macro'),
        'specificity_macro': specificity_macro(y_true, pred),
        'f1_macro': f1_score(y_true, pred, average='macro'),
    }
    for c, name in enumerate(CLASSES):
        out[f'auc_{name}'] = roc_auc_score(y_true == c, prob[:, c])
    return out


def summarize(df, group_cols, metric_cols):
    """Mean and SD of each metric across the cross-validation splits."""
    g = df.groupby(group_cols)[metric_cols]
    out = pd.concat({'mean': g.mean(), 'sd': g.std()}, axis=1)
    out.columns = [f'{m}_{stat}' for stat, m in out.columns]
    return out.round(4).reset_index()
