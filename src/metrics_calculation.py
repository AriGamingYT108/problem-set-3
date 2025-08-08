'''
PART 2: METRICS CALCULATION
- Tailor the code scaffolding below to calculate various metrics
- Write the functions below
    - Further info and hints are provided in the docstrings
    - These should return values when called by the main.py
'''
#This Version
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
import ast

def calculate_metrics(model_pred_df, genre_list, genre_true_counts, genre_tp_counts, genre_fp_counts):
    '''
    Calculate micro and macro metrics
    
    Args:
        model_pred_df (pd.DataFrame): DataFrame containing model predictions
        genre_list (list): List of unique genres
        genre_true_counts (dict): Dictionary of true genre counts
        genre_tp_counts (dict): Dictionary of true positive genre counts
        genre_fp_counts (dict): Dictionary of false positive genre counts
    
    Returns:
        tuple: Micro precision, recall, F1 score
        lists of macro precision, recall, and F1 scores
    
    Hint #1: 
    tp -> true positives
    fp -> false positives
    tn -> true negatives
    fn -> false negatives

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    Hint #2: Micro metrics are tuples, macro metrics are lists

    '''

    # Your code here

    # 1) Compute false negatives per genre
    genre_fn_counts = {
        g: genre_true_counts[g] - genre_tp_counts.get(g, 0)
        for g in genre_list
    }

    # 2) Build per-genre metrics
    macro_precision = []
    macro_recall    = []
    macro_f1        = []

    for g in genre_list:
        tp = genre_tp_counts.get(g, 0)
        fp = genre_fp_counts.get(g, 0)
        fn = genre_fn_counts.get(g, 0)

        # avoid division by zero
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

        macro_precision.append(p)
        macro_recall.append(r)
        macro_f1.append(f)

    # 3) Micro-averaged metrics
    total_tp = sum(genre_tp_counts.values())
    total_fp = sum(genre_fp_counts.values())
    total_fn = sum(genre_fn_counts.values())

    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f = (2 * micro_p * micro_r / (micro_p + micro_r)) if (micro_p + micro_r) > 0 else 0.0

    return micro_p, micro_r, micro_f, macro_precision, macro_recall, macro_f1
    

    
def calculate_sklearn_metrics(model_pred_df, genre_list):
    '''
    Calculate metrics using sklearn's precision_recall_fscore_support.
    
    Args:
        model_pred_df (pd.DataFrame): DataFrame containing model predictions.
        genre_list (list): List of unique genres.
    
    Returns:
        tuple: Macro precision, recall, F1 score, and micro precision, recall, F1 score.
    
    Hint #1: You'll need these two lists
    pred_rows = []
    true_rows = []
    
    Hint #2: And a little later you'll need these two matrixes for sk-learn
    pred_matrix = pd.DataFrame(pred_rows)
    true_matrix = pd.DataFrame(true_rows)
    '''

    # Your code here

    pred_rows = []
    true_rows = []

    # parse the actual list literal once
    model_pred_df['true_list'] = model_pred_df['actual genres'].apply(ast.literal_eval)

    for _, row in model_pred_df.iterrows():
        actual = set(row['true_list'])
        predicted = row['predicted']

        # build a 0/1 vector per genre
        true_vec = [1 if g in actual    else 0 for g in genre_list]
        pred_vec = [1 if g == predicted else 0 for g in genre_list]

        true_rows.append(true_vec)
        pred_rows.append(pred_vec)

    true_matrix = pd.DataFrame(true_rows, columns=genre_list)
    pred_matrix = pd.DataFrame(pred_rows, columns=genre_list)

    # sklearn returns (p_vec, r_vec, f_vec, support_vec)
    # average=None → per-class; average='macro' or 'micro' → scalars
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
        true_matrix, pred_matrix, average='macro', zero_division=0
    )
    p_micro, r_micro, f_micro, _ = precision_recall_fscore_support(
        true_matrix, pred_matrix, average='micro', zero_division=0
    )

    return p_macro, r_macro, f_macro, p_micro, r_micro, f_micro