'''
PART 1: PRE-PROCESSING
- Tailor the code scaffolding below to load and process the data
- Write the functions below
    - Further info and hints are provided in the docstrings
    - These should return values when called by the main.py
'''
import numpy as np
import pandas as pd

def load_data():
    '''
    Load data from CSV files
    
    Returns:
        model_pred_df (pd.DataFrame): DataFrame containing model predictions
        genres_df (pd.DataFrame): DataFrame containing genre information
    '''
    # Your code here

    model_pred_df = pd.read_csv('data/prediction_model_03.csv')
    genres_df = pd.read_csv('data/genres.csv')

    return model_pred_df, genres_df


def process_data(model_pred_df, genres_df):
    '''
    Process data to get genre lists and count dictionaries
    
    Returns:
        genre_list (list): List of unique genres
        genre_true_counts (dict): Dictionary of true genre counts
        genre_tp_counts (dict): Dictionary of true positive genre counts
        genre_fp_counts (dict): Dictionary of false positive genre counts
    '''

    # Your code here

    # 2. Build the master list of genres (from the true labels DataFrame)
    genre_list = genres_df['genre'].tolist()

    # 3. Count how often each genre truly appears
    import ast
    model_pred_df['true_list'] = model_pred_df['actual genres'].apply(ast.literal_eval)
    exploded = model_pred_df.explode('true_list')
    genre_true_counts = exploded['true_list'].value_counts().to_dict()

    # 4. Count true positives: predicted genre, and correct? == 0
    genre_tp_counts = (model_pred_df[model_pred_df['correct?'] == 0]['predicted'].value_counts().to_dict())

    # 5. Count false positives: predicted genre, but correct? == 1
    genre_fp_counts = (model_pred_df[model_pred_df['correct?'] == 1]['predicted'].value_counts().to_dict())

    return genre_list, genre_true_counts, genre_tp_counts, genre_fp_counts