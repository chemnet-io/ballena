import os
import pandas as pd
from ast import literal_eval
import numpy as np

def hits_at(k, true, list_pred):
    """
    Calculate the Hits@k metric.

    Parameters:
    - k (int): The cutoff rank.
    - true (list of tuples): List of true values in the format [(doi, true_value), ...].
    - list_pred (list of lists): List of predicted values in the format [[doi, [pred1, pred2, ...]], ...].

    Returns:
    - float: The average Hits@k score.
    - list: List of missed entries for debugging.
    """
    hits = []
    missed_entries = []

    for index_t, t in enumerate(true):
        hit = False
        doi_true, true_val = t
        predictions = list_pred[index_t][1]

        for index_lp, lp in enumerate(predictions[:k]):
            if true_val == lp:
                hits.append(1)
                hit = True
                break
        if not hit:
            hits.append(0)
            missed_entries.append((index_t, t, predictions[:k]))
    return np.mean(hits), missed_entries

def mrr(true, list_pred):
    """
    Calculate the Mean Reciprocal Rank (MRR) metric.

    Parameters:
    - true (list of tuples): List of true values in the format [(doi, true_value), ...].
    - list_pred (list of lists): List of predicted values in the format [[doi, [pred1, pred2, ...]], ...].

    Returns:
    - float: The average MRR score.
    - list: List of missed entries for debugging.
    """
    rrs = []
    missed_entries = []
    for index_t, t in enumerate(true):
        hit = False
        doi_true, true_val = t
        predictions = list_pred[index_t][1]

        for index_lp, lp in enumerate(predictions):
            if true_val == lp:
                rrs.append(1 / (index_lp + 1))
                hit = True
                break
        if not hit:
            missed_entries.append((index_t, t, predictions))
    return np.mean(rrs), missed_entries

def evaluate_attribute(input_csv_path, output_dir, attribute_name):
    # Evaluation parameters
    k_at = [1,2,3,4,5]

    # Initialize metrics storage
    hitsatk_df = {'k': [], 'metric': [], 'value': []}
    missed_hits = []

    mrr_df = {'metric': [], 'value': []}
    missed_mrr = []

    # Load the CSV file
    try:
        restored_df = pd.read_csv(input_csv_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"The file {input_csv_path} does not exist. Please check the path.")

    # Ensure 'true' and 'restored' columns exist
    if 'true' not in restored_df.columns or 'restored' not in restored_df.columns:
        raise ValueError("The input CSV must contain 'true' and 'restored' columns.")

    # Apply literal_eval to parse string representations of lists/tuples
    restored_df['true'] = restored_df['true'].apply(literal_eval)
    restored_df['restored'] = restored_df['restored'].apply(literal_eval)

    # Extract true and predicted lists
    true_values = restored_df['true'].to_list()  # List of tuples: [(doi, true_value), ...]
    predicted_values = restored_df['restored'].to_list()  # List of lists: [[doi, [pred1, pred2, ...]], ...]

    # Calculate Hits@k
    for k in k_at:
        mean_hits, missed = hits_at(k, true_values, predicted_values)
        hitsatk_df['k'].append(k)
        hitsatk_df['metric'].append('hits@k')
        hitsatk_df['value'].append(mean_hits)
        missed_hits.extend(missed)

    # Calculate MRR
    mean_mrr, missed_m = mrr(true_values, predicted_values)
    mrr_df['metric'].append('mrr')
    mrr_df['value'].append(mean_mrr)
    missed_mrr.extend(missed_m)

    # Convert metrics to DataFrames
    hitsatk_df = pd.DataFrame(hitsatk_df)
    mrr_df = pd.DataFrame(mrr_df)

    # Save Hits@k results
    hitsatk_output_path = os.path.join(output_dir, f'hits@k_{attribute_name}_0_1st.csv')
    hitsatk_df.to_csv(hitsatk_output_path, index=False)
    print(f"Hits@k results saved to {hitsatk_output_path}")

    # Save MRR results
    mrr_output_path = os.path.join(output_dir, f'mrr_{attribute_name}_0_1st.csv')
    mrr_df.to_csv(mrr_output_path, index=False)
    print(f"MRR results saved to {mrr_output_path}")

    # Save missed hits entries for debugging
    missed_hits_df = pd.DataFrame(missed_hits, columns=['index', 'true', 'predictions'])
    missed_hits_output_path = os.path.join(output_dir, f'missed_hits@k_{attribute_name}_0_1st.csv')
    missed_hits_df.to_csv(missed_hits_output_path, index=False)
    print(f"Missed Hits@k entries saved to {missed_hits_output_path}")

    # Save missed MRR entries for debugging
    missed_mrr_df = pd.DataFrame(missed_mrr, columns=['index', 'true', 'predictions'])
    missed_mrr_output_path = os.path.join(output_dir, f'missed_mrr_{attribute_name}_0_1st.csv')
    missed_mrr_df.to_csv(missed_mrr_output_path, index=False)
    print(f"Missed MRR entries saved to {missed_mrr_output_path}")

# Define paths
output_dir = 'evaluation_results'  # Changed output directory for clarity
os.makedirs(output_dir, exist_ok=True)

# Evaluate bioActivity attribute
bioActivity_csv_path = 'llm_ft_results/llm_results_ft_4o_0.8_doi_bioActivity_0_1st.csv'
evaluate_attribute(bioActivity_csv_path, output_dir, 'bioActivity')
