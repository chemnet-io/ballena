import os
import pandas as pd
from ast import literal_eval
import numpy as np
import re
from unidecode import unidecode
import logging

# Configure logging
logging.basicConfig(
    filename='evaluation.log',
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Define k_at globally to make it accessible in both main and evaluate_attribute
k_at = {
    'bioActivity': 5,
    'collectionSpecie': 50,
    'collectionSite': 20,
    'collectionType': 1,
    'name': 50  # Assuming k=50 for 'name' based on provided results
}

def normalize(text):
    """Normalize text by lowercasing, removing diacritics, and stripping whitespace."""
    if isinstance(text, dict):
        # Extract the relevant string from the dictionary.
        # Modify the key based on your data structure.
        # Example: {'name': 'Some name', 'details': {...}}
        text = text.get('name', '')
    elif isinstance(text, list):
        # If it's a list, join the elements into a single string
        text = ' '.join(map(str, text))
    elif not isinstance(text, str):
        # Convert other types to string
        text = str(text)
    
    text = unidecode(text.lower()).strip()
    return text

def hits_at(k, true, list_pred, attribute):
    hits = []
    missed_entries = []

    for index_t, t in enumerate(true):
        doi_true, true_val = t
        predictions = list_pred[index_t][1]

        # Normalize the true value
        true_norm = normalize(true_val)

        hit = False
        for lp in predictions[:k]:
            # Normalize the prediction
            pred_norm = normalize(lp)

            # Exact match
            if true_norm == pred_norm:
                hits.append(1)
                hit = True
                break

        if not hit:
            hits.append(0)
            missed_entries.append((index_t, t, predictions[:k]))
            # Log the missed entry
            logging.info(f"Missed Entry - Attribute: {attribute}, Index: {index_t}, True: {t}, Predictions: {predictions[:k]}")

    return np.mean(hits) if hits else 0.0, missed_entries

def mrr_score(true, list_pred, attribute):
    rrs = []
    missed_entries = []
    for index_t, t in enumerate(true):
        doi_true, true_val = t
        predictions = list_pred[index_t][1]

        # Normalize the true value
        true_norm = normalize(true_val)

        hit = False
        for index_lp, lp in enumerate(predictions):
            pred_norm = normalize(lp)

            # Exact match
            if true_norm == pred_norm:
                rrs.append(1 / (index_lp + 1))
                hit = True
                break

        if not hit:
            missed_entries.append((index_t, t, predictions))

    return np.mean(rrs) if rrs else np.nan, missed_entries

def evaluate_attribute(input_csv_path):
    restored_df = pd.read_csv(input_csv_path, quoting=1)  # Use quoting=1 for quote-minimal
    true_parsed = []
    restored_parsed = []
    malformed_indices = []

    for index, row in restored_df.iterrows():
        try:
            true_val = literal_eval(row['true'])
            restored_val = literal_eval(row['restored'])
            true_parsed.append(true_val)
            restored_parsed.append(restored_val)
        except (SyntaxError, ValueError) as e:
            malformed_indices.append(index)
            print(f"Row {index} is malformed and will be skipped. Error: {e}")
            logging.error(f"Row {index} is malformed and will be skipped. Error: {e}")

    if malformed_indices:
        restored_df = restored_df.drop(index=malformed_indices)
        true_parsed = [t for idx, t in enumerate(true_parsed) if idx not in malformed_indices]
        restored_parsed = [r for idx, r in enumerate(restored_parsed) if idx not in malformed_indices]
        print(f"Skipped {len(malformed_indices)} malformed rows.")

    true_values = true_parsed
    predicted_values = restored_parsed

    # Extract attribute and split information from the filename
    attribute_match = re.search(r'doi_(\w+)_0_(\d+)(st|nd|rd|th)\.csv', input_csv_path)
    if attribute_match:
        attribute_name = attribute_match.group(1)
        split_number = attribute_match.group(2)
        split_suffix = attribute_match.group(3)
        split_name = f"{split_number}{split_suffix}"

        # Handle attributes that do not require evaluation (e.g., 'name' might require)
        # Assuming 'name' does require evaluation based on user-provided results
        # If 'name' should be skipped, uncomment the following lines:
        # if attribute_name == 'name':
        #     print(f"Attribute '{attribute_name}' does not require evaluation.")
        #     return None

        k = k_at.get(attribute_name)
        if k:
            mean_hits, missed_hits = hits_at(k, true_values, predicted_values, attribute_name)
            mean_mrr, missed_mrr = mrr_score(true_values, predicted_values, attribute_name)

            return {
                'Attribute': attribute_name,
                'Split': split_name,
                'Hits@k': mean_hits,
                'MRR': mean_mrr
            }

    return None

def main():
    input_dir = 'old/results/LLM_simple'
    splits = ['1st', '2nd', '3rd', '4th']
    attributes = ['bioActivity', 'collectionSpecie', 'collectionSite', 'collectionType', 'name']
    results = []

    # Iterate through each attribute and each split
    for attribute in attributes:
        for split in splits:
            filename_pattern = f'llm_results_gpt4_0.8_doi_{attribute}_0_{split}.csv'
            input_path = os.path.join(input_dir, filename_pattern)
            if os.path.isfile(input_path):
                print(f"Evaluating {filename_pattern}:")
                result = evaluate_attribute(input_path)
                if result:
                    hits_k = f"Hits@{k_at.get(result['Attribute'], 'N/A')}: {result['Hits@k']:.4f}"
                    mrr = f"MRR: {result['MRR']:.4f}"
                    print(f"{hits_k}")
                    print(f"{mrr}")
                    results.append(result)
                print("-" * 40)
            else:
                print(f"File {filename_pattern} does not exist. Skipping.")
                print("-" * 40)

    # Create a DataFrame from the results
    if results:
        results_df = pd.DataFrame(results)
        # Pivot the table for better readability
        pivot_table = results_df.pivot(index='Attribute', columns='Split', values=['Hits@k', 'MRR'])
        # Flatten the MultiIndex columns
        pivot_table.columns = [' '.join(col).strip() for col in pivot_table.columns.values]
        pivot_table = pivot_table.reset_index()

        # Reorder columns for clarity
        cols = ['Attribute'] + [f"{metric} {split}" for metric in ['Hits@k', 'MRR'] for split in splits]
        pivot_table = pivot_table.reindex(columns=cols)

        # Display the table in markdown format
        print("\n```markdown")
        print(pivot_table.to_markdown(index=False))
        print("```")
    else:
        print("No evaluation results to display.")

if __name__ == "__main__":
    main()