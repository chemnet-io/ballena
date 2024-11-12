import os
import pandas as pd
from ast import literal_eval
import numpy as np
import re
from unidecode import unidecode
import logging
from fuzzywuzzy import fuzz

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
    'collectionType': 1
}

def normalize(text):
    """Normalize text by lowercasing, removing diacritics, and removing non-alphanumeric characters."""
    if isinstance(text, dict):
        # Extract the relevant string from the dictionary.
        # Modify the key based on your data structure.
        # Example: {'name': 'Some bio activity', 'details': {...}}
        text = text.get('name', '')
    elif isinstance(text, list):
        # If it's a list, join the elements into a single string
        text = ' '.join(map(str, text))
    elif not isinstance(text, str):
        # Convert other types to string
        text = str(text)
    
    text = unidecode(text.lower())
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_city_state(address):
    """Extract city and state from the full address."""
    # Example address: 'Parque Estadual Carlos Botelho, São Miguel Arcanjo, São Paulo, Brazil'
    parts = address.split(',')
    if len(parts) >= 3:
        city = parts[-3].strip()
        state = parts[-2].strip().split('/')[-1] if '/' in parts[-2] else parts[-2].strip()
        return f"{city}/{state}"
    return ""

def hits_at(k, true, list_pred, attribute):
    hits = []
    missed_entries = []

    for index_t, t in enumerate(true):
        hit = False
        doi_true, true_val = t
        predictions = list_pred[index_t][1]

        # Normalize the true value
        true_norm = normalize(true_val)

        # Specialized handling for 'collectionSite'
        if attribute == 'collectionSite':
            true_components = true_val.split('/')
            if len(true_components) == 2:
                true_city = normalize(true_components[0])
                true_state = normalize(true_components[1])
            else:
                true_city = normalize(true_val)
                true_state = ""

        for lp in predictions[:k]:
            # Normalize the prediction
            pred_norm = normalize(lp)

            if attribute == 'collectionSite':
                # Extract city/state from prediction
                extracted_pred = extract_city_state(lp)
                pred_components = extracted_pred.split('/')
                if len(pred_components) == 2:
                    pred_city = normalize(pred_components[0])
                    pred_state = normalize(pred_components[1])
                else:
                    pred_city = normalize(extracted_pred)
                    pred_state = ""

                # Check if both city and state match
                if (true_city in pred_city) and (true_state in pred_state):
                    hits.append(1)
                    hit = True
                    break
            elif attribute == 'bioActivity':
                # For bioActivity, check if true_val is a substring in prediction
                if true_norm in pred_norm:
                    hits.append(1)
                    hit = True
                    break
            else:
                # Default exact match
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
        hit = False
        doi_true, true_val = t
        predictions = list_pred[index_t][1]

        # Normalize the true value
        true_norm = normalize(true_val)

        if attribute == 'collectionSite':
            true_components = true_val.split('/')
            if len(true_components) == 2:
                true_city = normalize(true_components[0])
                true_state = normalize(true_components[1])
            else:
                true_city = normalize(true_val)
                true_state = ""

        for index_lp, lp in enumerate(predictions):
            pred_norm = normalize(lp)

            if attribute == 'collectionSite':
                extracted_pred = extract_city_state(lp)
                pred_components = extracted_pred.split('/')
                if len(pred_components) == 2:
                    pred_city = normalize(pred_components[0])
                    pred_state = normalize(pred_components[1])
                else:
                    pred_city = normalize(extracted_pred)
                    pred_state = ""

                if (true_city in pred_city) and (true_state in pred_state):
                    rrs.append(1 / (index_lp + 1))
                    hit = True
                    break
            elif attribute == 'bioActivity':
                if true_norm in pred_norm:
                    rrs.append(1 / (index_lp + 1))
                    hit = True
                    break
            else:
                if true_norm == pred_norm:
                    rrs.append(1 / (index_lp + 1))
                    hit = True
                    break

        if not hit:
            missed_entries.append((index_t, t, predictions))

    return np.mean(rrs) if rrs else np.nan, missed_entries

def hits_at_fuzzy(k, true, list_pred, attribute, threshold=90):
    hits = []
    missed_entries = []

    for index_t, t in enumerate(true):
        hit = False
        doi_true, true_val = t
        predictions = list_pred[index_t][1]

        # Normalize the true value
        true_norm = normalize(true_val)

        for lp in predictions[:k]:
            pred_norm = normalize(lp)
            similarity = fuzz.partial_ratio(true_norm, pred_norm)

            if similarity >= threshold:
                hits.append(1)
                hit = True
                break

        if not hit:
            hits.append(0)
            missed_entries.append((index_t, t, predictions[:k]))
            # Log the missed entry
            logging.info(f"Missed Entry (Fuzzy) - Attribute: {attribute}, Index: {index_t}, True: {t}, Predictions: {predictions[:k]}")

    return np.mean(hits) if hits else 0.0, missed_entries

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

        # Handle attributes that do not require evaluation (e.g., 'name')
        if attribute_name == 'name':
            print(f"Attribute '{attribute_name}' does not require evaluation.")
            return None

        k = k_at.get(attribute_name)
        if k:
            if attribute_name in ['bioActivity', 'collectionSite']:
                # Use exact matching logic
                mean_hits, missed_hits = hits_at(k, true_values, predicted_values, attribute_name)
            else:
                # Use exact matching logic for other attributes
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

        # Display the table in markdown format
        print("\n```markdown")
        print(pivot_table.to_markdown(index=False))
        print("```")
    else:
        print("No evaluation results to display.")

if __name__ == "__main__":
    main()