import os
import pandas as pd
import json
from tqdm import tqdm
import csv

# Define the directory paths
splits_dir = 'splits'
extracted_text_path = os.path.join('pypdfextraction', 'extracted_text.parquet')
output_dir = 'TEST_results'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Define the attributes and their corresponding JSON keys
attributes_info = {
    'name': {'json_key': 'name'},
    'bioActivity': {'json_key': 'bioActivity'},
    'collectionSpecie': {'json_key': 'collectionSpecie'},
    'collectionSite': {'json_key': 'collectionSite'},
    'collectionType': {'json_key': 'collectionType'}
}

# Load the extracted texts
extracted_text_df = pd.read_parquet(extracted_text_path)

# Preprocess filenames to extract DOIs
extracted_text_df['doi'] = extracted_text_df['filename'].str.replace('.pdf', '').str.replace('@', '/')

# Dummy function to simulate API call
def dummy_api_call(doi):
    return json.dumps([{attributes_info[attribute]['json_key']: f"Dummy {attribute} for {doi}"} for attribute in attributes_info])

# Iterate over each attribute
for attribute, info in attributes_info.items():
    print(f"Processing attribute: {attribute}")
    
    # Define the test split file path for iteration 0 and 1st split
    test_split_filename = f'test_doi_{attribute}_0_1st.csv'
    test_split_path = os.path.join(splits_dir, test_split_filename)
    
    if not os.path.exists(test_split_path):
        print(f"Test split file {test_split_filename} does not exist. Skipping attribute: {attribute}")
        continue
    
    # Load the test split CSV
    test_split_df = pd.read_csv(test_split_path)
    
    # Extract the DOIs from the test split
    test_dois = test_split_df['node'].unique()
    
    # Filter the extracted texts for the test DOIs
    test_texts_df = extracted_text_df[extracted_text_df['doi'].isin(test_dois)]
    
    # Merge the true values with the texts based on DOI
    merged_df = pd.merge(test_texts_df, test_split_df, left_on='doi', right_on='node', how='inner')
    
    # Prepare lists to store results
    true_values = []
    restored_values = []
    edge_types = []
    
    # Iterate over each row and simulate API call
    for _, row in tqdm(merged_df.iterrows(), total=merged_df.shape[0], desc=f"Attribute: {attribute}"):
        doi = row['doi']
        true_value = row['neighbor']
        edge_type = row['type']
        
        # Simulate API call
        assistant_reply = dummy_api_call(doi)
        
        # Parse the simulated reply
        try:
            restored_data = json.loads(assistant_reply)
            # Extract the values for the attribute
            restored_values_list = [item.get(info['json_key'], '') for item in restored_data]
            # Ensure all extracted values are strings
            restored_values_list = [str(value) for value in restored_values_list if value]
            # Convert list to string representation with single quotes
            restored_values_str = "[" + ", ".join(f"'{value}'" for value in restored_values_list) + "]"
        except Exception as e:
            print(f"Unexpected error for DOI {doi}: {e}")
            restored_values_str = "[]"
        
        # Assemble the data into the desired format
        true_entry = f"['{doi}', '{true_value}']"
        restored_entry = f"['{doi}', {restored_values_str}]"
        
        true_values.append(true_entry)
        restored_values.append(restored_entry)
        edge_types.append(edge_type)
    
    # Define the output filename
    output_filename = f'dummy_llm_results_gpt4_0.8_doi_{attribute}_0_1st.csv'
    output_path = os.path.join(output_dir, output_filename)
    
    # Save the results to CSV with proper quoting and without quotes for column names and edge_type
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['true', 'restored', 'edge_type'])  # Write header without quotes
        for true, restored, edge_type in zip(true_values, restored_values, edge_types):
            writer.writerow([true, restored, edge_type])  # Write edge_type without quotes
    
    print(f"Dummy results saved to {output_path}\n")

print("Processing complete for all attributes.")