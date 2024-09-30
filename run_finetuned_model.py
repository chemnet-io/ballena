import os
import pandas as pd
import openai
import json
import csv
from tqdm import tqdm

# Set your OpenAI API key
openai.api_key = 'YOUR_OPENAI_API_KEY'  # Replace with your actual API key

# Define the directory paths
splits_dir = 'splits'
extracted_text_path = os.path.join('pypdfextraction', 'extracted_text.parquet')
output_dir = 'llm_results'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Define the attributes and their corresponding system prompts
attributes_info = {
    'name': {
        'json_key': 'name',
        'prompt': (
            "You are a chemist expert in natural products. "
            "Extract the compound names from the following text. "
            "Provide the answers in JSON format: "
            "[{\"compoundName\": \"Example Compound Name 1\"}, {\"compoundName\": \"Example Compound Name 2\"}]. "
            "If the compound names are not specified, leave it empty like \"\"."
        )
    },
    'bioActivity': {
        'json_key': 'bioActivity',
        'prompt': (
            "You are a chemist expert in natural products. "
            "Extract the bioactivities from the following text. "
            "Provide the answers in JSON format: "
            "[{\"bioactivity\": \"Example Bioactivity 1\"}, {\"bioactivity\": \"Example Bioactivity 2\"}]. "
            "If the bioactivities are not specified, leave it empty like \"\"."
        )
    },
    'collectionSpecie': {
        'json_key': 'collectionSpecie',
        'prompt': (
            "You are a chemist expert in natural products. "
            "Extract the species from the following text. "
            "Provide the answers in JSON format: "
            "[{\"species\": \"Example Species 1\"}, {\"species\": \"Example Species 2\"}]. "
            "If the species are not specified, leave it empty like \"\"."
        )
    },
    'collectionSite': {
        'json_key': 'collectionSite',
        'prompt': (
            "You are a chemist expert in natural products. "
            "Extract the collection sites from the following text. "
            "Provide the answers in JSON format: "
            "[{\"collectionSite\": \"Example Collection Site 1\"}, {\"collectionSite\": \"Example Collection Site 2\"}]. "
            "If the collection sites are not specified, leave it empty like \"\"."
        )
    },
    'collectionType': {
        'json_key': 'collectionType',
        'prompt': (
            "You are a chemist expert in natural products. "
            "Extract the isolation types from the following text. "
            "Provide the answers in JSON format: "
            "[{\"isolationType\": \"Example Isolation Type 1\"}, {\"isolationType\": \"Example Isolation Type 2\"}]. "
            "If the isolation types are not specified, leave it empty like \"\"."
        )
    }
}

# Load the extracted texts
extracted_text_df = pd.read_parquet(extracted_text_path)

# Preprocess filenames to extract DOIs
extracted_text_df['doi'] = extracted_text_df['filename'].str.replace('.pdf', '').str.replace('@', '/')

# Function to call the GPT API
def call_gpt_api(model_name, system_prompt, user_input, doi):
    try:
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ],
            temperature=0.0  # Set to 0 for deterministic output
        )
        assistant_reply = response['choices'][0]['message']['content']
        return assistant_reply
    except Exception as e:
        print(f"Error calling GPT API for DOI {doi}: {e}")
        return ""

# Iterate over each attribute
for attribute, info in attributes_info.items():
    print(f"Processing attribute: {attribute}")
    
    # Define the model name for the fine-tuned model
    model_name = f'ft-{attribute}-model'  # Replace with your actual fine-tuned model names
    
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
    
    # Iterate over each row and call the GPT API
    for _, row in tqdm(merged_df.iterrows(), total=merged_df.shape[0], desc=f"Attribute: {attribute}"):
        doi = row['doi']
        text = row['text']
        true_value = row['neighbor']
        edge_type = row['type']
        
        # Call GPT API with only the system prompt and the extracted text
        system_prompt = info['prompt']
        user_input = text
        
        assistant_reply = call_gpt_api(model_name, system_prompt, user_input, doi)
        
        # Parse the assistant's reply
        try:
            restored_data = json.loads(assistant_reply)
            # Extract the values for the attribute
            restored_values_list = [item.get(info['json_key'], '') for item in restored_data]
            # Ensure all extracted values are strings
            restored_values_list = [str(value) for value in restored_values_list if value]
            # Convert list to string representation with single quotes
            restored_values_str = "[" + ", ".join(f"'{value}'" for value in restored_values_list) + "]"
        except json.JSONDecodeError:
            print(f"JSON decoding failed for DOI {doi}. Raw response: {assistant_reply}")
            restored_values_str = "[]"
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
    output_filename = f'llm_results_gpt4_0.8_doi_{attribute}_0_1st.csv'
    output_path = os.path.join(output_dir, output_filename)
    
    # Save the results to CSV with proper quoting and without quotes for column names and edge_type
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['true', 'restored', 'edge_type'])  # Write header without quotes
        for true, restored, edge_type in zip(true_values, restored_values, edge_types):
            writer.writerow([true, restored, edge_type])  # Write edge_type without quotes
    
    print(f"Results saved to {output_path}\n")

print("Processing complete for all attributes.")