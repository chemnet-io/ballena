import os
import pandas as pd
import openai
import json
import csv
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Ensure the OpenAI API key is set in environment variables
openai.api_key = os.environ.get("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OpenAI API key not found. Please set the 'OPENAI_API_KEY' environment variable.")

# Define the directory paths
splits_dir = 'splits'
extracted_text_path = os.path.join('pypdfextraction', 'extracted_text.parquet')
output_dir = 'llm_ft_results'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Define the bioActivity attribute information
bioActivity_info = {
    'json_key': 'bioactivity',
    'prompt': (
        "You are a chemist expert in natural products. "
        "Extract the bioactivities from the following text. "
        "Provide the answers in JSON format: "
        "[{\"bioactivity\": \"Example Bioactivity 1\"}, {\"bioactivity\": \"Example Bioactivity 2\"}]. "
        "If the bioactivities are not specified, leave it empty like \"\"."
    )
}

# Load the extracted texts
try:
    extracted_text_df = pd.read_parquet(extracted_text_path)
except FileNotFoundError:
    raise FileNotFoundError(f"The file {extracted_text_path} does not exist. Please check the path.")

# Preprocess filenames to extract DOIs
extracted_text_df['doi'] = extracted_text_df['filename'].str.replace('.pdf', '', regex=False).str.replace('@', '/', regex=False)

# Function to call the GPT API
def call_gpt_api(model_name, system_prompt, user_input, doi):
    try:
        response = openai.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ],
            temperature=0.0  # Set to 0 for deterministic output
        )
        # Corrected attribute access
        assistant_reply = response.choices[0].message.content
        return assistant_reply
    except openai.error.OpenAIError as e:
        print(f"OpenAI API error for DOI {doi}: {e}")
        return ""
    except Exception as e:
        print(f"Unexpected error calling GPT API for DOI {doi}: {e}")
        return ""

# Processing only the 'bioActivity' attribute
attribute = 'bioActivity'
info = bioActivity_info
print(f"Processing attribute: {attribute}")

# Define the specific fine-tuned model name for 'bioActivity' attribute
model_name = 'ft:gpt-4o-2024-08-06:eccenca-gmbh:ballena-bioactivity-0-1st:AEDQucg2'

# Define the test split file path for iteration 0 and 1st split
test_split_filename = f'test_doi_{attribute}_0_1st.csv'
test_split_path = os.path.join(splits_dir, test_split_filename)

if not os.path.exists(test_split_path):
    print(f"Test split file {test_split_filename} does not exist. Exiting.")
    exit(1)

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

# Define the backup file
backup_filename = f'llm_results_ft_4o_0.8_doi_{attribute}_0_1st_backup.csv'
backup_path = os.path.join(output_dir, backup_filename)

# Check if backup file exists to determine if header should be written
backup_exists = os.path.exists(backup_path)

# Open the backup file in append mode and write header if it doesn't exist
with open(backup_path, 'a', newline='', encoding='utf-8') as backup_f:
    backup_writer = csv.writer(backup_f, quoting=csv.QUOTE_MINIMAL)
    if not backup_exists:
        backup_writer.writerow(['doi', 'assistant_reply'])  # Write header

    # Iterate over each row and call the GPT API
    for _, row in tqdm(merged_df.iterrows(), total=merged_df.shape[0], desc=f"Attribute: {attribute}"):
        doi = row['doi']
        text = row['text']
        true_value = row['neighbor']
        edge_type = row['type']

        # Call GPT API with the system prompt and the extracted text
        system_prompt = info['prompt']
        user_input = text

        assistant_reply = call_gpt_api(model_name, system_prompt, user_input, doi)

        # Write the DOI and assistant's reply to the backup file
        backup_writer.writerow([doi, assistant_reply])

        # Parse the assistant's reply
        try:
            restored_data = json.loads(assistant_reply)
            # Extract the values for the attribute
            restored_values_list = [item.get(info['json_key'], '') for item in restored_data]
            # Ensure all extracted values are non-empty strings
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

# Define the main output filename
output_filename = f'llm_results_ft_4o_0.8_doi_{attribute}_0_1st.csv'
output_path = os.path.join(output_dir, output_filename)

# Save the results to the main CSV file with proper quoting
with open(output_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['true', 'restored', 'edge_type'])  # Write header
    for true, restored, edge_type in zip(true_values, restored_values, edge_types):
        writer.writerow([true, restored, edge_type])

print(f"Results saved to {output_path}\n")
print("Processing complete for the 'bioActivity' attribute.")