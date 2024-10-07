import os
import pandas as pd
import openai
import json
import csv
import ast
import re
import shutil
import logging
from tqdm import tqdm
from dotenv import load_dotenv
from glob import glob
import numpy as np

from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS

# =========================
# Initialization and Setup
# =========================

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure the OpenAI API key is set in environment variables
openai.api_key = os.environ.get("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OpenAI API key not found. Please set the 'OPENAI_API_KEY' environment variable.")

# Define directory paths
splits_dir = 'splits'
extracted_text_path = os.path.join('pypdfextraction', 'extracted_text.parquet')
output_dir = 'llm_ft_results'  # Updated to match your environment
os.makedirs(output_dir, exist_ok=True)

# Define similarity search parameters
attributes_k = {
    'name': 50,
    'bioActivity': 5,
    'collectionSpecie': 50,
    'collectionSite': 20,
    'collectionType': 1
}

# Define the attributes and their corresponding system prompts and model names
attributes_info = {
    'collectionSite': {
        'json_key': 'collectionSite',
        'prompt': (
            "You are a chemist expert in natural products. "
            "Extract the collection sites from the following text. "
            "Provide the answers in JSON format: "
            "[{\"collectionSite\": \"Example Collection Site 1\"}, {\"collectionSite\": \"Example Collection Site 2\"}]. "
            "If the collection sites are not specified, leave it empty like \"\"."
        ),
        'model_name': 'ft:gpt-4o-2024-08-06:eccenca-gmbh:ballena-site-0-1st-train-only:AF1jozl6'
    },
    'name': {
        'json_key': 'compoundName',  # Ensure this matches GPT's JSON key
        'prompt': (
            "You are a chemist expert in natural products. "
            "Extract the compound names from the following text. "
            "Provide the answers in JSON format: "
            "[{\"compoundName\": \"Example Compound Name 1\"}, {\"compoundName\": \"Example Compound Name 2\"}]. "
            "If the compound names are not specified, leave it empty like \"\"."
        ),
        'model_name': 'ft:gpt-4o-2024-08-06:eccenca-gmbh:ballena-name-0-1st-train-only:AFMpNyLv'
    },
    'bioActivity': {
        'json_key': 'bioActivity',
        'prompt': (
            "You are a chemist expert in natural products. "
            "Extract the bioactivities from the following text. "
            "Provide the answers in JSON format: "
            "[{\"bioActivity\": \"Example BioActivity 1\"}, {\"bioActivity\": \"Example BioActivity 2\"}]. "
            "If the bioActivities are not specified, leave it empty like \"\"."
        ),
        'model_name': 'ft:gpt-4o-2024-08-06:eccenca-gmbh:ballena-bioactivity-0-1st-train-only:AFes8HvM'
    },
    'collectionSpecie': {
        'json_key': 'species',
        'prompt': (
            "You are a chemist expert in natural products. "
            "Extract the species from the following text. "
            "Provide the answers in JSON format: "
            "[{\"species\": \"Example Species 1\"}, {\"species\": \"Example Species 2\"}]. "
            "If the species are not specified, leave it empty like \"\"."
        ),
        'model_name': 'PLACEHOLDER_MODEL_NAME'  # Replace with actual model name when available
    },
    'collectionType': {
        'json_key': 'isolationType',
        'prompt': (
            "You are a chemist expert in natural products. "
            "Extract the isolation types from the following text. "
            "Provide the answers in JSON format: "
            "[{\"isolationType\": \"Example Isolation Type 1\"}, {\"isolationType\": \"Example Isolation Type 2\"}]. "
            "If the isolation types are not specified, leave it empty like \"\"."
        ),
        'model_name': 'PLACEHOLDER_MODEL_NAME'  # Replace with actual model name when available
    }
}

# =========================
# GPT Extraction Functions
# =========================

def call_gpt_api(model_name, system_prompt, user_input, doi):
    """
    Call the GPT-4o API.
    """
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
        logging.debug(f"GPT response for DOI {doi}: {assistant_reply}")
        return assistant_reply
    except openai.error.OpenAIError as e:
        logging.error(f"OpenAI API error for DOI {doi}: {e}")
        return ""
    except Exception as e:
        logging.error(f"Unexpected error calling GPT API for DOI {doi}: {e}")
        return ""

def extract_gpt_data(attributes_info, attributes_k, extracted_text_df, output_dir='llm_ft_results'):
    """
    Iterate over each attribute, call GPT API, and save the results.
    Skips attributes if the output file already exists.
    """
    for attribute, info in attributes_info.items():
        output_filename = f'llm_results_ft_4o_0.8_doi_{attribute}_0_1st.csv'  # Updated filename
        output_path = os.path.join(output_dir, output_filename)

        # Check if the output file already exists
        if os.path.exists(output_path):
            print(f"Output file {output_filename} already exists. Skipping attribute: {attribute}")
            continue

        print(f"Processing attribute: {attribute}")
        model_name = info.get('model_name', 'PLACEHOLDER_MODEL_NAME')
        
        if model_name == 'PLACEHOLDER_MODEL_NAME':
            print(f"Model name for attribute '{attribute}' is not set. Skipping processing for this attribute.")
            continue

        # Define the test split file path for iteration 0 and 1st split
        test_split_filename = f'test_doi_{attribute}_0_1st.csv'
        test_split_path = os.path.join(splits_dir, test_split_filename)

        if not os.path.exists(test_split_path):
            print(f"Test split file {test_split_filename} does not exist. Skipping attribute: {attribute}")
            continue

        # Load the test split CSV
        try:
            test_split_df = pd.read_csv(test_split_path)
        except Exception as e:
            logging.error(f"Error reading {test_split_path}: {e}")
            continue

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

            # Call GPT API with the system prompt and the extracted text
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

        # Save the results to CSV with proper quoting and without quotes for column names and edge_type
        try:
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
                writer.writerow(['true', 'restored', 'edge_type'])  # Write header without quotes
                for true, restored, edge_type in zip(true_values, restored_values, edge_types):
                    writer.writerow([true, restored, edge_type])  # Write edge_type without quotes
            print(f"GPT extraction results saved to {output_path}\n")
        except Exception as e:
            logging.error(f"Error writing to {output_path}: {e}")

    # ============================
    # Similarity Search Functions
    # ============================

    def load_faiss_index(attribute, embeddings, index_directory='faiss_index_trained'):
        """
        Load the FAISS index for a given attribute.
        """
        index_path = os.path.join(index_directory, f'unique_{attribute}.txt.index')
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index file for attribute '{attribute}' not found at {index_path}.")
        return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

    def similarity_search(faiss_index, query, top_k):
        """
        Perform a similarity search using the FAISS index.
        """
        docs_with_score = faiss_index.similarity_search_with_score(query, k=top_k)
        return docs_with_score

    def clean_restored_field(restored_str):
        """
        Clean and ensure the 'restored' field is in proper list format.
        """
        restored_str = restored_str.strip()
        if not (restored_str.startswith('[') and restored_str.endswith(']')):
            restored_str = f"[{restored_str}]"
        return restored_str

    def fix_quotes(s):
        """
        Replace single quotes with double quotes, but not within words.
        """
        return re.sub(r"(?<!\w)'|'(?!\w)", '"', s)

    def process_restored_field(attribute, restored_str, faiss_index, top_k):
        """
        Process the 'restored' field by performing similarity search and special handling for 'name'.
        """
        restored_str = clean_restored_field(restored_str)
        try:
            restored = ast.literal_eval(restored_str)
        except (ValueError, SyntaxError):
            restored_str = fix_quotes(restored_str)
            try:
                restored = ast.literal_eval(restored_str)
            except Exception as e:
                logging.error(f"Failed to parse restored field: {restored_str}. Error: {e}")
                return restored_str  # Return original if parsing fails

        if isinstance(restored, list) and len(restored) == 2 and isinstance(restored[1], list):
            # Convert all predicted values to strings
            restored[1] = [str(item) for item in restored[1]]
            # Create a query by joining the predicted values
            query = ' '.join(restored[1])
            # Perform similarity search
            docs_with_score = similarity_search(faiss_index, query, top_k)
            if docs_with_score:
                # Extract the most similar entries
                similar_entries = [doc.page_content for doc, _ in docs_with_score]
                # Update the 'restored' field with similar entries
                new_restored_value = [restored[0], similar_entries]
                if attribute == 'name':
                    # Apply special quote fixing for 'name' attribute
                    new_restored_value = fix_quotes(str(new_restored_value))
                    try:
                        new_restored_value = ast.literal_eval(new_restored_value)
                    except Exception as e:
                        logging.error(f"Failed to fix quotes for 'name' attribute: {new_restored_value}. Error: {e}")
                        new_restored_value = restored_str  # Revert to original if fixing fails
                return str(new_restored_value)
            else:
                logging.warning(f"No similar entries found for query: {query}")
                return restored_str
        else:
            logging.warning(f"Unexpected 'restored' format: {restored_str}")
            return restored_str

    def update_restored_with_similarity_search(attributes_info, attributes_k, output_dir='llm_ft_results'):
        """
        Update the 'restored' field in the GPT extraction results using similarity search.
        """
        # Initialize OpenAI embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)

        for attribute, info in attributes_info.items():
            print(f"Starting similarity search processing for attribute: {attribute}")
            input_filename = f'llm_results_ft_4o_0.8_doi_{attribute}_0_1st.csv'  # Updated filename
            input_path = os.path.join(output_dir, input_filename)

            if not os.path.exists(input_path):
                logging.warning(f"Input file {input_filename} does not exist. Skipping similarity search for attribute: {attribute}")
                continue

            try:
                faiss_index = load_faiss_index(attribute, embeddings)
            except FileNotFoundError as e:
                logging.error(e)
                continue

            top_k = attributes_k.get(attribute, 1)  # Default to 1 if not specified

            temp_file = input_path + '.temp'

            try:
                with open(input_path, 'r', newline='', encoding='utf-8') as infile, \
                     open(temp_file, 'w', newline='', encoding='utf-8') as outfile:
                    reader = csv.DictReader(infile)
                    writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames, quoting=csv.QUOTE_ALL)
                    writer.writeheader()

                    for row in tqdm(reader, desc=f"Processing rows in {os.path.basename(input_path)}"):
                        restored_original = row['restored']
                        row['restored'] = process_restored_field(attribute, restored_original, faiss_index, top_k)
                        writer.writerow(row)
            except Exception as e:
                logging.error(f"Error during similarity search processing for {attribute}: {e}")
                continue

            # Replace the original file with the updated file
            try:
                os.replace(temp_file, input_path)
                logging.info(f"Similarity search updated file saved at {input_path}")
            except Exception as e:
                logging.error(f"Error replacing the original file with the updated file for {attribute}: {e}")

        print("Similarity search processing completed for all attributes.")

    # =====================================
    # Special Handling for 'name' Attribute
    # =====================================

    def fix_quotes_name(s):
        """
        Replace single quotes with double quotes, but not within words for 'name' attribute.
        """
        return re.sub(r"(?<!\w)'|'(?!\w)", '"', s)

    def process_true_restored_field_for_name(cell):
        """
        Process the 'true' or 'restored' field for the 'name' attribute to ensure proper formatting.
        """
        try:
            # Attempt to parse the cell as a Python list
            parsed = ast.literal_eval(cell)
            if isinstance(parsed, list):
                # Ensure all elements are strings
                fixed = [str(item) for item in parsed]
                return str(fixed)
            else:
                # If not a list, treat it as a single string
                return f'["{str(parsed)}"]'
        except (ValueError, SyntaxError):
            # If parsing fails, attempt to fix quotes and reformat
            fixed_str = fix_quotes_name(cell)
            fixed_str = fixed_str.strip()
            if not (fixed_str.startswith('[') and fixed_str.endswith(']')):
                fixed_str = f'[{fixed_str}]'
            try:
                content = fixed_str[1:-1]
                doi, name = content.split(',', 1)
                doi = doi.strip().strip('"')
                name = name.strip().strip('"')
                # Escape any existing double quotes in the name
                name = name.replace('"', '\\"')
                return f'["{doi}", "{name}"]'
            except Exception as e:
                logging.error(f"Failed to process cell '{cell}': {e}")
                return cell  # Return the original cell if all else fails

    def process_name_csv(input_file, output_file):
        """
        Process the 'name' CSV file to fix quotes and ensure proper formatting.
        Overwrites the original file in the output directory.
        """
        try:
            with open(input_file, 'r', newline='', encoding='utf-8') as infile, \
                 open(output_file, 'w', newline='', encoding='utf-8') as outfile:
                reader = csv.reader(infile)
                writer = csv.writer(outfile, quoting=csv.QUOTE_ALL)

                for row in reader:
                    if reader.line_num == 1:
                        # Write header as is
                        writer.writerow(row)
                        continue

                    try:
                        true_col, restored_col, edge_type = row
                    except ValueError:
                        logging.warning(f"Unexpected number of columns in row {reader.line_num}: {row}")
                        writer.writerow(row)
                        continue

                    # Process 'true' and 'restored' columns
                    true_col_fixed = process_true_restored_field_for_name(true_col)
                    restored_col_fixed = process_true_restored_field_for_name(restored_col)

                    writer.writerow([true_col_fixed, restored_col_fixed, edge_type])
        except Exception as e:
            logging.error(f"Error processing name CSV: {e}")

        logging.info(f"Processed 'name' file saved as: {output_file}")

    def fix_name_quotes_in_similarity_search(output_dir='llm_ft_results'):
        """
        Apply special quote fixing to the 'name' attribute after similarity search.
        Overwrites the original 'name' CSV file in the output directory.
        """
        attribute = 'name'
        input_filename = f'llm_results_ft_4o_0.8_doi_{attribute}_0_1st.csv'  # Updated filename
        input_path = os.path.join(output_dir, input_filename)
        output_file = input_path  # Overwrite the same file

        if not os.path.exists(input_path):
            logging.warning(f"Input 'name' file {input_filename} does not exist. Skipping special processing.")
            return

        process_name_csv(input_path, output_file)

        print(f"Special 'name' processing completed. File overwritten at {output_file}")

    # ========================
    # Evaluation Functions
    # ========================

    def hits_at(k, true, list_pred):
        """
        Calculate the Hits@k metric.
        """
        hits = []
        missed_entries = []

        for index_t, t in enumerate(true):
            hit = False
            doi_true, true_val = t
            predictions = list_pred[index_t][1]

            for lp in predictions[:k]:
                if true_val == lp:
                    hits.append(1)
                    hit = True
                    break
            if not hit:
                hits.append(0)
                missed_entries.append((index_t, t, predictions[:k]))

        return np.mean(hits), missed_entries

    def mrr_metric(true, list_pred):
        """
        Calculate the Mean Reciprocal Rank (MRR) metric.
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
                rrs.append(0)
                missed_entries.append((index_t, t, predictions))

        return np.mean(rrs), missed_entries

    def parse_list(s):
        """
        Parse a string representation of a list into an actual Python list.
        """
        try:
            return ast.literal_eval(s)
        except (ValueError, SyntaxError):
            # Attempt to fix common formatting issues
            s = s.strip()
            if not (s.startswith('[') and s.endswith(']')):
                s = f'[{s}]'
            try:
                return ast.literal_eval(s)
            except:
                # As a last resort, split by comma
                return [item.strip().strip('"') for item in s.strip('[]').split(',')]

    def evaluate_attribute(input_csv_path, output_dict, attribute_name, is_name=False):
        """
        Evaluate the specified attribute using Hits@k and MRR metrics.
        Accumulates results into the provided output_dict.
        """
        # Evaluation parameters
        if is_name:
            k_at = [50]
        else:
            # Define k values based on the attributes_k dictionary
            k_at = [attributes_k.get(attribute_name, 1)]

        # Load the CSV file
        try:
            restored_df = pd.read_csv(input_csv_path)
        except FileNotFoundError:
            logging.error(f"The file {input_csv_path} does not exist. Please check the path.")
            return
        except Exception as e:
            logging.error(f"Error reading {input_csv_path}: {e}")
            return

        # Ensure 'true' and 'restored' columns exist
        if 'true' not in restored_df.columns or 'restored' not in restored_df.columns:
            logging.error(f"The input CSV must contain 'true' and 'restored' columns.")
            return

        # Apply literal_eval to parse string representations of lists/tuples
        restored_df['true'] = restored_df['true'].apply(parse_list)
        restored_df['restored'] = restored_df['restored'].apply(parse_list)

        # Extract true and predicted lists
        true_values = restored_df['true'].to_list()  # List of tuples: [(doi, true_value), ...]
        predicted_values = restored_df['restored'].to_list()  # List of lists: [[doi, [pred1, pred2, ...]], ...]

        # Calculate Hits@k
        for k in k_at:
            mean_hits, missed = hits_at(k, true_values, predicted_values)
            output_dict['hits_at_k'].append({
                'attribute': attribute_name,
                'k': k,
                'value': mean_hits
            })

        # Calculate MRR
        mean_mrr, missed_m = mrr_metric(true_values, predicted_values)
        output_dict['mrr'].append({
            'attribute': attribute_name,
            'value': mean_mrr
        })

    def evaluate_all_attributes(attributes_info, output_dir='llm_ft_results', evaluation_output_dir='ft_evaluation_results'):
        """
        Evaluate all attributes using Hits@k and MRR metrics and save to consolidated CSV files.
        """
        os.makedirs(evaluation_output_dir, exist_ok=True)

        # Initialize metrics storage
        evaluation_results = {
            'hits_at_k': [],
            'mrr': []
        }

        for attribute in attributes_info.keys():
            input_filename = f'llm_results_ft_4o_0.8_doi_{attribute}_0_1st.csv'  # Updated filename
            input_path = os.path.join(output_dir, input_filename)

            if not os.path.exists(input_path):
                logging.warning(f"Input file {input_filename} does not exist. Skipping evaluation for attribute: {attribute}")
                continue

            if attribute == 'name':
                evaluate_attribute(input_path, evaluation_results, attribute, is_name=True)
            else:
                evaluate_attribute(input_path, evaluation_results, attribute, is_name=False)

        # Convert metrics to DataFrames
        hitsatk_df = pd.DataFrame(evaluation_results['hits_at_k'])
        mrr_df = pd.DataFrame(evaluation_results['mrr'])

        # Save Hits@k results
        hitsatk_output_path = os.path.join(evaluation_output_dir, f'hits@k_evaluation.csv')
        try:
            hitsatk_df.to_csv(hitsatk_output_path, index=False)
            print(f"Hits@k results saved to {hitsatk_output_path}")
        except Exception as e:
            logging.error(f"Error saving Hits@k results: {e}")

        # Save MRR results
        mrr_output_path = os.path.join(evaluation_output_dir, f'mrr_evaluation.csv')
        try:
            mrr_df.to_csv(mrr_output_path, index=False)
            print(f"MRR results saved to {mrr_output_path}")
        except Exception as e:
            logging.error(f"Error saving MRR results: {e}")

        print("Evaluation completed for all attributes.")

    # =====================
    # Main Execution Flow
    # =====================

    def main():
        # Load the extracted texts
        try:
            extracted_text_df = pd.read_parquet(extracted_text_path)
        except FileNotFoundError:
            logging.error(f"The file {extracted_text_path} does not exist. Please check the path.")
            return
        except Exception as e:
            logging.error(f"Error reading {extracted_text_path}: {e}")
            return

        # Preprocess filenames to extract DOIs
        if 'filename' not in extracted_text_df.columns:
            logging.error(f"The extracted text DataFrame must contain a 'filename' column.")
            return

        extracted_text_df['doi'] = extracted_text_df['filename'].str.replace('.pdf', '').str.replace('@', '/')

        # Step 1: GPT Extraction
        extract_gpt_data(attributes_info, attributes_k, extracted_text_df, output_dir=output_dir)

        # Step 2: Similarity Search
        update_restored_with_similarity_search(attributes_info, attributes_k, output_dir=output_dir)

        # Step 3: Special Handling for 'name' Attribute
        fix_name_quotes_in_similarity_search(output_dir=output_dir)

        # Step 4: Evaluation
        evaluate_all_attributes(attributes_info, output_dir=output_dir, evaluation_output_dir='ft_evaluation_results')

        print("Master script execution completed successfully.")

    if __name__ == "__main__":
        main()