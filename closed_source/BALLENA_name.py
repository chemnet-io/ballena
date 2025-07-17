import os
import pandas as pd
import openai
import json
import csv
from tqdm import tqdm
from dotenv import load_dotenv
import ast
import re
import logging
import shutil
from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from glob import glob


# ===========================
# Name Processing Pipeline
# ===========================


# This script is designed to process the 'name' attribute of the BALLENA dataset. It uses a fine-tuned GPT model to extract 
# compound names from the text data, then performs a similarity search to restore the names based on the extracted text.
# It can be customized to only do certain splits, and it has a testing mode for only testing its functionality on x number of rows.


# ===========================
# Environment and Logging Setup
# ===========================

# Load environment variables from .env file
load_dotenv()

# Ensure the OpenAI API key is set in environment variables
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OpenAI API key not found. Please set the 'OPENAI_API_KEY' environment variable.")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ===========================
# Define Splits
# ===========================

# Define the list of splits to process
splits = ['1st', '2nd', '3rd', '4th']  # Add or remove splits as needed

# ===========================
# Script 1: GPT Processing
# ===========================

def run_gpt_processing(split, test_mode=False, test_size=20):
    """
    Processes extracted text data to extract compound names using a fine-tuned GPT model.

    Parameters:
    - split (str): The split identifier (e.g., '1st', '2nd', '3rd', '4th').
    - test_mode (bool): If True, processes only the first `test_size` entries.
    - test_size (int): Number of entries to process in test mode.
    """
    # Set OpenAI API key
    openai.api_key = OPENAI_API_KEY

    # Define the directory paths
    splits_dir = 'splits'
    extracted_text_path = os.path.join('pdf_extractions', 'nougat_OCR', 'nougat.parquet')
    # output_dir = 'extraction_results/PyMuPDF_FT'
    # output_dir = 'extraction_results/Nougat_FT'
    output_dir = 'extraction_results/Nougat_SS'
    # output_dir = 'extraction_results/Nougat'

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Define the name attribute information
    name_info = {
        'json_key': 'compoundName',
        'prompt': (
            "You are a chemist expert in natural products. "
            "Extract the compound names from the following text. "
            "Provide the answers in JSON format: "
            "[{\"compoundName\": \"Example Compound Name 1\"}, {\"compoundName\": \"Example Compound Name 2\"}]. "
            "If the compound names are not specified, leave it empty like \"\"."
        )
    }

    # Load the extracted texts
    try:
        extracted_text_df = pd.read_parquet(extracted_text_path)
        logging.info(f"Loaded {len(extracted_text_df)} entries from {extracted_text_path}.")
    except FileNotFoundError:
        logging.error(f"The file {extracted_text_path} does not exist. Please check the path.")
        return
    except Exception as e:
        logging.error(f"Unexpected error loading {extracted_text_path}: {e}")
        return

    # **Test Mode Adjustment: Limit to first `test_size` entries if `test_mode` is True**
    if test_mode:
        extracted_text_df = extracted_text_df.head(test_size)
        logging.info(f"Test mode enabled: Processing only the first {test_size} entries.")

    # Preprocess filenames to extract DOIs
    if 'filename' not in extracted_text_df.columns:
        logging.error("The extracted text dataframe does not contain the 'filename' column.")
        return

    extracted_text_df['doi'] = extracted_text_df['filename'].str.replace('.pdf', '', regex=False).str.replace('@', '/', regex=False)
    logging.info("Extracted DOIs from filenames.")

    # Verify that 'doi' column has been created correctly
    if extracted_text_df['doi'].isnull().any():
        logging.warning("Some DOIs could not be extracted and are NaN.")

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
            assistant_reply = response.choices[0].message.content
            return assistant_reply
        except openai.error.OpenAIError as e:
            logging.error(f"OpenAI API error for DOI {doi}: {e}")
            return ""
        except Exception as e:
            logging.error(f"Unexpected error calling GPT API for DOI {doi}: {e}")
            return ""

    # Processing the 'name' attribute
    attribute = 'name'
    info = name_info
    logging.info(f"Processing attribute: {attribute}")

    # Define the specific fine-tuned model name for 'name' attribute
    # model_name = 'ft:gpt-4o-2024-08-06:chemnet:ballena-nougat-name-0-1st:AItbMK9c'
    model_name = 'gpt-4o-2024-08-06'

    # Define the test split file path based on the current split
    test_split_filename = f'test_doi_{attribute}_0_{split}.csv'
    test_split_path = os.path.join(splits_dir, test_split_filename)

    if not os.path.exists(test_split_path):
        logging.error(f"Test split file {test_split_filename} does not exist. Skipping split '{split}'.")
        return

    # Load the test split CSV
    try:
        test_split_df = pd.read_csv(test_split_path)
        logging.info(f"Loaded test split from {test_split_path} with {len(test_split_df)} entries.")
    except Exception as e:
        logging.error(f"Error reading test split file {test_split_path}: {e}")
        return

    # Extract the DOIs from the test split
    if 'node' not in test_split_df.columns:
        logging.error("The test split dataframe does not contain the 'node' column.")
        return

    test_dois = test_split_df['node'].unique()
    logging.info(f"Extracted {len(test_dois)} unique DOIs from the test split '{split}'.")

    # Filter the extracted texts for the test DOIs
    test_texts_df = extracted_text_df[extracted_text_df['doi'].isin(test_dois)]
    logging.info(f"Filtered {len(test_texts_df)} entries matching test DOIs for split '{split}'.")

    # Merge the true values with the texts based on DOI
    merged_df = pd.merge(test_texts_df, test_split_df, left_on='doi', right_on='node', how='inner')
    logging.info(f"Merged dataframe has {len(merged_df)} entries for split '{split}'.")

    if merged_df.empty:
        logging.warning(f"The merged dataframe is empty for split '{split}'. No data to process.")
        return

    # Prepare lists to store results
    true_values = []
    restored_values = []
    edge_types = []

    # Define the specific output filename based on the current split
    output_filename = f'llm_results_ft_4o_0.8_doi_{attribute}_0_{split}.csv'
    output_path = os.path.join(output_dir, output_filename)

    # Open the output file in write mode and write header
    try:
        with open(output_path, 'w', newline='', encoding='utf-8') as output_f:
            output_writer = csv.writer(output_f, quoting=csv.QUOTE_ALL)
            output_writer.writerow(['true', 'restored', 'edge_type'])  # Write header
            logging.info(f"Created output file with header: {output_filename}")

            # Iterate over each row and call the GPT API
            for _, row in tqdm(merged_df.iterrows(), total=merged_df.shape[0], desc=f"Attribute: {attribute} | Split: {split}"):
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
                    # Clean reply of any code blocking
                    assistant_reply = re.sub('```json', '', assistant_reply) # type: ignore
                    assistant_reply = re.sub('```', '', assistant_reply)
                    restored_data = json.loads(assistant_reply)
                    # Extract the values for the attribute
                    restored_values_list = [item.get(info['json_key'], '') for item in restored_data]
                    # Ensure all extracted values are non-empty strings
                    restored_values_list = [str(value) for value in restored_values_list if value]
                except json.JSONDecodeError:
                    logging.error(f"JSON decoding failed for DOI {doi}. Raw response: {assistant_reply}")
                    restored_values_list = []
                except Exception as e:
                    logging.error(f"Unexpected error for DOI {doi}: {e}")
                    restored_values_list = []

                # Convert list to string representation with single quotes
                restored_values_str = json.dumps(restored_values_list)

                # Assemble the data into the desired format using JSON dumps
                true_entry = json.dumps([doi, true_value])
                restored_entry = json.dumps([doi, restored_values_list])

                true_values.append(true_entry)
                restored_values.append(restored_entry)
                edge_types.append(edge_type)

                # Write the row to the output CSV
                output_writer.writerow([true_entry, restored_entry, edge_type])

        logging.info(f"Results saved to {output_path}\n")
        logging.info(f"Processing complete for the 'name' attribute and split '{split}'.")
    except Exception as e:
        logging.error(f"Failed to write to the output file {output_path}: {e}")

# ===========================
# Script 2: Similarity Search
# ===========================

def run_similarity_search(split):
    """
    Perform similarity search for a specific split.
    
    Parameters:
    - split (str): The split identifier (e.g., '2nd', '3rd', '4th').
    """
    import os
    import ast
    import csv
    import re
    import logging
    import shutil
    from tqdm import tqdm
    from langchain.docstore.document import Document
    from langchain_openai import OpenAIEmbeddings
    from langchain.vectorstores.faiss import FAISS

    # Initialize OpenAI embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    # Directory containing the FAISS indexes
    index_directory = 'faiss_index'

    def load_faiss_index(attribute):
        """
        Load the FAISS index for a given attribute.

        Parameters:
        - attribute (str): The attribute name (e.g., 'name').

        Returns:
        - FAISS: The loaded FAISS index object.
        """
        index_path = os.path.join(index_directory, f'unique_{attribute}.txt.index')
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index file for attribute '{attribute}' not found at {index_path}.")
        return FAISS.load_local(index_path, embeddings)

    def similarity_search(faiss_index, query, top_k):
        """
        Perform a similarity search using the FAISS index.

        Parameters:
        - faiss_index (FAISS): The FAISS index object.
        - query (str): The query string.
        - top_k (int): Number of top similar documents to retrieve.

        Returns:
        - list of tuples: Each tuple contains (Document, score).
        """
        try:
            docs_with_score = faiss_index.similarity_search_with_score(query, k=top_k)
            return docs_with_score
        except Exception as e:
            logging.error(f"Similarity search failed: {e}")
            return []

    def clean_restored_field(restored_str):
        """
        Clean and ensure the 'restored' field is in proper list format.

        Parameters:
        - restored_str (str): The string representation of the 'restored' field.

        Returns:
        - str: Cleaned string in list format.
        """
        restored_str = restored_str.strip()
        if not (restored_str.startswith('[') and restored_str.endswith(']')):
            restored_str = f"[{restored_str}]"
        return restored_str

    def process_row(row, faiss_index, attribute, top_k, line_num):
        """
        Process a single row by performing similarity search and updating the 'restored' field.

        Parameters:
        - row (dict): The CSV row as a dictionary.
        - faiss_index (FAISS): The FAISS index object.
        - attribute (str): The attribute being processed.
        - top_k (int): Number of top similar documents to retrieve.
        - line_num (int): Line number in the CSV for logging purposes.

        Returns:
        - dict: The updated row.
        """
        try:
            restored_str = clean_restored_field(row['restored'])
            restored = json.loads(restored_str)
            # Check if 'restored' has the expected structure
            if isinstance(restored, list) and len(restored) == 2 and isinstance(restored[1], list):
                # Convert all predicted values to strings
                restored[1] = [str(item) for item in restored[1]]
                # Create a query by joining the predicted values without quotes
                query = ' '.join(restored[1])
                # Perform similarity search
                docs_with_score = similarity_search(faiss_index, query, top_k)
                if docs_with_score:
                    # Extract the most similar entries and wrap them in double quotes
                    similar_entries = [f'"{doc.page_content}"' for doc, _ in docs_with_score]
                    # Update the 'restored' field with similar entries
                    new_restored_value = [restored[0], similar_entries]
                    old_value = row['restored']
                    row['restored'] = json.dumps(new_restored_value)
                    logging.info(f"Line {line_num}: Updated 'restored' from {old_value} to {new_restored_value} for DOI {row['true']}")
            else:
                logging.warning(f"Line {line_num}: Unexpected 'restored' format in row: {row}")
        except (ValueError, SyntaxError, TypeError) as e:
            logging.error(f"Line {line_num}: Error parsing 'restored' field in row {row}: {e}")
        return row

    def update_restored_with_similarity_search(file_path, attribute, top_k):
        """
        Update the 'restored' field in the CSV file using similarity search.

        Parameters:
        - file_path (str): Path to the input CSV file.
        - attribute (str): The attribute being processed.
        - top_k (int): Number of top similar documents to retrieve.
        """
        try:
            faiss_index = load_faiss_index(attribute)
            logging.info(f"Loaded FAISS index from {index_directory} for attribute '{attribute}'.")
        except FileNotFoundError as e:
            logging.error(e)
            return
        except Exception as e:
            logging.error(f"Error loading FAISS index: {e}")
            return

        temp_file = file_path + '.temp'
        batch_size = 50  # Adjust this based on your available memory

        try:
            with open(file_path, 'r', newline='', encoding='utf-8') as infile, \
                 open(temp_file, 'w', newline='', encoding='utf-8') as outfile:
                reader = csv.DictReader(infile)
                writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames, quoting=csv.QUOTE_ALL)
                writer.writeheader()

                for line_num, row in enumerate(reader, start=2):  # Start at 2 to account for header
                    updated_row = process_row(row, faiss_index, attribute, top_k, line_num)
                    writer.writerow(updated_row)

            os.replace(temp_file, file_path)
            logging.info(f"Updated file saved at {file_path}")
        except Exception as e:
            logging.error(f"Failed to process similarity search for file {file_path}: {e}")

    def process_files_for_similarity_search(directory, attribute, top_k, split):
        """
        Process all relevant CSV files in the specified directory for similarity search.

        Parameters:
        - directory (str): Directory containing the CSV files.
        - attribute (str): The attribute being processed.
        - top_k (int): Number of top similar documents to retrieve.
        - split (str): The split identifier (e.g., '2nd', '3rd', '4th').
        """
        pattern = re.compile(rf'llm_results_ft_4o_0\.8_doi_{re.escape(attribute)}_0_{re.escape(split)}\.csv$')
        files = [f for f in os.listdir(directory) if pattern.match(f)]

        if not files:
            logging.warning(f"No files found matching pattern for attribute '{attribute}' and split '{split}' in directory '{directory}'.")
            return

        for filename in tqdm(files, desc=f"Processing files for split '{split}'"):
            file_path = os.path.join(directory, filename)
            logging.info(f"Processing file: {filename} for attribute: {attribute} and split: {split}")
            update_restored_with_similarity_search(file_path, attribute, top_k)

    # Process only the 'name' attribute with top_k=50
    # directory = 'extraction_results/Nougat_FT'
    directory = 'extraction_results/Nougat_SS'
    # directory = 'extraction_results/Nougat'
    attributes_and_k = {
        'name': 50,
    }

    for attribute, top_k in attributes_and_k.items():
        logging.info(f"Starting similarity search processing for attribute: {attribute} with top_k: {top_k} for split '{split}'")
        process_files_for_similarity_search(directory, attribute, top_k, split)

    logging.info(f"Similarity search processing completed for split '{split}'.")

# ===========================
# Script 3: Fix Quotes
# ===========================

def run_fix_quotes(split):
    """
    Fix quotes in the 'true' and 'restored' fields for a specific split.

    Parameters:
    - split (str): The split identifier (e.g., '2nd', '3rd', '4th').
    """
    import os
    import csv
    import re
    import ast
    import shutil
    from glob import glob
    from tqdm import tqdm
    import logging
    import json

    def fix_quotes(s):
        """
        Replace single quotes with double quotes, but not within words.
        """
        return re.sub(r"(?<!\w)'|'(?!\w)", '"', s)

    def process_true_restored_field(cell, line_num):
        """
        Process the 'true' or 'restored' field to ensure it's a properly formatted list of strings.
        """
        try:
            # Attempt to parse the cell as a Python list
            parsed = json.loads(cell)
            if isinstance(parsed, list):
                # Ensure all elements are strings and properly quoted
                fixed = [str(item) for item in parsed]
                return json.dumps(fixed)
            else:
                # If not a list, treat it as a single string
                return json.dumps([str(parsed)])
        except json.JSONDecodeError:
            try:
                # If JSON parsing fails, attempt to fix quotes and reformat
                fixed_str = fix_quotes(cell)
                # Remove any leading/trailing whitespace and ensure it starts and ends with brackets
                fixed_str = fixed_str.strip()
                if not (fixed_str.startswith('[') and fixed_str.endswith(']')):
                    fixed_str = f"[{fixed_str}]"
                # Split by comma, assuming the first comma separates DOI and name
                content = fixed_str[1:-1]
                doi, name = content.split(',', 1)
                doi = doi.strip().strip('"')
                name = name.strip().strip('"')
                # Escape any existing double quotes in the name
                name = name.replace('"', '\\"')
                return json.dumps([doi, name])
            except Exception as e:
                logging.error(f"Line {line_num}: Failed to process cell '{cell}': {e}")
                return json.dumps(cell)  # Return the original cell as a JSON string if all else fails

    def process_csv(input_file, output_file):
        """
        Process a single CSV file, fixing the 'true' and 'restored' columns.
        """
        rows_processed = 0
        rows_written = 0
        try:
            with open(input_file, 'r', newline='', encoding='utf-8') as infile, \
                 open(output_file, 'w', newline='', encoding='utf-8') as outfile:
                reader = csv.reader(infile)
                writer = csv.writer(outfile, quoting=csv.QUOTE_ALL)

                for line_num, row in enumerate(reader, start=1):
                    if line_num == 1:
                        # Write header as is
                        writer.writerow(row)
                        continue

                    if len(row) != 3:
                        logging.warning(f"Line {line_num}: Unexpected number of columns: {row}")
                        writer.writerow(row)
                        continue

                    true_col, restored_col, edge_type = row

                    # Process 'true' and 'restored' columns
                    true_col_fixed = process_true_restored_field(true_col, line_num)
                    restored_col_fixed = process_true_restored_field(restored_col, line_num)

                    writer.writerow([true_col_fixed, restored_col_fixed, edge_type])
                    rows_written += 1
                    rows_processed += 1

            logging.info(f"Processed {rows_processed} rows from {input_file} and wrote {rows_written} rows to {output_file}.")
        except Exception as e:
            logging.error(f"Failed to process CSV {input_file}: {e}")

    def process_all_files(input_directory, attribute, split):
        """
        Process all relevant CSV files in the input directory for a specific split.
        """
        # Define the pattern for the current split's 'name' CSV files without any suffix
        pattern = rf'llm_results_ft_4o_0\.8_doi_{re.escape(attribute)}_0_{re.escape(split)}\.csv$'

        # Get all matching files in the input directory
        files = [f for f in os.listdir(input_directory) if re.match(pattern, f)]

        if not files:
            logging.warning(f"No 'name' CSV files found for split '{split}' in directory '{input_directory}'.")
            return

        for input_file in tqdm(files, desc=f"Processing 'name' CSV files for split '{split}'"):
            file_path = os.path.join(input_directory, input_file)
            filename = os.path.basename(input_file)
            # Define the output file with a temporary suffix
            output_file = os.path.join(input_directory, f"{filename}.fixed.csv")
            logging.info(f"Processing {filename} for split '{split}'...")

            process_csv(file_path, output_file)

            # Verify that the fixed file is not empty
            if os.path.getsize(output_file) > 0:
                # Replace the original file with the fixed file
                shutil.move(output_file, file_path)
                logging.info(f"Successfully fixed and replaced {file_path} for split '{split}'.")
            else:
                logging.error(f"Fixed file {output_file} is empty. Original file {file_path} remains unchanged.")

    # Usage
    # input_directory = 'extraction_results/Nougat_FT'  # Directory containing the original 'name' CSV files
    input_directory = 'extraction_results/Nougat_SS'
    # input_directory = 'extraction_results/Nougat'
    attribute = 'name'
    process_all_files(input_directory, attribute, split)

    logging.info(f"Quote fixing processing completed for split '{split}'.")

# ===========================
# Script 4: Evaluation
# ===========================

def run_evaluation(split):
    """
    Evaluate the specified split using Hits@k and MRR metrics.

    Parameters:
    - split (str): The split identifier (e.g., '2nd', '3rd', '4th').
    """
    import os
    import pandas as pd
    import json
    import csv
    from tqdm import tqdm
    import logging
    import numpy as np

    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def parse_list(s):
        """
        Parse a string representation of a list into an actual Python list.
        Handles escaped quotes within the list items.
        """
        try:
            # Load the string as JSON
            return json.loads(s)
        except json.JSONDecodeError:
            try:
                # Replace escaped double quotes with actual double quotes
                s = s.replace('\\"', '"')
                return json.loads(s)
            except json.JSONDecodeError as e:
                logging.error(f"JSON decoding failed for string: {s}. Error: {e}")
                return []
        except Exception as e:
            logging.error(f"Unexpected error while parsing string: {s}. Error: {e}")
            return []

    def clean_predicted(pred_list):
        """
        Remove extra quotes from each predicted compound name.
        """
        if not isinstance(pred_list, list):
            return []
        cleaned = []
        for item in pred_list:
            if isinstance(item, str):
                # Remove leading and trailing quotes if present
                if item.startswith('"') and item.endswith('"'):
                    item = item[1:-1]
                cleaned.append(item)
            else:
                # If the item is not a string, skip it
                continue
        return cleaned

    def hits_at_k(k, true_values, predicted_values):
        """
        Calculate the Hits@k metric.
        """
        hits = 0
        for true, preds in zip(true_values, predicted_values):
            if true in preds[:k]:
                hits += 1
        return hits / len(true_values) if true_values else 0.0

    def mean_reciprocal_rank(true_values, predicted_values):
        """
        Calculate the Mean Reciprocal Rank (MRR) metric.
        """
        reciprocal_ranks = []
        for true, preds in zip(true_values, predicted_values):
            try:
                rank = preds.index(true) + 1
                reciprocal_ranks.append(1 / rank)
            except ValueError:
                reciprocal_ranks.append(0.0)
        return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0

    def evaluate_attribute(input_file, output_dir, attribute, split):
        """
        Evaluate the specified attribute using Hits@k and MRR metrics.

        Parameters:
        - input_file (str): Path to the input CSV file.
        - output_dir (str): Directory to save evaluation results.
        - attribute (str): The attribute being evaluated.
        - split (str): The split identifier (e.g., '2nd', '3rd', '4th').
        """
        logging.info(f"Starting evaluation for attribute: {attribute} and split '{split}'")

        try:
            restored_df = pd.read_csv(input_file)
            logging.info(f"Loaded {len(restored_df)} entries from {input_file}.")
        except FileNotFoundError:
            logging.error(f"The file {input_file} does not exist. Please check the path.")
            return
        except Exception as e:
            logging.error(f"Error reading {input_file}: {e}")
            return

        # Ensure 'true' and 'restored' columns exist
        if 'true' not in restored_df.columns or 'restored' not in restored_df.columns:
            logging.error(f"The input CSV must contain 'true' and 'restored' columns.")
            return

        # Parse 'true' and 'restored' columns
        restored_df['true'] = restored_df['true'].apply(parse_list)
        restored_df['restored'] = restored_df['restored'].apply(parse_list)

        # Extract true and predicted compound names
        true_values = []
        predicted_values = []

        for idx, row in restored_df.iterrows():
            # Parse 'true' field
            if isinstance(row['true'], list) and len(row['true']) == 2:
                doi_true, true_val = row['true']
                true_values.append(true_val)
            else:
                logging.warning(f"Row {idx} has an unexpected format in 'true' column: {row['true']}")
                true_values.append("")

            # Parse 'restored' field
            if isinstance(row['restored'], list) and len(row['restored']) == 2:
                doi_restored, preds = row['restored']
                cleaned_preds = clean_predicted(preds)
                predicted_values.append(cleaned_preds)
            else:
                logging.warning(f"Row {idx} has an unexpected format in 'restored' column: {row['restored']}")
                predicted_values.append([])

        # Define evaluation parameters
        k = 50  # As per your script

        # Calculate Hits@k and MRR
        hits_at_k_score = hits_at_k(k, true_values, predicted_values)
        mrr_score = mean_reciprocal_rank(true_values, predicted_values)

        logging.info(f"Hits@{k}: {hits_at_k_score}")
        logging.info(f"MRR: {mrr_score}")

        # Prepare results DataFrames
        hits_at_k_df = pd.DataFrame([{'k': k, 'metric': 'Hits@k', 'value': hits_at_k_score}])
        mrr_df = pd.DataFrame([{'metric': 'MRR', 'value': mrr_score}])

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Save results
        hitsatk_output_path = os.path.join(output_dir, f'hits@k_evaluation_{attribute}_{split}.csv')
        mrr_output_path = os.path.join(output_dir, f'mrr_evaluation_{attribute}_{split}.csv')

        try:
            hits_at_k_df.to_csv(hitsatk_output_path, index=False)
            mrr_df.to_csv(mrr_output_path, index=False)
            logging.info(f"Hits@k results saved to {hitsatk_output_path}")
            logging.info(f"MRR results saved to {mrr_output_path}")
        except Exception as e:
            logging.error(f"Failed to save evaluation results: {e}")

    # Usage
    # path = 'extraction_results/Nougat_FT'
    path = 'extraction_results/Nougat_SS'
    # path = 'extraction_results/Nougat'
    file_name = f"llm_results_ft_4o_0.8_doi_name_0_{split}.csv"
    name_csv_path = os.path.join(path, file_name)
    output_dir = 'evaluation_results/Nougat_ss_evaluation_results'
    attribute = 'name'

    # Evaluate the attribute for the specific split
    evaluate_attribute(name_csv_path, output_dir, attribute, split)

    logging.info(f"Evaluation processing completed for split '{split}'.")

# ===========================
# Main Execution
# ===========================

def main():
    """
    Orchestrates the execution of the four scripts in the correct order for each split.
    """
    for split in splits:
        print("===============================")
        print(f"Starting processing for split: {split}")
        print("===============================")

        print("Starting Script 1: GPT Processing...")
        # Set `test_mode=True` and `test_size=20` for testing if needed
        run_gpt_processing(split=split, test_mode=False)
        print("\nScript 1 Completed.\n")

        print("Starting Script 2: Similarity Search...")
        run_similarity_search(split=split)
        print("\nScript 2 Completed.\n")

        # print("Starting Script 3: Fix Quotes...")
        # run_fix_quotes(split=split)
        # print("\nScript 3 Completed.\n")

        print("Starting Script 4: Evaluation...")
        run_evaluation(split=split)
        print("\nScript 4 Completed.\n")

        print(f"Completed processing for split: {split}\n")

    print("All scripts have been executed successfully for all splits.")

if __name__ == "__main__":
    main()