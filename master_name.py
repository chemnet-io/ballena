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
# Script 1: GPT Processing
# ===========================

def run_gpt_processing(test_mode=False, test_size=20):
    """
    Processes extracted text data to extract compound names using a fine-tuned GPT model.

    Parameters:
    - test_mode (bool): If True, processes only the first `test_size` entries.
    - test_size (int): Number of entries to process in test mode.
    """
    # Set OpenAI API key
    openai.api_key = OPENAI_API_KEY

    # Define the directory paths
    splits_dir = 'splits'
    extracted_text_path = os.path.join('pypdfextraction', 'extracted_text.parquet')
    output_dir = 'llm_ft_results'

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
    model_name = 'ft:gpt-4o-2024-08-06:eccenca-gmbh:ballena-name-0-1st-train-only:AFMpNyLv'

    # Define the test split file path for iteration 0 and 1st split
    test_split_filename = f'test_doi_{attribute}_0_1st.csv'
    test_split_path = os.path.join(splits_dir, test_split_filename)

    if not os.path.exists(test_split_path):
        logging.error(f"Test split file {test_split_filename} does not exist. Exiting.")
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
    logging.info(f"Extracted {len(test_dois)} unique DOIs from the test split.")

    # Filter the extracted texts for the test DOIs
    test_texts_df = extracted_text_df[extracted_text_df['doi'].isin(test_dois)]
    logging.info(f"Filtered {len(test_texts_df)} entries matching test DOIs.")

    # Merge the true values with the texts based on DOI
    merged_df = pd.merge(test_texts_df, test_split_df, left_on='doi', right_on='node', how='inner')
    logging.info(f"Merged dataframe has {len(merged_df)} entries.")

    if merged_df.empty:
        logging.warning("The merged dataframe is empty. No data to process.")
        return

    # Prepare lists to store results
    true_values = []
    restored_values = []
    edge_types = []

    # Define the specific output filename without any suffix
    output_filename = f'llm_results_ft_4o_0.8_doi_{attribute}_0_1st.csv'
    output_path = os.path.join(output_dir, output_filename)

    # Open the output file in write mode and write header
    try:
        with open(output_path, 'w', newline='', encoding='utf-8') as output_f:
            output_writer = csv.writer(output_f, quoting=csv.QUOTE_MINIMAL)
            output_writer.writerow(['true', 'restored', 'edge_type'])  # Write header
            logging.info(f"Created output file with header: {output_filename}")

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
                    # Ensure all extracted values are non-empty strings
                    restored_values_list = [str(value) for value in restored_values_list if value]
                    # Convert list to string representation with single quotes
                    restored_values_str = "[" + ", ".join(f"'{value}'" for value in restored_values_list) + "]"
                except json.JSONDecodeError:
                    logging.error(f"JSON decoding failed for DOI {doi}. Raw response: {assistant_reply}")
                    restored_values_str = "[]"
                except Exception as e:
                    logging.error(f"Unexpected error for DOI {doi}: {e}")
                    restored_values_str = "[]"

                # Assemble the data into the desired format
                true_entry = f"['{doi}', '{true_value}']"
                restored_entry = f"['{doi}', {restored_values_str}]"

                true_values.append(true_entry)
                restored_values.append(restored_entry)
                edge_types.append(edge_type)

                # Write the row to the output CSV
                output_writer.writerow([true_entry, restored_entry, edge_type])

        logging.info(f"Results saved to {output_path}\n")
        logging.info("Processing complete for the 'name' attribute.")
    except Exception as e:
        logging.error(f"Failed to write to the output file {output_path}: {e}")

# ===========================
# Script 2: Similarity Search
# ===========================

def run_similarity_search():
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
    index_directory = 'faiss_index_trained'

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
        return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

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

        Returns:
        - dict: The updated row.
        """
        try:
            restored_str = clean_restored_field(row['restored'])
            restored = ast.literal_eval(restored_str)
            # Check if 'restored' has the expected structure
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
                    old_value = row['restored']
                    row['restored'] = str(new_restored_value)
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
        batch_size = 100  # Adjust this based on your available memory

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

    def process_files_for_similarity_search(directory, attribute, top_k):
        """
        Process all relevant CSV files in the specified directory for similarity search.

        Parameters:
        - directory (str): Directory containing the CSV files.
        - attribute (str): The attribute being processed.
        - top_k (int): Number of top similar documents to retrieve.
        """
        # Updated pattern to match the base filename without any suffix
        pattern = re.compile(rf'llm_results_ft_4o_0\.8_doi_{re.escape(attribute)}_0_1st\.csv$')
        files = [f for f in os.listdir(directory) if pattern.match(f)]

        if not files:
            logging.warning(f"No files found matching pattern for attribute '{attribute}' in directory '{directory}'.")
            return

        for filename in tqdm(files, desc="Processing files"):
            file_path = os.path.join(directory, filename)
            logging.info(f"Processing file: {filename} for attribute: {attribute}")
            update_restored_with_similarity_search(file_path, attribute, top_k)

    # Process only the 'name' attribute with top_k=50
    directory = 'llm_ft_results'
    attributes_and_k = {
        'name': 50,
    }

    for attribute, top_k in attributes_and_k.items():
        logging.info(f"Starting similarity search processing for attribute: {attribute} with top_k: {top_k}")
        process_files_for_similarity_search(directory, attribute, top_k)

    logging.info("Similarity search processing completed.")

# ===========================
# Script 3: Fix Quotes
# ===========================

def run_fix_quotes():
    import os
    import csv
    import re
    import ast
    import shutil
    from glob import glob
    from tqdm import tqdm
    import logging

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
            fixed_str = fix_quotes(cell)
            # Remove any leading/trailing whitespace and ensure it starts and ends with brackets
            fixed_str = fixed_str.strip()
            if not (fixed_str.startswith('[') and fixed_str.endswith(']')):
                fixed_str = f"[{fixed_str}]"
            # Split by comma, assuming the first comma separates DOI and name
            try:
                content = fixed_str[1:-1]
                doi, name = content.split(',', 1)
                doi = doi.strip().strip('"')
                name = name.strip().strip('"')
                # Escape any existing double quotes in the name
                name = name.replace('"', '\\"')
                return f'["{doi}", "{name}"]'
            except Exception as e:
                logging.error(f"Line {line_num}: Failed to process cell '{cell}': {e}")
                return cell  # Return the original cell if all else fails

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

                    try:
                        true_col, restored_col, edge_type = row
                    except ValueError:
                        logging.warning(f"Line {line_num}: Unexpected number of columns: {row}")
                        writer.writerow(row)
                        continue

                    # Process 'true' and 'restored' columns
                    true_col_fixed = process_true_restored_field(true_col, line_num)
                    restored_col_fixed = process_true_restored_field(restored_col, line_num)

                    writer.writerow([true_col_fixed, restored_col_fixed, edge_type])
                    rows_written += 1
                    rows_processed += 1

            logging.info(f"Processed {rows_processed} rows from {input_file} and wrote {rows_written} rows to {output_file}.")
        except Exception as e:
            logging.error(f"Failed to process CSV {input_file}: {e}")

    def process_all_files(input_directory, attribute):
        """
        Process all relevant CSV files in the input directory.
        """
        # Define the pattern for 'name' CSV files without any suffix
        pattern = r'llm_results_ft_4o_0\.8_doi_name_0_1st\.csv$'

        # Get all matching files in the input directory
        files = glob(os.path.join(input_directory, '*.csv'))
        name_files = [f for f in files if re.search(pattern, os.path.basename(f))]

        if not name_files:
            logging.warning(f"No 'name' CSV files found in directory '{input_directory}'.")
            return

        for input_file in tqdm(name_files, desc="Processing 'name' CSV files"):
            filename = os.path.basename(input_file)
            # Define the output file with a temporary suffix
            output_file = os.path.join(input_directory, f"{filename}.fixed.csv")
            logging.info(f"Processing {filename}...")

            process_csv(input_file, output_file)

            # Verify that the fixed file is not empty
            if os.path.getsize(output_file) > 0:
                # Replace the original file with the fixed file
                shutil.move(output_file, input_file)
                logging.info(f"Successfully fixed and replaced {input_file}.")
            else:
                logging.error(f"Fixed file {output_file} is empty. Original file {input_file} remains unchanged.")

    # Usage
    input_directory = 'llm_ft_results'  # Directory containing the original 'name' CSV files
    attribute = 'name'
    process_all_files(input_directory, attribute)

    logging.info("Quote fixing processing completed.")

# ===========================
# Script 4: Evaluation
# ===========================

def run_evaluation():
    import pandas as pd
    import numpy as np
    import ast
    import re
    import os
    from tqdm import tqdm
    import logging

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
                s = f"[{s}]"
            try:
                return ast.literal_eval(s)
            except:
                # As a last resort, split by comma
                return [item.strip().strip('"') for item in s.strip('[]').split(',')]

    def hits_at(k, true, list_pred):
        """
        Calculate the Hits@k metric.
        """
        hits = []
        for t, p in zip(true, list_pred):
            if isinstance(p, list) and len(p) >= 2:
                pred_list = p[1]
                if isinstance(pred_list, str):
                    pred_list = parse_list(pred_list)
                hit = int(t[1] in pred_list[:k])
            else:
                hit = 0
            hits.append(hit)
        return np.mean(hits)

    def mrr(true, list_pred):
        """
        Calculate the Mean Reciprocal Rank (MRR) metric.
        """
        rrs = []
        for t, p in zip(true, list_pred):
            if isinstance(p, list) and len(p) >= 2:
                pred_list = p[1]
                if isinstance(pred_list, str):
                    pred_list = parse_list(pred_list)
                try:
                    rank = pred_list.index(t[1]) + 1
                    rrs.append(1 / rank)
                except ValueError:
                    rrs.append(0)
            else:
                rrs.append(0)
        return np.mean(rrs)

    def evaluate_attribute(input_file, output_dir, attribute):
        """
        Evaluate the specified attribute using Hits@k and MRR metrics.
        """
        logging.info(f"Starting evaluation for attribute: {attribute}")

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

        # Extract lists
        true_values = restored_df['true'].to_list()  # List of tuples: [(doi, true_value), ...]
        predicted_values = restored_df['restored'].to_list()  # List of lists: [[doi, [pred1, pred2, ...]], ...]

        # Evaluation parameters
        k_at = [50]  # You can adjust or extend these values

        # Initialize metrics storage
        hitsatk_records = []
        mrr_records = []

        # Calculate metrics
        for k in k_at:
            mean_hits = hits_at(k, true_values, predicted_values)
            hitsatk_records.append({
                'k': k,
                'metric': 'hits@k',
                'value': mean_hits
            })
            logging.info(f"Hits@{k}: {mean_hits}")

        mean_mrr = mrr(true_values, predicted_values)
        mrr_records.append({
            'metric': 'mrr',
            'value': mean_mrr
        })
        logging.info(f"MRR: {mean_mrr}")

        # Convert to DataFrames
        hitsatk_df = pd.DataFrame(hitsatk_records)
        mrr_df = pd.DataFrame(mrr_records)

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Save results
        hitsatk_output_path = os.path.join(output_dir, f'hits@k_{attribute}_evaluation.csv')
        mrr_output_path = os.path.join(output_dir, f'mrr_{attribute}_evaluation.csv')

        try:
            hitsatk_df.to_csv(hitsatk_output_path, index=False)
            mrr_df.to_csv(mrr_output_path, index=False)
            logging.info(f"Hits@k results saved to {hitsatk_output_path}")
            logging.info(f"MRR results saved to {mrr_output_path}")
        except Exception as e:
            logging.error(f"Failed to save evaluation results: {e}")

    # Usage
    path = 'llm_ft_results'
    file_name = "llm_results_ft_4o_0.8_doi_name_0_1st.csv"
    name_csv_path = os.path.join(path, file_name)
    output_dir = 'ft_evaluation_results'
    attribute = 'name'

    # Evaluate the attribute
    evaluate_attribute(name_csv_path, output_dir, attribute)

# ===========================
# Main Execution
# ===========================

def main():
    """
    Orchestrates the execution of the four scripts in the correct order.
    """
    print("Starting Script 1: GPT Processing...")
    # Set `test_mode=True` and `test_size=20` for testing
    run_gpt_processing(test_mode=True, test_size=4)
    print("\nScript 1 Completed.\n")

    print("Starting Script 2: Similarity Search...")
    run_similarity_search()
    print("\nScript 2 Completed.\n")

    print("Starting Script 3: Fix Quotes...")
    run_fix_quotes()
    print("\nScript 3 Completed.\n")

    print("Starting Script 4: Evaluation...")
    run_evaluation()
    print("\nScript 4 Completed.\n")

    print("All scripts have been executed successfully.")

if __name__ == "__main__":
    main()