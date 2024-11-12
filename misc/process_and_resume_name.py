import os
import pandas as pd
import openai
import json
import csv
from tqdm import tqdm
from dotenv import load_dotenv
import logging
import shutil
from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
import re

# ===========================
# WARNING: Specialized Recovery Script
# ===========================

# This script is specifically designed to resume interrupted processing of the 'name' attribute.
# It attempts to continue from where processing was halted.
# CAUTION: This is an emergency recovery tool that has not been thoroughly tested.
# Use at your own risk and verify results carefully.
# It is recommended to backup all data before running this script.


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
# Define Split
# ===========================

split = '3rd'  # The split you want to resume processing for

# ===========================
# Resume GPT Processing and Similarity Search
# ===========================

def resume_and_similarity_search(split):
    """
    Resumes GPT processing for the specified split, processes remaining entries,
    appends them to the existing CSV, and performs similarity search on the new entries.
    
    Parameters:
    - split (str): The split identifier (e.g., '3rd').
    """
    # ===========================
    # GPT Processing
    # ===========================

    def resume_gpt_processing():
        """
        Resume GPT processing by processing remaining entries and appending them to the existing output CSV.
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

        attribute = 'name'
        model_name = 'ft:gpt-4o-2024-08-06:chemnet:ballena-nougat-name-0-1st:AItbMK9c'

        # Define the test split file path based on the current split
        test_split_filename = f'test_doi_{attribute}_0_{split}.csv'
        test_split_path = os.path.join(splits_dir, test_split_filename)

        if not os.path.exists(test_split_path):
            logging.error(f"Test split file {test_split_filename} does not exist. Cannot resume processing.")
            return pd.DataFrame()  # Return empty DataFrame

        # Load the test split CSV
        try:
            test_split_df = pd.read_csv(test_split_path)
            logging.info(f"Loaded test split from {test_split_path} with {len(test_split_df)} entries.")
        except Exception as e:
            logging.error(f"Error reading test split file {test_split_path}: {e}")
            return pd.DataFrame()

        # Extract the DOIs from the test split
        if 'node' not in test_split_df.columns:
            logging.error("The test split dataframe does not contain the 'node' column.")
            return pd.DataFrame()

        test_dois = test_split_df['node'].unique()
        logging.info(f"Extracted {len(test_dois)} unique DOIs from the test split '{split}'.")

        # Load the extracted texts
        try:
            extracted_text_df = pd.read_parquet(extracted_text_path)
            logging.info(f"Loaded {len(extracted_text_df)} entries from {extracted_text_path}.")
        except FileNotFoundError:
            logging.error(f"The file {extracted_text_path} does not exist. Please check the path.")
            return pd.DataFrame()
        except Exception as e:
            logging.error(f"Unexpected error loading {extracted_text_path}: {e}")
            return pd.DataFrame()

        # Preprocess filenames to extract DOIs
        if 'filename' not in extracted_text_df.columns:
            logging.error("The extracted text dataframe does not contain the 'filename' column.")
            return pd.DataFrame()

        extracted_text_df['doi'] = extracted_text_df['filename'].str.replace('.pdf', '', regex=False).str.replace('@', '/', regex=False)
        logging.info("Extracted DOIs from filenames.")

        # Verify that 'doi' column has been created correctly
        if extracted_text_df['doi'].isnull().any():
            logging.warning("Some DOIs could not be extracted and are NaN.")

        # Filter the extracted texts for the test DOIs
        test_texts_df = extracted_text_df[extracted_text_df['doi'].isin(test_dois)]
        logging.info(f"Filtered {len(test_texts_df)} entries matching test DOIs for split '{split}'.")

        # Merge the true values with the texts based on DOI
        merged_df = pd.merge(test_texts_df, test_split_df, left_on='doi', right_on='node', how='inner')
        logging.info(f"Merged dataframe has {len(merged_df)} entries for split '{split}'.")

        if merged_df.empty:
            logging.warning(f"The merged dataframe is empty for split '{split}'. No data to process.")
            return pd.DataFrame()

        # Define the specific output filename based on the current split
        output_filename = f'llm_results_ft_4o_0.8_doi_{attribute}_0_{split}.csv'
        output_path = os.path.join(output_dir, output_filename)

        # Load existing processed DOIs
        if os.path.exists(output_path):
            try:
                existing_df = pd.read_csv(output_path)
                # Extract DOIs from the 'true' column which is JSON formatted
                processed_dois = set(existing_df['true'].apply(lambda x: json.loads(x)[0]))
                logging.info(f"Loaded {len(processed_dois)} already processed DOIs from {output_path}.")
            except Exception as e:
                logging.error(f"Error reading existing output file {output_path}: {e}")
                return pd.DataFrame()
        else:
            processed_dois = set()
            logging.info(f"No existing output file found at {output_path}. Starting fresh.")

        # Identify DOIs to process
        merged_df['doi'] = merged_df['doi'].astype(str)
        remaining_df = merged_df[~merged_df['doi'].isin(processed_dois)]
        num_remaining = len(remaining_df)
        logging.info(f"{num_remaining} DOIs remaining to process for split '{split}'.")

        if num_remaining == 0:
            logging.info("No remaining DOIs to process. Skipping GPT processing.")
            return pd.DataFrame()

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

        # Prepare to append to the existing CSV
        new_entries = []

        try:
            with open(output_path, 'a', newline='', encoding='utf-8') as output_f:
                output_writer = csv.writer(output_f, quoting=csv.QUOTE_ALL)
                # Do not write header when appending
                # Iterate over each row and call the GPT API
                for _, row in tqdm(remaining_df.iterrows(), total=remaining_df.shape[0], desc=f"Processing remaining GPT entries for split '{split}'"):
                    doi = row['doi']
                    text = row['text']
                    true_value = row['neighbor']
                    edge_type = row['type']

                    # Call GPT API with the system prompt and the extracted text
                    system_prompt = name_info['prompt']
                    user_input = text

                    assistant_reply = call_gpt_api(model_name, system_prompt, user_input, doi)

                    # Parse the assistant's reply
                    try:
                        restored_data = json.loads(assistant_reply)
                        # Extract the values for the attribute
                        restored_values_list = [item.get(name_info['json_key'], '') for item in restored_data]
                        # Ensure all extracted values are non-empty strings
                        restored_values_list = [str(value) for value in restored_values_list if value]
                    except json.JSONDecodeError:
                        logging.error(f"JSON decoding failed for DOI {doi}. Raw response: {assistant_reply}")
                        restored_values_list = []
                    except Exception as e:
                        logging.error(f"Unexpected error for DOI {doi}: {e}")
                        restored_values_list = []

                    # Assemble the data into the desired format using JSON dumps
                    true_entry = json.dumps([doi, true_value])
                    restored_entry = json.dumps([doi, restored_values_list])

                    # Write the row to the output CSV
                    output_writer.writerow([true_entry, restored_entry, edge_type])

                    # Collect new entries for similarity search
                    new_entries.append({
                        'true': true_entry,
                        'restored': restored_entry,
                        'edge_type': edge_type
                    })

            logging.info(f"Successfully processed and appended {len(new_entries)} new GPT entries to {output_path}.")
        except Exception as e:
            logging.error(f"Failed to write to the output file {output_path}: {e}")
            return pd.DataFrame()

        # Convert new entries to DataFrame for similarity search
        new_entries_df = pd.DataFrame(new_entries)
        return new_entries_df

    # ===========================
    # Similarity Search
    # ===========================

    def perform_similarity_search(new_entries_df):
        """
        Perform similarity search on the new GPT processed entries and update the 'restored' field.
        
        Parameters:
        - new_entries_df (DataFrame): DataFrame containing the new entries to perform similarity search on.
        """
        if new_entries_df.empty:
            logging.info("No new entries to perform similarity search on.")
            return

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

        attribute = 'name'
        top_k = 50
        model_name = 'ft:gpt-4o-2024-08-06:chemnet:ballena-nougat-name-0-1st:AItbMK9c'

        try:
            faiss_index = load_faiss_index(attribute)
            logging.info(f"Loaded FAISS index from {index_directory} for attribute '{attribute}'.")
        except FileNotFoundError as e:
            logging.error(e)
            return
        except Exception as e:
            logging.error(f"Error loading FAISS index: {e}")
            return

        # Define the output filename for similarity search
        output_dir = 'llm_ft_results'
        similarity_output_filename = f'llm_results_ft_4o_0.8_doi_{attribute}_0_{split}.csv'
        similarity_output_path = os.path.join(output_dir, similarity_output_filename)

        # Load the entire existing CSV to append similarity search results
        try:
            existing_df = pd.read_csv(similarity_output_path)
            logging.info(f"Loaded existing CSV with {len(existing_df)} entries for similarity search.")
        except Exception as e:
            logging.error(f"Error reading existing output file {similarity_output_path}: {e}")
            return

        # Extract processed DOIs from the new entries
        new_dois = new_entries_df['true'].apply(lambda x: json.loads(x)[0]).unique()
        logging.info(f"Performing similarity search on {len(new_dois)} new DOIs.")

        # Iterate over each new DOI and perform similarity search
        for idx, row in tqdm(new_entries_df.iterrows(), total=new_entries_df.shape[0], desc="Performing similarity search"):
            true_entry = row['true']
            restored_entry = row['restored']
            edge_type = row['edge_type']

            # Parse the 'restored' field to extract predicted values
            try:
                restored_data = json.loads(restored_entry)
                # Extract the values for the attribute
                restored_values_list = [item.get('compoundName', '') for item in restored_data]
                # Ensure all extracted values are non-empty strings
                restored_values_list = [str(value) for value in restored_values_list if value]
            except json.JSONDecodeError:
                logging.error(f"JSON decoding failed for 'restored' field: {restored_entry}")
                restored_values_list = []
            except Exception as e:
                logging.error(f"Unexpected error parsing 'restored' field: {e}")
                restored_values_list = []

            # Create a query by joining the predicted values without quotes
            query = ' '.join(restored_values_list)

            # Perform similarity search
            docs_with_score = similarity_search(faiss_index, query, top_k)

            if docs_with_score:
                # Extract the most similar entries and wrap them in double quotes
                similar_entries = [f'"{doc.page_content}"' for doc, _ in docs_with_score]
                # Update the 'restored' field with similar entries
                new_restored_value = [json.loads(true_entry)[0], similar_entries]
                updated_restored_entry = json.dumps(new_restored_value)
            else:
                # If no similar entries found, keep the original restored values
                updated_restored_entry = restored_entry

            # Update the 'restored' field in the existing DataFrame
            doi = json.loads(true_entry)[0]
            existing_df.loc[existing_df['true'] == true_entry, 'restored'] = json.dumps([doi, similar_entries]) if docs_with_score else restored_entry

        # Save the updated DataFrame back to the CSV
        try:
            existing_df.to_csv(similarity_output_path, index=False, quoting=csv.QUOTE_ALL)
            logging.info(f"Similarity search results updated in {similarity_output_path}.")
        except Exception as e:
            logging.error(f"Failed to save updated similarity search results: {e}")

    # ===========================
    # Execute GPT Processing and Similarity Search
    # ===========================

    # Step 1: Resume GPT Processing and get new entries
    new_entries_df = resume_gpt_processing()

    # Step 2: Perform similarity search on new entries
    perform_similarity_search(new_entries_df)

    logging.info(f"Resume GPT Processing and Similarity Search completed for split '{split}'.")

if __name__ == "__main__":
    resume_and_similarity_search(split=split)