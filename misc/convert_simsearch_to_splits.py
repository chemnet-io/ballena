import pandas as pd
import os

splits_dir = 'splits'
csv_output_data_location = 'results/SimSearch'
simsearch_file = 'pdf_similarity_results_splitready.csv'

# Ensure the output directory exists
os.makedirs(csv_output_data_location, exist_ok=True)

# Read the simsearch results into a DataFrame
simsearch_df = pd.read_csv(simsearch_file)

# Loop through each file in the splits directory
for file in os.listdir(splits_dir):
    if file.startswith('test_doi_collectionSite_'):
        # Read the test file into a DataFrame
        test_df = pd.read_csv(os.path.join(splits_dir, file))
        
        # Initialize a list to store the results for this test file
        results_list = []
        
        # Process each row in the test DataFrame
        for index, row in test_df.iterrows():
            doi = row['node']
            neighbor = row['neighbor']
            edge_type = row['type']
            
            # Get the list of neighbors from the simsearch results for the same DOI
            restored_neighbors = simsearch_df[simsearch_df['node'] == doi]['neighbor'].tolist()
            
            # Debugging: Print the DOI and its corresponding neighbors
            print(f"DOI: {doi}, Restored Neighbors: {restored_neighbors}")
            
            # Format the 'true' and 'restored' columns
            true_value = f"['{doi}', '{neighbor}']"
            restored_value = f"['{doi}', {restored_neighbors}]"
            
            # Append the result to the results list
            results_list.append({
                'true': true_value,
                'restored': restored_value,
                'edge_type': edge_type
            })
        
        # Convert the results list to a DataFrame
        combined_results = pd.DataFrame(results_list)
        
        # Construct the output file name
        output_file_name = file.replace('test', 'llm_results_simsearch_0.8')
        output_file_path = os.path.join(csv_output_data_location, output_file_name)
        
        # Save the combined results to a new CSV file
        combined_results.to_csv(output_file_path, index=False)