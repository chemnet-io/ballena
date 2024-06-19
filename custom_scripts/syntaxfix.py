import pandas as pd

# Specify the single CSV file to process
input_filepath = 'pdf_similarity_results.csv'
output_filepath = 'pdf_similarity_results_improved_syntax.csv'

# Read the CSV file
df = pd.read_csv(input_filepath)

# Update the 'restored' column
df['neighbor'] = df['neighbor'].str.replace(' ,', '/')

# Save the updated DataFrame
df.to_csv(output_filepath, index=False)

print("The file has been processed and saved to the improved syntax directory.")
