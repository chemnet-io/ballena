import os
import pandas as pd
import pyarrow.parquet as pq

# Define the directory containing the .md files
directory = 'nougat_OCR/nougat_output'

# Initialize lists to store filenames and their corresponding text
filenames = []
texts = []

# Iterate through all files in the directory
for filename in os.listdir(directory):
    # Check if the file has a .md extension
    if filename.endswith('.md'):
        # Construct the full file path
        file_path = os.path.join(directory, filename)
        
        # Read the content of the .md file
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Replace the .md extension with .pdf
        pdf_filename = os.path.splitext(filename)[0] + '.pdf'
        
        # Append to the lists
        filenames.append(pdf_filename)
        texts.append(content)

# Create a pandas DataFrame with the collected data
df = pd.DataFrame({
    'filename': filenames,
    'text': texts
})

# Optionally convert columns to bytes
# df['filename'] = df['filename'].astype(bytes)
# df['text'] = df['text'].astype(bytes)

# Define the output Parquet file path
output_parquet = 'output.parquet'

# Save the DataFrame to a Parquet file
df.to_parquet(output_parquet, engine='pyarrow', compression='snappy', row_group_size=392)

# Verify the Parquet file
df_parquet = pd.read_parquet(output_parquet)
print("First few rows of the DataFrame:")
print(df_parquet.head())

# Inspect Parquet metadata
parquet_file = pq.ParquetFile(output_parquet)
print(f"\nNumber of row groups: {parquet_file.num_row_groups}")

for i, column in enumerate(parquet_file.schema):
    print(f"\nColumn: {column.name}")
    print(f"  Type: {column.physical_type}")