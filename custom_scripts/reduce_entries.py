import pandas as pd

# Load the CSV file
df = pd.read_csv('pdf_similarity_results_improved_syntax.csv')

# Group by 'node' and then aggregate by the most common 'neighbor'
def most_common_neighbor(series):
    return series.value_counts().idxmax()

# Apply the aggregation
result = df.groupby('node').agg({
    'neighbor': most_common_neighbor,
    'type': 'first'  # Assuming 'type' remains constant per 'node'
})

# Reset index to turn the grouped 'node' back into a column
result.reset_index(inplace=True)

# Save the result back to a CSV
result.to_csv('pdf_similarity_results_reduced.csv', index=False)