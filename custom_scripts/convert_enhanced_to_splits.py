import os
import csv

input_directory = 'results/LLM_enhanced'
output_directory = 'natuke_test_ready'

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

for filename in os.listdir(input_directory):
    if filename.endswith('.csv'):
        input_file = os.path.join(input_directory, filename)
        output_file = os.path.join(output_directory, filename.replace('llm_results_gpt4_0.8', 'test'))
        with open(input_file, 'r', newline='', encoding='utf-8') as infile, open(output_file, 'w', newline='', encoding='utf-8') as outfile:
            reader = csv.reader(infile)
            writer = csv.writer(outfile)
            
            # Write the header
            writer.writerow(['node', 'neighbor', 'type'])
            
            # Skip the header of the input file
            next(reader)
            
            for row in reader:
                restored_data = eval(row[1])
                node = restored_data[0]
                neighbors = restored_data[1]
                edge_type = row[2]
                
                for neighbor in neighbors:
                    writer.writerow([node, neighbor, edge_type])

print("Transformation completed.")