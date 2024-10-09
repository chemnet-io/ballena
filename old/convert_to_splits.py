import re
import pandas as pd
import os
import json

splits = 'splits'
csv_output_data_location = 'results/SimSearch'

def extract_edge_groups(text, edge_type):
    try:
        data = json.loads(text)
        bioactivities = []
        for molecule_key, molecule_info in data.items():
            bioactivity = molecule_info.get(edge_type)
            if bioactivity:
                bioactivities.append((bioactivity))
        print(f"Extracted bioactivities: {bioactivities}")
        return bioactivities
    except Exception as e:
        print(f"Error extracting bioactivity: {e}")
        return []

def extract_result(gpt_input: json, split):
    print(gpt_input)
    split['true'] = split.apply(lambda row: f"['{row['node']}', '{row['neighbor']}']", axis=1)
    edge_type = split.iloc[0, 2]
    edge_type = re.sub(r'^doi_', '', edge_type)
    gpt_input['text'] = gpt_input.apply(lambda row: extract_edge_groups(row['text'], edge_type), axis=1)
    gpt_input = gpt_input.groupby('doi')['text'].agg(list).reset_index()

    gpt_input['text'] = gpt_input.apply(lambda row: [row['doi'], row['text'][0] if row['text'] else []], axis=1)

    merged_df = pd.merge(gpt_input, split, left_on='doi', right_on='node', how='inner')

    merged_df = merged_df[['true', 'text', 'type']]
    merged_df = merged_df.rename(columns={'text': 'restored', 'type': 'edge_type'})

    pd.DataFrame.to_csv(merged_df, csv_output_data_location + f"llm_results_gpt4_0.8_{filename_without_test}", index=False)

for file in os.listdir(splits):
    gpt_input = pd.read_csv("path to gpt input here")
    gpt_input['doi'] = gpt_input['doi'].str.replace('@', '/')
    gpt_input = gpt_input.drop(labels='file_name', axis=1)
    filename = os.fsdecode(file)
    filename_without_test = filename.replace("test_", "")
    split = pd.read_csv(f'{csv_output_data_location}{filename}')
    extract_result(gpt_input, split)




