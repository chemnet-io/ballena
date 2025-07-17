import json

import pandas as pd

from sentence_transformers import SentenceTransformer

df = pd.read_csv('flat-data.csv')
df = df.dropna()

unique_dict = {'bioActivity': [], 'collectionSite': [], 'collectionSpecie': [], 'collectionType': [], 'name': []}
for key in unique_dict.keys():
    unique_dict[key] = df[key].unique().tolist()

models = [
            'sentence-transformers/all-MiniLM-L6-v2',
            'Qwen/Qwen3-Embedding-0.6B',
            'BAAI/bge-m3',
            'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
            'intfloat/multilingual-e5-large'
        ]

index_dict = {}
for model_name in models:
    print(f"Processing model: {model_name}")
    model = SentenceTransformer(model_name)
    for key in unique_dict.keys():
        sentences = unique_dict[key]
        if key not in index_dict:
            index_dict[key] = {}
        index_dict[key][model_name] = model.encode(sentences).tolist()
        for index, sentence in enumerate(unique_dict[key]):
            index_dict[key][model_name][index] = {"item": sentence, "embedding": index_dict[key][model_name][index]}

with open("index.json", "w") as outfile:
    json.dump(index_dict, outfile, indent=2)