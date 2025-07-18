import ast
import json
import multiprocess

from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

with open("index.json", "r") as file:
    index_dict = json.load(file)

# tasks = ['bioActivity', 'collectionSite', 'collectionSpecie', 'collectionType']
# evaluation_stages = ['1st', '2nd', '3rd', '4th']
# model_types = ['pre-trained', 'finetuning']
model_names = ['qwen14b: ', 'llama8b: ', 'phi14b: ']
embedding_models = [
            'sentence-transformers/all-MiniLM-L6-v2',
            'Qwen/Qwen3-Embedding-0.6B',
            'BAAI/bge-m3',
            'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
            'intfloat/multilingual-e5-large'
        ]

tasks = ['bioActivity']
evaluation_stages = ['1st']
model_types = ['pre-trained']

def get_knn_data(index_dict, task, embedding_model):
    item_list, vector_list = [], []
    for item_vector_dict in index_dict[task][embedding_model]:
        item_list.append(item_vector_dict["item"])
        vector_list.append(item_vector_dict["embedding"])

    return item_list, vector_list

def run_one_nn(vector_pred, item_list, vector_list):
    knn = NearestNeighbors(n_neighbors=1, metric='cosine')
    knn.fit(vector_list)
    indice = knn.kneighbors(vector_pred.reshape(1, -1), return_distance=False)
    return item_list[indice[0][0]]

def similarity_run(task, stage, fold, model_type, embedding_model, item_list, vector_list, pred_list, n_jobs=4):
    def process(start, end, task, stage, fold, model_type, embedding_model, item_list, vector_list, pred_list):
        model = SentenceTransformer(embedding_model)
        ss_pred_list = []
        for pred_l in tqdm(pred_list[start:end]):
            ss_pred_l = []
            for pred in pred_l:
                vector_pred = model.encode(pred)
                ss_pred_l.append(run_one_nn(vector_pred, item_list, vector_list))
            ss_pred_list.append(ss_pred_l)
        with open(f"{model_type}/{task}_{stage}_{fold}_{embedding_model.split('/')[1]}_ss", 'a', encoding='utf-8') as f:
            f.write(str(model_name) + ': ' + str(ss_pred_list) + '\n')

    def split_processing(task, stage, fold, model_type, embedding_model, item_list, vector_list, pred_list, n_jobs):
        split_size = round(len(pred_list) / n_jobs)
        threads = []                                                                
        for i in range(n_jobs):                                                 
            start = i * split_size                                                  
            end = len(pred_list) if i+1 == n_jobs else (i+1) * split_size                
            threads.append(                                                         
                multiprocess.Process(target=process, args=(start, end, task, stage, fold, model_type, embedding_model, item_list, vector_list, pred_list)))
            threads[-1].start()            

        for t in threads:
            t.join()
    
    split_processing(task, stage, fold, model_type, embedding_model, item_list, vector_list, pred_list, n_jobs)

for embedding_model in embedding_models:
    print(f"Processing embedding model: {embedding_model}")
    for task in tasks:
        print(f"Processing task: {task}")
        item_list, vector_list = get_knn_data(index_dict, task, embedding_model)
        for model_type in model_types:
            print(f"Processing model type: {model_type}")
            for stage in evaluation_stages:
                print(f"Processing stage: {stage}")
                for fold in range(1):
                    print(f"Processing fold: {fold}")
                    with open(f"{model_type}/{task}_{stage}_{fold}", 'r', encoding='utf-8') as f:
                        for line in f:
                            for model_name in model_names:
                                if model_name in line:
                                    split_string = model_name
                                    break
                            split_line = line.split(split_string)
                            model_name = split_line[0]
                            pred_list = ast.literal_eval(split_line[1])
                            similarity_run(task, stage, fold, model_type, embedding_model, item_list, vector_list, pred_list, n_jobs=4)