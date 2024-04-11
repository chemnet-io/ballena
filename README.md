# ballena

## Description
This project shall enchance the process of extracting chemical compounds from scientific papers. 

## Setup
Get the pdfs from the following link and put them in the `pdfs` folder:
https://drive.google.com/drive/folders/1YhAT8kQllSAYly4HtlmNd2jWxmcp-BrG?usp=sharing

Create the folder `pdfs` in the root of the project.
```mkdir pdfs ```
Put the `pdfs` folder in the root of the project.

Change the env.example file to .env and fill the variables with the correct values.

### Preparing new files for performance evaluation

The files must follow a naming convention, for example `knn_results_deep_walk_0.8_doi_bioActivity_0_2nd.csv`: 
* `knn_results` is the name of the experiments batch used in natuke;
* `deep_walk` is the name of the algorithm;
* `0.8` represents the maximum train/test splits percentage in the evaluation stages;
* `doi_bioActivity` is the edge_type restored for evaluation;
* `0` is the random sampling for the splits;
* and `2nd` is the evaluation_stage.

All these parameters can be changed as needed in this portion of the `dynamic_benchmark_evaluation.py`
```
path = 'path-to-data-repository'
file_name = "knn_results"
splits = [0.8]
#edge_groups = ['doi_name', 'doi_bioActivity', 'doi_collectionSpecie', 'doi_collectionSite', 'doi_collectionType']
edge_group = 'doi_collectionType'
#algorithms = ['bert', 'deep_walk', 'node2vec', 'metapath2vec', 'regularization']
algorithms = ['deep_walk', 'node2vec', 'metapath2vec', 'regularization']
k_at = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
dynamic_stages = ['1st', '2nd', '3rd', '4th']
```
