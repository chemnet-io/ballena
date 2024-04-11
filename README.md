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
## Tables 

Table 1 shows the results from experiments extracting five different natural product properties from biochemical academic papers. They are presented on different values of k to the hits@k metric: (1) name, k = 50; (2) bioactivity, k = 5; (3) specie, k = 50; (4) collection site, k = 20; and (5) isolation type, k = 1. The final k value for each extraction is defined either when a score higher than 0.50 is achieved at any evaluation stage or the upper limit of k = 50.

**Table 1**

Results table for extracting: chemical compound (C), bioactivity (B), specie (S), collection site (L), and isolation type (T). Performance metric with the average and standard deviation of the metric hits@k and k is respectively: 50, 5, 50, 20, and 1.

<table class="tg">
<thead>
  <tr>
    <th class="tg-0pky">Property</th>
    <th class="tg-0pky">Evaluation Stage</th>
    <th class="tg-0pky">GPT4</th>
</thead>
<tbody>
  <tr>
    <td class="tg-0pky" rowspan="4">C</td>
    <td class="tg-0pky">1st</td>
    <td class="tg-0pky">0.25 ± 0.00</td>
  </tr>
  <tr>
    <td class="tg-0pky">2nd</td>
    <td class="tg-0pky">0.25 ± 0.00</td>
  </tr>
  <tr>
    <td class="tg-0pky">3rd</td>
    <td class="tg-0pky">0.22 ± 0.00</td>
  </tr>
  <tr>
    <td class="tg-0pky">4th</td>
    <td class="tg-0pky">0.19 ± 0.00</td>
  </tr>
  <tr>
    <td class="tg-0pky" rowspan="4">B</td>
    <td class="tg-0pky">1st</td>
    <td class="tg-0pky">0.00 ± 0.00</td>
  </tr>
  <tr>
    <td class="tg-0pky">2nd</td>
    <td class="tg-0pky">0.00 ± 0.00</td>
  </tr>
  <tr>
    <td class="tg-0pky">3rd</td>
    <td class="tg-0pky">0.00 ± 0.00</td>
  </tr>
  <tr>
    <td class="tg-0pky">4th</td>
    <td class="tg-0pky">0.00 ± 0.00</td>
  </tr>
  <tr>
    <td class="tg-0pky" rowspan="4">S</td>
    <td class="tg-0pky">1st</td>
    <td class="tg-0pky">0.34 ± 0.00</td>
  </tr>
  <tr>
    <td class="tg-0pky">2nd</td>
    <td class="tg-0pky">0.35 ± 0.00</td>
  </tr>
  <tr>
    <td class="tg-0pky">3rd</td>
    <td class="tg-0pky">0.37 ± 0.00</td>
  </tr>
  <tr>
    <td class="tg-0pky">4th</td>
    <td class="tg-0pky">0.37 ± 0.00</td>
  </tr>
  <tr>
    <td class="tg-0pky" rowspan="4">L</td>
    <td class="tg-0pky">1st</td>
    <td class="tg-0pky">0.00 ± 0.00</td>
  </tr>
  <tr>
    <td class="tg-0pky">2nd</td>
    <td class="tg-0pky">0.00 ± 0.00</td>
  </tr>
  <tr>
    <td class="tg-0pky">3rd</td>
    <td class="tg-0pky">0.00 ± 0.00</td>
  </tr>
  <tr>
    <td class="tg-0pky">4th</td>
    <td class="tg-0pky">0.00 ± 0.00</td>
  </tr>
  <tr>
    <td class="tg-0pky" rowspan="4">T</td>
    <td class="tg-0pky">1st</td>
    <td class="tg-0pky">0.02 ± 0.00</td>
  </tr>
  <tr>
    <td class="tg-0pky">2nd</td>
    <td class="tg-0pky">0.02 ± 0.00</td>
  </tr>
  <tr>
    <td class="tg-0pky">3rd</td>
    <td class="tg-0pky">0.02 ± 0.00</td>
  </tr>
  <tr>
    <td class="tg-0pky">4th</td>
    <td class="tg-0pky">0.01 ± 0.00</td>
  </tr>
</tbody>
</table>

