# BALLENA: Biochemical Attribute Learning and Literature Extraction using Neural Architectures

BALLENA is an advanced system for automated knowledge extraction from scientific literature, specifically designed for biochemical research papers. This project addresses the growing challenge of processing the exponentially increasing volume of scientific publications, with a particular focus on biochemistry and natural products research.

## Key Features
- Utilizes state-of-the-art proprietary (GPT-4o) and open-source (Llama-3.1, Phi-4, Qwen-2.5) for precise information extraction
- Implements similarity search and vector databases for efficient data retrieval
- Incorporates fine-tuning strategies to enhance extraction accuracy
- Builds upon award-winning methodology from the BiKE Challenge 2023 and 2025
- Specializes in extracting data about natural products, their properties, and bioactivities

## Practical Impact
The system aims to accelerate:
- Drug discovery processes
- Identification of therapeutic molecules
- Analysis of chemical structure-biological activity relationships
- Development of innovative treatment strategies

By automating the extraction of relevant information about natural compounds, including their chemical properties, bioactivities, and origins, BALLENA contributes to more efficient research processes and faster scientific discoveries in medicine and pharmaceutics.

## Overview of the Project

In this repository, you can find the code and data used in the creation and evaluation of the BALLENA project. Note that downloading the PDFs and splits is only necessary if you want to run the code yourself.

### Repository Structure

#### Evaluation GPT models
- **`extraction_results` and `evaluation_results`**: Contains the results of the extraction and evaluation of attributes for all methodologies described in the thesis
  - An interactive HTML viewer is available at `extraction_results/Nougat_FT/viewer.html`

#### Similarity Search GPT models
- **`faiss_index`**: Contains the FAISS index created for similarity search of extracted attributes
  - Index was created using the `FAISS_workflow.ipynb` notebook

#### Finetune Data GPT models
- **`finetune_data`**: Contains training data for model fine-tuning
  - `finetune_data_nougat`: Data for Nougat model
  - `finetune_data_pymupdf`: Data for PyMuPDF model
  - Data was created from PDF extractions in `pdf_extractions` folder using `create_finetuning_data.ipynb`
  - For better visualization of the training data, view `finetune_data/finetune_data_nougat/check_finetune_dataviewer.html`

#### Core Scripts GPT models
- **`BALLENA.py`**: Processes all attributes except names
  - Handles extraction of bioActivity, collectionSite, collectionSpecie, and collectionType attributes
  - Uses fine-tuned GPT-4o models for each attribute type
  - Performs similarity search using FAISS indices to find closest matches
  - Calculates evaluation metrics like Hits@k and Mean Reciprocal Rank (MRR)
  - Supports test mode for quick validation and split-based processing
  - Saves results and evaluation metrics to CSV files
- **`BALLENA_name.py`**: Specialized script for processing name attributes
  - Required due to complex nomenclature that needed special handling
  - Both scripts support:
    - Processing selected splits
    - Testing mode for specified number of rows

#### Additional Files GPT models
- **`misc`**: Contains miscellaneous development files (included for transparency)

## Results

The model was evaluated using Hits@k on the test sets of the NatUKE Benchmark (do Carmo et al. 2023). Below are the results for each method using specific hits@k values:

### Pre-trained Llama-3.1

| Attribute Type | hits@k | 1st Split | 2nd Split | 3rd Split | 4th Split |
|---------------|--------|------------|------------|------------|------------|
| Name (C) | 50 | 0.05 | 0.05 | 0.06 | 0.07 |
| BioActivity (B) | 5 | 0.00 | 0.00 | 0.00 | 0.00 |
| Species (S) | 50 | 0.10 | 0.09 | 0.09 | 0.09 |
| Location (L) | 20 | 0.01 | 0.01 | 0.01 | 0.01 |
| Type (T) | 1 | 0.00 | 0.00 | 0.00 | 0.00 |

### Pre-trained Llama-3.1 with Similarity Search

| Attribute Type | hits@k | 1st Split | 2nd Split | 3rd Split | 4th Split |
|---------------|--------|------------|------------|------------|------------|
| Name (C) | 50 | 0.49 | 0.51 | 0.50 | 0.50 |
| BioActivity (B) | 5 | 0.44 | 0.45 | 0.44 | 0.45 |
| Species (S) | 50 | 0.52 | 0.52 | 0.50 | 0.49 |
| Location (L) | 20 | 0.30 | 0.31 | 0.32 | 0.31 |
| Type (T) | 1 | 0.81 | 0.82 | 0.83 | 0.84 |

### Fine-tuned Llama-3.1 with Similarity Search

| Attribute Type | hits@k | 1st Split | 2nd Split | 3rd Split | 4th Split |
|---------------|--------|------------|------------|------------|------------|
| Name (C) | 50 | 0.70 | 0.72 | 0.73 | 0.74 |
| BioActivity (B) | 5 | 0.74 | 0.74 | 0.76 | 0.77 |
| Species (S) | 50 | 0.86 | 0.88 | 0.88 | 0.86 |
| Location (L) | 20 | 0.63 | 0.63 | 0.64 | 0.65 |
| Type (T) | 1 | 0.91 | 0.92 | 0.93 | 0.94 |

### Pre-trained Qwen-2.5

| Attribute Type | hits@k | 1st Split | 2nd Split | 3rd Split | 4th Split |
|---------------|--------|------------|------------|------------|------------|
| Name (C) | 50 | 0.09 | 0.09 | 0.08 | 0.10 |
| BioActivity (B) | 5 | 0.06 | 0.06 | 0.05 | 0.06 |
| Species (S) | 50 | 0.37 | 0.39 | 0.41 | 0.38 |
| Location (L) | 20 | 0.01 | 0.01 | 0.01 | 0.01 |
| Type (T) | 1 | 0.00 | 0.00 | 0.00 | 0.00 |

### Pre-trained Qwen-2.5 with Similarity Search

| Attribute Type | hits@k | 1st Split | 2nd Split | 3rd Split | 4th Split |
|---------------|--------|------------|------------|------------|------------|
| Name (C) | 50 | 0.81 | 0.82 | 0.81 | 0.81 |
| BioActivity (B) | 5 | 0.71 | 0.72 | 0.74 | 0.72 |
| Species (S) | 50 | 0.86 | 0.86 | 0.86 | 0.86 |
| Location (L) | 20 | 0.61 | 0.62 | 0.64 | 0.66 |
| Type (T) | 1 | 0.79 | 0.80 | 0.81 | 0.84 |

### Fine-tuned Qwen-2.5 with Similarity Search

| Attribute Type | hits@k | 1st Split | 2nd Split | 3rd Split | 4th Split |
|---------------|--------|------------|------------|------------|------------|
| Name (C) | 50 | 0.62 | 0.64 | 0.65 | 0.67 |
| BioActivity (B) | 5 | 0.71 | 0.73 | 0.75 | 0.76 |
| Species (S) | 50 | 0.89 | 0.91 | 0.92 | 0.92 |
| Location (L) | 20 | 0.62 | 0.63 | 0.64 | 0.66 |
| Type (T) | 1 | 0.92 | 0.92 | 0.93 | 0.94 |

### Pre-trained Phi-4

| Attribute Type | hits@k | 1st Split | 2nd Split | 3rd Split | 4th Split |
|---------------|--------|------------|------------|------------|------------|
| Name (C) | 50 | 0.08 | 0.08 | 0.08 | 0.09 |
| BioActivity (B) | 5 | 0.02 | 0.02 | 0.02 | 0.01 |
| Species (S) | 50 | 0.22 | 0.23 | 0.24 | 0.23 |
| Location (L) | 20 | 0.01 | 0.01 | 0.01 | 0.01 |
| Type (T) | 1 | 0.00 | 0.00 | 0.00 | 0.00 |

### Pre-trained Phi-4 with Similarity Search

| Attribute Type | hits@k | 1st Split | 2nd Split | 3rd Split | 4th Split |
|---------------|--------|------------|------------|------------|------------|
| Name (C) | 50 | 0.85 | 0.87 | 0.87 | 0.89 |
| BioActivity (B) | 5 | 0.72 | 0.74 | 0.75 | 0.74 |
| Species (S) | 50 | 0.90 | 0.90 | 0.90 | 0.89 |
| Location (L) | 20 | 0.61 | 0.63 | 0.64 | 0.66 |
| Type (T) | 1 | 0.66 | 0.65 | 0.66 | 0.65 |

### Fine-tuned Phi-4 with Similarity Search

| Attribute Type | hits@k | 1st Split | 2nd Split | 3rd Split | 4th Split |
|---------------|--------|------------|------------|------------|------------|
| Name (C) | 50 | 0.65 | 0.68 | 0.69 | 0.70 |
| BioActivity (B) | 5 | 0.73 | 0.74 | 0.75 | 0.74 |
| Species (S) | 50 | 0.83 | 0.84 | 0.86 | 0.87 |
| Location (L) | 20 | 0.61 | 0.61 | 0.63 | 0.64 |
| Type (T) | 1 | 0.91 | 0.92 | 0.93 | 0.94 |

### Pre-trained GPT-4o

| Attribute Type | hits@k | 1st Split | 2nd Split | 3rd Split | 4th Split |
|---------------|--------|------------|------------|------------|------------|
| Name (C) | 50 | 0.00 | 0.00 | 0.00 | 0.00 |
| BioActivity (B) | 5 | 0.03 | 0.04 | 0.03 | 0.03 |
| Species (S) | 50 | 0.19 | 0.20 | 0.24 | 0.22 |
| Location (L) | 20 | 0.00 | 0.00 | 0.00 | 0.00 |
| Type (T) | 1 | 0.00 | 0.00 | 0.00 | 0.00 |

### Pre-trained GPT-4o with Similarity Search

| Attribute Type | hits@k | 1st Split | 2nd Split | 3rd Split | 4th Split |
|---------------|--------|------------|------------|------------|------------|
| Name (C) | 50 | 0.70 | 0.74 | 0.74 | 0.81 |
| BioActivity (B) | 5 | 0.74 | 0.77 | 0.76 | 0.76 |
| Species (S) | 50 | 1.0 | 1.0 | 1.0 | 1.0 |
| Location (L) | 20 | 0.69 | 0.67 | 0.75 | 0.77 |
| Type (T) | 1 | 0.15 | 0.14 | 0.14 | 0.07 |

### Fine-tuned GPT-4o with Similarity Search

| Attribute Type | hits@k | 1st Split | 2nd Split | 3rd Split | 4th Split |
|---------------|--------|------------|------------|------------|------------|
| Name (C) | 50 | 0.60 | 0.59 | 0.60 | 0.63 |
| BioActivity (B) | 5 | 0.68 | 0.68 | 0.67 | 0.73 |
| Species (S) | 50 | 0.94 | 0.96 | 0.94 | 0.92 |
| Location (L) | 20 | 0.82 | 0.77 | 0.84 | 0.92 |
| Type (T) | 1 | 0.92 | 0.92 | 0.93 | 0.96 |

---

Do Carmo, Paulo Viviurka, et al. "NatUKE: A Benchmark for Natural Product Knowledge Extraction from Academic Literature." 2023 IEEE 17th International Conference on Semantic Computing (ICSC). IEEE, 2023.

## Hugging Face fine-tuned models

The following available fine-tuned single-task models were fine-tuned using all the data available in the original NatUKE dataset. Therefore, they are not suitable for evaluation in the original benchmark, however they should have better performance in the real-world.

* [`Bike-name`](https://huggingface.co/aksw/Bike-name) is a Medium fine-tuned language model designed to **extract biochemical names from scientific text articles**. It is ideal for Information Retrieval systems based on Biohemical Knowledge Extraction.
* [`Bike-bioactivity`](https://huggingface.co/aksw/Bike-bioactivity) is a Medium fine-tuned language model designed to **extract biochemical biological activities from scientific text articles**. It is ideal for Information Retrieval systems based on Biohemical Knowledge Extraction.
* [`Bike-specie`](https://huggingface.co/aksw/Bike-specie) is a Medium fine-tuned language model designed to **extract biochemical collection species from scientific text articles**. It is ideal for Information Retrieval systems based on Biohemical Knowledge Extraction.
* [`Bike-site`](https://huggingface.co/aksw/Bike-site) is a Medium fine-tuned language model designed to **extract biochemical collection sites from scientific text articles**. It is ideal for Information Retrieval systems based on Biohemical Knowledge Extraction.
* [`Bike-isolation`](https://huggingface.co/aksw/Bike-isolation) is a Medium fine-tuned language model designed to **extract biochemical isolation types from scientific text articles**. It is ideal for Information Retrieval systems based on Biohemical Knowledge Extraction.

## Citation

If you use this code or any of the fine-tuned models in your work, please cite it as:

```
@inproceedings{ref:doCarmo2025,
  title={Improving Natural Product Knowledge Extraction from Academic Literature with Enhanced PDF Text Extraction and Large Language Models},
  author={Viviurka do Carmo, Paulo and Silva G{\^o}lo, Marcos Paulo and Gwozdz, Jonas and Marx, Edgard and Marcondes Marcacini, Ricardo},
  booktitle={Proceedings of the 40th ACM/SIGAPP Symposium on Applied Computing},
  pages={980--987},
  year={2025}
}
```
