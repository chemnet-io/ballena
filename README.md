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

The project evaluated multiple approaches for attribute extraction from scientific documents. Below are the results for each method using specific hits@k values:

### Fine-tuned GPT-4o with Similarity Search

| Attribute Type | hits@k | 1st Split | 2nd Split | 3rd Split | 4th Split |
|---------------|--------|------------|------------|------------|------------|
| Name (C) | 50 | 0.603 | 0.585 | 0.602 | **0.628** |
| BioActivity (B) | 5 | 0.675 | 0.689 | 0.671 | **0.727** |
| Species (S) | 50 | 0.944 | **0.963** | 0.944 | 0.926 |
| Location (L) | 20 | 0.818 | 0.767 | 0.842 | **0.923** |
| Type (T) | 1 | 0.922 | 0.916 | 0.929 | **0.963** |
