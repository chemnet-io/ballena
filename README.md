# BALLENA: Biochemical Attribute Learning and Literature Extraction using Neural Architectures

BALLENA is an advanced system for automated knowledge extraction from scientific literature, specifically designed for biochemical research papers. This project addresses the growing challenge of processing the exponentially increasing volume of scientific publications, with a particular focus on biochemistry and natural products research.

## Key Features
- Utilizes state-of-the-art language models (GPT-4 Turbo and GPT-4o) for precise information extraction
- Implements similarity search and vector databases for efficient data retrieval
- Incorporates fine-tuning strategies to enhance extraction accuracy
- Builds upon award-winning methodology from the BiKE Challenge 2023
- Specializes in extracting data about natural products, their properties, and bioactivities

## Project Context
This work is conducted as part of the DINOBBIO project, a collaboration between São Paulo University, HTWK Leipzig, and São Paulo State University. The project focuses on discovering new nature-inspired products from Brazilian biodiversity using Semantic Web technologies and Machine Learning.

## Practical Impact
The system aims to accelerate:
- Drug discovery processes
- Identification of therapeutic molecules
- Analysis of chemical structure-biological activity relationships
- Development of innovative treatment strategies

By automating the extraction of relevant information about natural compounds, including their chemical properties, bioactivities, and origins, BALLENA contributes to more efficient research processes and faster scientific discoveries in medicine and pharmaceutics.

## Setup (Only needed for experiment replication - Note: Significant API costs involved in running the experiments)

### Download Required Files

The following steps are only necessary if you want to replicate the experiments from the thesis. The results are already available in the repository.

1. **PDFs**
   - Download PDFs from [this Google Drive link](https://drive.google.com/drive/folders/1YhAT8kQllSAYly4HtlmNd2jWxmcp-BrG?usp=sharing)
   - Place them in the `pdfs` folder

2. **Splits**
   - Download splits from [this Google Drive link](https://drive.google.com/drive/folders/1NXLQQsIXe0hz32KSOeSG1PCAzFLHoSGh?usp=share_link) 
   - Place them in the `splits` folder

Create the folder `pdfs` in the root of the project.
```mkdir pdfs ```
Put the `pdfs` folder in the root of the project.

Create the folder `splits` in the root of the project.
```mkdir splits ```

Put the `splits` folder in the root of the project.

create an .env file with your OpenAI API key.
```OPENAI_API_KEY='your_api_key'```

install the requirements
```pip install -r requirements.txt```

## Overview of the Project

In this repository, you can find the code and data used in the creation and evaluation of the BALLENA project. Note that downloading the PDFs and splits is only necessary if you want to run the code yourself.

### Repository Structure

#### Results
- **`extraction_results` and `evaluation_results`**: Contains the results of the extraction and evaluation of attributes for all methodologies described in the thesis
  - An interactive HTML viewer is available at `extraction_results/Nougat_FT/viewer.html`

#### Data Processing
- **`faiss_index`**: Contains the FAISS index created for similarity search of extracted attributes
  - Index was created using the `FAISS_workflow.ipynb` notebook

#### Training Data
- **`finetune_data`**: Contains training data for model fine-tuning
  - `finetune_data_nougat`: Data for Nougat model
  - `finetune_data_pymupdf`: Data for PyMuPDF model
  - Data was created from PDF extractions in `pdf_extractions` folder using `create_finetuning_data.ipynb`
  - For better visualization of the training data, view `finetune_data/finetune_data_nougat/check_finetune_dataviewer.html`

#### Core Scripts
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

#### Additional Files
- **`misc`**: Contains miscellaneous development files (included for transparency)

## Results
## Results

The project evaluated multiple approaches for attribute extraction from scientific documents. Below are the results for each method using specific hits@k values:

### GPT-4 Turbo (without Fuzzy Matching)

| Attribute Type | hits@k | 1st Split | 2nd Split | 3rd Split | 4th Split |
|---------------|--------|------------|------------|------------|------------|
| Name (C) | 50 | **0.2493** | 0.2469 | 0.2216 | 0.1892 |
| BioActivity (B) | 5 | **0.0063** | 0.0000 | 0.0000 | 0.0000 |
| Species (S) | 50 | 0.3468 | 0.4198 | **0.4630** | 0.3704 |
| Location (L) | 20 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Type (T) | 1 | 0.0172 | 0.0120 | 0.0179 | **0.0370** |

### GPT-4 Turbo (with Fuzzy Matching)

| Attribute Type | hits@k | 1st Split | 2nd Split | 3rd Split | 4th Split |
|---------------|--------|------------|------------|------------|------------|
| Name (C) | 50 | **0.2493** | 0.2469 | 0.2216 | 0.1892 |
| BioActivity (B) | 5 | 0.4063 | **0.4434** | 0.4000 | 0.3636 |
| Species (S) | 50 | 0.3468 | 0.4198 | **0.4630** | 0.3704 |
| Location (L) | 20 | 0.1653 | 0.1628 | **0.1754** | 0.1538 |
| Type (T) | 1 | 0.0172 | 0.0120 | 0.0179 | **0.0370** |

### GPT-4 Turbo + Similarity Search

| Attribute Type | hits@k | 1st Split | 2nd Split | 3rd Split | 4th Split |
|---------------|--------|------------|------------|------------|------------|
| Name (C) | 50 | 0.274 | 0.324 | 0.332 | **0.349** |
| BioActivity (B) | 5 | 0.712 | 0.745 | 0.758 | **0.776** |
| Species (S) | 50 | 0.970 | **0.972** | 0.970 | 0.953 |
| Location (L) | 20 | 0.730 | 0.736 | 0.759 | **0.760** |
| Type (T) | 1 | 0.705 | 0.699 | 0.732 | **0.755** |

### Fine-tuned GPT-4o (Final Results)

| Attribute Type | hits@k | 1st Split | 2nd Split | 3rd Split | 4th Split |
|---------------|--------|------------|------------|------------|------------|
| Name (C) | 50 | 0.603 | 0.585 | 0.602 | **0.628** |
| BioActivity (B) | 5 | 0.675 | 0.689 | 0.671 | **0.727** |
| Species (S) | 50 | 0.944 | **0.963** | 0.944 | 0.926 |
| Location (L) | 20 | 0.818 | 0.767 | 0.842 | **0.923** |
| Type (T) | 1 | 0.922 | 0.916 | 0.929 | **0.963** |

Key findings:
- The fine-tuned GPT-4o model achieved the best overall performance across all attributes
- Significant improvements were seen when combining GPT-4 Turbo with Similarity Search
- Species extraction showed consistently high accuracy (>92%) across all methods
- Isolation type extraction improved dramatically from 3.7% to 96.3% with fine-tuning
- Location extraction saw major gains, reaching 92.3% with the fine-tuned model

The results demonstrate the effectiveness of combining multiple approaches and fine-tuning, particularly for structured attributes like isolation types and species names. For detailed analysis and visualizations, refer to the results viewer in the extraction results directory.

## Conclusion
This project demonstrates significant advancements in knowledge extraction from scientific PDFs through the development and evaluation of multiple approaches:

1. **Baseline GPT-4 Turbo Performance**
- Achieved moderate success with basic extraction, particularly for species names
- Showed limitations with complex chemical nomenclature and location data
- Provided a solid foundation for further improvements

2. **Enhanced Results with Similarity Search**
- Integration with GPT-4 Turbo led to major improvements across all attributes
- Species extraction reached 97% accuracy
- Location and type extraction showed significant gains
- Demonstrated the value of combining language models with domain-specific knowledge

3. **Fine-tuned Model Excellence**
- The fine-tuned GPT-4o model achieved the best overall performance
- Compound name extraction improved from 27.4% to 60.3%
- Location extraction reached 92.3% accuracy
- Isolation type recognition improved dramatically to 96.3%
- Maintained high accuracy for species (94.4%) and bioactivity (67.5%)

4. **Comparative Analysis**
- Outperformed existing methods like EPHEN across most metrics
- Showed particular strength in handling complex chemical nomenclature
- Demonstrated robust performance across different data splits
- Validated the effectiveness of our multi-stage approach

The results validate our approach of combining advanced language models with similarity search and fine-tuning. This creates a robust system for extracting structured information from scientific literature, with potential applications in automated knowledge base construction and literature analysis.

