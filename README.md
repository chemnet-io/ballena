
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
