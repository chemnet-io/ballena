import os
import fitz  # PyMuPDF
import pandas as pd
import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def extract_text(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page_num, page in enumerate(doc, start=1): # type: ignore
            page_text = page.get_text("text")
            text += page_text + "\n"

        # Normalize and clean the extracted text
        text = normalize_text(text)

        if not text.strip():
            logging.warning(f"No text extracted from {os.path.basename(pdf_path)}")
            return ""
        
        return text
    except Exception as e:
        logging.error(f"Error processing {os.path.basename(pdf_path)}: {str(e)}")
        return f"Error processing {os.path.basename(pdf_path)}: {str(e)}"

def normalize_text(text):
    # Replace multiple line breaks with a space
    text = re.sub(r'\n+', ' ', text)
    
    # Remove non-printable/control characters except common punctuation
    # Retain newline and carriage return if needed
    text = ''.join(c for c in text if c.isprintable() or c in ['\n', '\r', '\t', '.', ',', ';', ':', '!', '?', '-', '(', ')'])

    # Remove specific problematic Unicode characters
    text = re.sub(r'[\u0000-\u001F\u007F-\u009F]', '', text)

    # Replace multiple spaces with a single space
    text = re.sub(r' +', ' ', text)

    # Strip leading and trailing whitespace
    text = text.strip()

    return text

def main():
    # Step 1: Define input and output directories
    input_dir = "pdfs"
    output_dir = "pypdfextraction"
    
    # Step 2: Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 3: Initialize data structures
    data = []
    empty_files = []
    problematic_files = []
    
    # Step 4: Process all PDF files in the input directory
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(input_dir, filename)
            logging.info(f"Processing {filename}...")
            text = extract_text(pdf_path)
            
            if not text or text.startswith("Error processing"):
                empty_files.append(filename)
            
            # Check for specific problematic characters
            if re.search(r'[\u0004\u0010]', text):
                problematic_files.append(filename)
            
            data.append({"filename": filename, "text": text})
    
    # Step 5: Create a DataFrame and save as Parquet
    df = pd.DataFrame(data)
    parquet_path = os.path.join(output_dir, "extracted_text.parquet")
    df.to_parquet(parquet_path, index=False)
    
    logging.info(f"Extracted text from all PDFs and saved to {parquet_path}")
    logging.info(f"Number of files with empty text or errors: {len(empty_files)}")
    
    if empty_files:
        logging.info("Files with empty text or errors:")
        for file in empty_files[:10]:  # Print first 10 empty files
            logging.info(f" - {file}")
        if len(empty_files) > 10:
            logging.info(f"... and {len(empty_files) - 10} more")
    
    logging.info(f"Problematic files: {problematic_files}")

if __name__ == "__main__":
    main()