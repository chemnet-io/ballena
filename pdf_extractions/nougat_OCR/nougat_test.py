import os
import torch
from nougat import NougatModel
from nougat.postprocessing import markdown_compatible
from pdf2image import convert_from_path

# Explicitly set the Poppler path
poppler_path = r"I:\Bibliotheken\Dokumente\poppler-24.07.0\Library\bin"
os.environ["PATH"] += os.pathsep + poppler_path

# Path to the directory containing PDF files
pdf_dir = r"I:\Bibliotheken\Dokumente\ballena\nougat_OCR\pdfs"

# Path to the directory containing the downloaded model files
model_dir = r"I:\Bibliotheken\Dokumente\ballena\nougat_OCR\model_base_0.1.0"

# Output directory
output_dir = "nougat_output"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

try:
    # Initialize the Nougat model with the downloaded checkpoint
    model = NougatModel.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).to(torch.bfloat16)  # Explicitly set to bfloat16
    model.eval()

    # Process each PDF in the directory
    for pdf_file in os.listdir(pdf_dir):
        if pdf_file.lower().endswith('.pdf'):
            pdf_path = os.path.join(pdf_dir, pdf_file)
            print(f"Processing {pdf_file}")

            # Convert PDF to images
            images = convert_from_path(pdf_path, poppler_path=poppler_path)

            # List to store processed outputs
            all_pages_output = []

            # Process each page
            for idx, image in enumerate(images):
                print(f"Processing page {idx+1}")
                
                # Run inference
                with torch.no_grad():
                    output = model.inference(image=image)
                
                # Extract the prediction from the output dictionary
                if 'predictions' in output and len(output['predictions']) > 0:
                    prediction = output['predictions'][0]
                    processed_output = markdown_compatible(prediction)
                else:
                    print("No prediction found in the output.")
                    processed_output = str(output)
                
                # Add page number and processed output to the list
                all_pages_output.append(f"## Page {idx+1}\n\n{processed_output}\n\n")

            # Save all pages to a single markdown file named after the original PDF
            output_filename = os.path.splitext(pdf_file)[0] + ".md"
            combined_output_path = os.path.join(output_dir, output_filename)
            with open(combined_output_path, "w", encoding="utf-8") as f:
                f.write(f"# {pdf_file}\n\n")
                f.write("".join(all_pages_output))

            print(f"Processing complete for {pdf_file}. Output saved as {combined_output_path}")

    print(f"All PDFs processed. Outputs saved in {output_dir}")

except Exception as e:
    print(f"An error occurred: {str(e)}")
    import traceback
    traceback.print_exc()
    print("Please check your installation and file paths, then try again.")

# Print Nougat version
import nougat
print(f"Nougat version: {nougat.__version__}")