import fitz  # Import PyMuPDF for OCR
import openai
import json

# Ensure you have your OpenAI API key set in your environment variables or passed securely
openai.api_key = 'your-api-key'

# Function to check token count and slice into batches if necessary
def slice_into_batches(text, max_tokens=128000):
    # Tokenize the text using a simple space-based approximation
    tokens = text.split()
    batches = []
    
    current_batch = []
    current_count = 0
    for token in tokens:
        current_batch.append(token)
        current_count += 1
        # When the current batch reaches the max token count, add it to batches
        if current_count >= max_tokens:
            batches.append(' '.join(current_batch))
            current_batch = []
            current_count = 0
    # Add the last batch if it has any tokens
    if current_batch:
        batches.append(' '.join(current_batch))
    
    return batches

def ocr_pdf_by_chapters(pdf_path):
    chapters_text = {}
    with fitz.open(pdf_path) as pdf:
        toc = pdf.get_toc(simple=False)  # Get the table of contents
        for chapter in toc:
            chapter_title = chapter[1]
            start_page = chapter[2]
            # Assuming each chapter is one whole section without subchapters
            end_page = start_page + 1
            # Find the end page of the chapter
            for next_chapter in toc[toc.index(chapter)+1:]:
                if next_chapter[0] == chapter[0]:  # Same level of chapter
                    end_page = next_chapter[2]
                    break
            # Extract text for the chapter
            chapter_text = ""
            for page_num in range(start_page, end_page):
                page = pdf[page_num]
                chapter_text += page.get_text()
            chapters_text[chapter_title] = chapter_text
    return chapters_text

# Function to process text with GPT-4 and return JSON
def process_with_gpt4(text):
    # Check if the input surpasses the token limit and slice into batches
    batches = slice_into_batches(text)
    responses = []
    for batch in batches:
        response = openai.Completion.create(
            model="gpt-4-1106-preview",  # Use the specified GPT-4 model
            prompt=batch,
            max_tokens=750,
            n=1,
            stop=None,
            temperature=0.7,
            top_p=1,
            presence_penalty=0,
            frequency_penalty=0,
            best_of=1,
            user="user-id",  # Replace with your user ID if necessary
            logprobs=None,
            echo=False,
            return_prompt=False,
            return_metadata=False,
            expand=["completion"],
            logit_bias={},
            type="json_object"  # Set the output type to JSON object
        )
        responses.append(response)
    # Combine the responses or handle them as needed
    return json.dumps(responses, indent=2)

# Main function to run the script
def main(pdf_path):
    chapters_text = ocr_pdf_by_chapters(pdf_path)
    for chapter_title, text in chapters_text.items():
        print(f"Processing chapter: {chapter_title}")
        json_output = process_with_gpt4(text)
        print(json_output)

if __name__ == "__main__":
    pdf_path = 'path_to_your_pdf.pdf'  # Replace with your PDF file path
    main(pdf_path)