import os
from PyPDF2 import PdfReader
from pdfminer.high_level import extract_text

def pdf_to_text(pdf_path):
    try:
        # Use PyPDF2 for text extraction
        with open(pdf_path, 'rb') as file:
            pdf_reader = PdfReader(file)
            text = ''
            for page in pdf_reader.pages:
                #page = pdf_reader.getPage(page_number)
                text += page.extract_text()
        return text
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return None

def process_pdfs(input_dir, output_dir):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Process each PDF file in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(input_dir, filename)
            txt_path = os.path.join(output_dir, os.path.splitext(filename)[0] + ".txt")

            # Convert PDF to text
            text = pdf_to_text(pdf_path)

            if text is not None:
                # Save text to corresponding .txt file
                with open(txt_path, 'w', encoding='utf-8') as txt_file:
                    txt_file.write(text)

                print(f"Text extracted from {pdf_path} and saved to {txt_path}")

if __name__ == "__main__":
    input_directory = 'paper-pdfs/'
    output_directory = 'authors-test/'

    process_pdfs(input_directory, output_directory)
