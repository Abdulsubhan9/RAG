import os
import pdfplumber

def extract_text_from_pdf(pdf_path, output_txt):
    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
        return

    all_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                all_text += page_text + "\n"

    with open(output_txt, "w", encoding="utf-8") as f:
        f.write(all_text)
    print(f"Extracted and saved: {output_txt}")

# For English and Spanish PDFs
extract_text_from_pdf("english.pdf", "english_text.txt")
extract_text_from_pdf("es.pdf", "es_text.txt")
