
import fitz  # PyMuPDF

pdf_path = r"c:\Users\vidal\OneDrive\Documentos\13 - CLONEGIT\artigo-posdoc\2 - HIDRORRETENTOR\1 - OLD\Relatório quimico.pdf"

try:
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    print("--- START OF PDF CONTENT ---")
    print(text)
    print("--- END OF PDF CONTENT ---")
except Exception as e:
    print(f"Error reading PDF: {e}")
