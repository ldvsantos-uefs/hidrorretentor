
from pathlib import Path

bib_file = Path(r"c:\Users\vidal\OneDrive\Documentos\13 - CLONEGIT\artigo-posdoc\2 - HIDRORRETENTOR\3 - MANUSCRITO\1-MARKDOWN\1-MANUSCRITOS\referencias.bib")

try:
    # Read binary
    content_bytes = bib_file.read_bytes()
    # Decode with replacement
    content_str = content_bytes.decode('utf-8', errors='replace')
    # Write back as proper UTF-8
    bib_file.write_text(content_str, encoding='utf-8')
    print(f"File repaired. Size: {len(content_str)} chars.")
except Exception as e:
    print(f"Error: {e}")
