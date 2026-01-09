
import chardet
from pathlib import Path

bib_file = Path(r"c:\Users\vidal\OneDrive\Documentos\13 - CLONEGIT\artigo-posdoc\2 - HIDRORRETENTOR\3 - MANUSCRITO\1-MARKDOWN\1-MANUSCRITOS\referencias.bib")

# Try to detect encoding or fallback to latin1 then visual inspection
try:
    raw_data = bib_file.read_bytes()
    result = chardet.detect(raw_data)
    encoding = result['encoding']
    print(f"Detected encoding: {encoding}")
    
    if encoding is None:
        encoding = 'latin1'
        
    content = raw_data.decode(encoding)
    
except UnicodeDecodeError:
    print("Decoding failed, trying latin1 fallback")
    content = bib_file.read_text(encoding='latin1', errors='replace')
except Exception as e:
    print(f"Error: {e}")
    # Last resort: load as binary and save as utf-8 ignoring errors
    content = bib_file.read_bytes().decode('utf-8', errors='replace')

# Save as UTF-8
bib_file.write_text(content, encoding='utf-8')
print("File converted to UTF-8")
