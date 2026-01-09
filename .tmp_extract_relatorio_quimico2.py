from pypdf import PdfReader
import re
from pathlib import Path
p = Path(r"1 - OLD/Relatório quimico.pdf")
r = PdfReader(str(p))
txt = "\n".join((page.extract_text() or "") for page in r.pages)
for label in ["limite", "Legenda", "MAPA", "Nitrogênio Total"]:
    m = re.search(label, txt, flags=re.IGNORECASE)
    print("\n===", label, "===")
    if not m:
        print("NO MATCH")
        continue
    s = max(0, m.start()-250)
    e = min(len(txt), m.end()+400)
    print(txt[s:e].replace("\n"," "))
