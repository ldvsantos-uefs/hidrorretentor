from pypdf import PdfReader
import re
from pathlib import Path
p = Path(r"1 - OLD/Relatório quimico.pdf")
r = PdfReader(str(p))
txt = "\n".join((page.extract_text() or "") for page in r.pages)
print("PAGES", len(r.pages))
terms = ["Nitrog", "N total", "Kjeldahl", "limite", "detec", "LD", "ITPS", "MAPA", "Fósforo", "Potássio", "Magnésio", "Cálcio"]
for t in terms:
    print(t, "FOUND" if re.search(t, txt, flags=re.IGNORECASE) else "MISS")
print("\n--- FIRST MATCH AROUND 'Nitrog' ---")
m = re.search(r"Nitrog.{0,400}", txt, flags=re.IGNORECASE|re.DOTALL)
if m:
    print(m.group(0).replace("\n", " "))
else:
    print("NO MATCH")
