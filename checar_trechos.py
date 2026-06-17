import re
from pathlib import Path

def norm(s):
    s = (s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
           .replace("–", "-").replace("—", "-").replace("­", "").replace("ﬁ","fi").replace("ﬂ","fl"))
    s = re.sub(r"-\s*\n\s*", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip().lower()

def loose(s):
    s = norm(s)
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

# (arquivo, rotulo, trecho-a-procurar)
casos = [
    ("alnaabi2026", "contrib2-sent2 (verbatim?)",
     "Anomaly detection, Arabic natural language processing, and explainable AI emerged as the most contextually relevant techniques"),
    ("buczak2016", "problema-sent2 (verbatim?)",
     "The main disadvantage of anomaly-based techniques is the potential for high false alarm rates"),
    ("aljaradat2026", "contrib2 'sustained use of digital payment systems'?",
     "to support sustained use of digital payment systems"),
    ("shama2026", "contrib2 'first comprehensive scientometric mapping...research'?",
     "the first comprehensive scientometric mapping of neuro-symbolic AI research in cybersecurity"),
    ("molina2017", "objetivo1 trecho 'most popular and relevant tools'?",
     "the most popular and relevant tools available for risk assessment"),
    ("molina2017", "metod0 'properties, metrics and strategies'?",
     "their properties, metrics and strategies are analysed"),
]

for stem, rot, frase in casos:
    txt = Path(f"saida/{stem}.txt").read_text(encoding="utf-8", errors="ignore")
    c = norm(txt); cl = loose(txt)
    f = norm(frase); fl = loose(frase)
    hit = "EXATA" if f in c else ("LOOSE" if fl in cl else "AUSENTE")
    print(f"[{stem}] {rot}: {hit}")
    if hit == "AUSENTE":
        # mostra contexto da maior ancora
        anchor = loose(frase).split()
        for k in range(len(anchor), 3, -1):
            sub = " ".join(anchor[:k])
            i = cl.find(sub)
            if i >= 0:
                print(f"      ancora({k}w) achada -> ...{cl[i:i+len(sub)+80]}...")
                break
