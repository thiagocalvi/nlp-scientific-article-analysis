"""
Verifica se as frases do GABARITO (em avaliacao.py) aparecem VERBATIM no texto
extraido dos PDFs (saida/<stem>.txt).

Normaliza apenas artefatos de extracao de PDF (quebras de linha, hifenizacao de
fim de linha, espacos multiplos, aspas/travessoes unicode) — NAO altera as
palavras. Reporta:
  EXATA   : a frase aparece integralmente apos normalizacao.
  PARCIAL : nao aparece integral; mostra o trecho mais proximo no paper (difflib).
"""
import re
import difflib
from pathlib import Path

from avaliacao import GABARITO, CAMPOS

TXT_DIR = Path("saida")


def normalizar(s: str) -> str:
    # unifica aspas/apostrofos/travessoes unicode para ascii
    s = (s.replace("’", "'").replace("‘", "'")
           .replace("“", '"').replace("”", '"')
           .replace("–", "-").replace("—", "-")
           .replace("­", "")            # soft hyphen
           .replace("ﬁ", "fi").replace("ﬂ", "fl"))
    # junta palavras quebradas por hifen no fim da linha: "privately-\nowned"
    s = re.sub(r"-\s*\n\s*", "", s)
    # colapsa qualquer espaco em branco (inclui \n) em um unico espaco
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def normalizar_loose(s: str) -> str:
    # versao mais agressiva: remove hifens e pontuacao para comparar so palavras
    s = normalizar(s).lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def melhor_trecho(frase_norm: str, corpo_norm: str) -> tuple[float, str]:
    """Janela deslizante por difflib: retorna (ratio, trecho mais parecido)."""
    n = len(frase_norm)
    sm = difflib.SequenceMatcher(None, "", "", autojunk=False)
    sm.set_seq2(frase_norm)
    melhor_r, melhor_t = 0.0, ""
    # busca aproximada: ancorar nas primeiras palavras da frase
    chave = " ".join(frase_norm.split()[:6])
    idxs = [m.start() for m in re.finditer(re.escape(chave[:25]), corpo_norm)] if chave else []
    if not idxs:
        # fallback: varre em passos
        idxs = range(0, max(1, len(corpo_norm) - n), max(1, n // 2))
    for i in idxs:
        trecho = corpo_norm[i:i + n + 40]
        sm.set_seq1(trecho)
        r = sm.ratio()
        if r > melhor_r:
            melhor_r, melhor_t = r, trecho
    return melhor_r, melhor_t


def verificar():
    total = exatas = 0
    problemas = []
    for stem, campos in GABARITO.items():
        txt = TXT_DIR / f"{stem}.txt"
        if not txt.exists():
            print(f"[!] {stem}: texto nao encontrado")
            continue
        corpo = normalizar(txt.read_text(encoding="utf-8", errors="ignore"))
        corpo_l = corpo.lower()
        corpo_loose = normalizar_loose(txt.read_text(encoding="utf-8", errors="ignore"))
        for campo in CAMPOS:
            frases = campos.get(campo, "")
            if isinstance(frases, str):
                frases = [frases] if frases else []
            for i, frase in enumerate(frases):
                total += 1
                fn = normalizar(frase)
                fl = normalizar_loose(frase)
                tag = f"{stem}/{campo}[{i}]"
                if fn.lower() in corpo_l:
                    exatas += 1
                    status = "EXATA"
                elif fl in corpo_loose:
                    exatas += 1
                    status = "EXATA*"   # identica ignorando pontuacao/hifen
                else:
                    r, trecho = melhor_trecho(fn.lower(), corpo_l)
                    status = f"PARCIAL ({r:.0%})"
                    problemas.append((tag, frase, trecho, r))
                print(f"  {status:<14} {tag}")
    print("\n" + "=" * 72)
    print(f"Frases verificadas: {total} | exatas: {exatas} | divergentes: {len(problemas)}")
    if problemas:
        print("\n--- DIVERGENCIAS (frase do gabarito  vs  trecho mais proximo) ---")
        for tag, frase, trecho, r in problemas:
            print(f"\n[{tag}]  similaridade {r:.0%}")
            print(f"  GABARITO : {normalizar(frase)}")
            print(f"  PAPER    : {trecho}")


if __name__ == "__main__":
    verificar()
