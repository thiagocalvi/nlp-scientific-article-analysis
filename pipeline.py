"""
pipeline.py
===========
Trabalho 2 de IIA – UEM / Departamento de Informática
Tema: Segurança Cibernética
leitura de PDFs, pré-processamento, análise e extração de informações.
"""

import json
import re
import string
from collections import Counter
from pathlib import Path

import fitz
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize

# ──────────────────────────────────────────────────────────────────────────────
# 0.  CONFIGURAÇÃO
# ──────────────────────────────────────────────────────────────────────────────
PDF_DIR = Path("artigos")
OUT_DIR = Path("saida")
ONTO_DIR = Path("ontologia")
OUT_DIR.mkdir(exist_ok=True)
ONTO_DIR.mkdir(exist_ok=True)

STOP_EN = set(stopwords.words("english"))
# stop-words extras para textos acadêmicos
STOP_EXTRA = {
    "also",
    "use",
    "used",
    "using",
    "uses",
    "one",
    "two",
    "three",
    "may",
    "however",
    "thus",
    "therefore",
    "moreover",
    "furthermore",
    "et",
    "al",
    "fig",
    "table",
    "paper",
    "article",
    "study",
    "research",
    "propose",
    "proposed",
    "present",
    "presented",
    "show",
    "shown",
    "based",
    "approach",
    "method",
    "work",
    "result",
    "results",
    "data",
    "system",
    "model",
    "new",
    "different",
    "many",
    "can",
    "well",
    "within",
    "without",
    "whether",
    "given",
    "set",
    "per",
    "first",
    "second",
    "third",
    "section",
    "e",
    "g",
    "i",
}
STOP_ALL = STOP_EN | STOP_EXTRA

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()


# ──────────────────────────────────────────────────────────────────────────────
# 1.  LEITURA DE PDFs  –  suporte a layouts de coluna única e dupla coluna
# ──────────────────────────────────────────────────────────────────────────────


def _clean_extracted_text(input_text: str) -> str:
    """
    Pós-processamento aplicado ao texto extraído do PDF:
      • reconecta palavras hifenizadas no final da linha
      • corrige drop caps isolados (ex: "T\\nHIS paper" → "THIS paper")
      • remove cabeçalhos/rodapés (números de página isolados)
      • colapsa espaços múltiplos e linhas em branco excessivas
    """

    # Soft hyphens (U+00AD) usados para quebra de linha em PDFs — remove e une
    input_text = re.sub(r"\xad\s*\n\s*", "", input_text)
    # Hifenização regular no fim da linha
    input_text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", input_text)

    # Drop caps: letra maiúscula sozinha em uma linha seguida de continuação
    # Ex: "T\nHIS paper" → "THIS paper"  (padrão IEEE/ACM com capital decorativo)
    input_text = re.sub(r"(?m)^([A-Z])$\n([A-Za-z])", r"\1\2", input_text)

    # Limpeza de linhas e remoção de números de página isolados
    linhas_limpas = []
    for ln in input_text.splitlines():
        ln_stripped = ln.strip()
        if re.fullmatch(r"\d{1,4}", ln_stripped):
            continue
                            linhas_limpas.append(ln_stripped)

    input_text = "\n".join(linhas_limpas)

    # Colapsa 3+ quebras de linha em apenas 2
    input_text = re.sub(r"\n{3,}", "\n\n", input_text)

    # Remove caracteres de controle exceto newline/tab
    input_text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", input_text)

    return input_text.strip()


def _find_column_gap(blocks: list, page_width: float) -> float | None:
    """
    Detecta a coordenada x do gap entre duas colunas de texto.
    Retorna o ponto médio do gap, ou None se a página for coluna única.

    Estratégia: busca na faixa central [30%, 70%] da largura da página por uma
    região sem cobertura de blocos. Ignora blocos muito estreitos (cabeçalhos de
    seção, notas de margem) que não constituem uma coluna real.
    """
    min_col_width = page_width * 0.15  # coluna real tem ao menos 15% da largura
    search_start = page_width * 0.30
    search_end = page_width * 0.70

    # Intervalos x dos blocos significativos que sobrepõem a zona de busca.
    # Blocos full-width (cabeçalhos/rodapés) são excluídos: se incluídos cobrem
    # toda a faixa [30%-70%] e impedem a detecção do gap de coluna.
    max_col_width = page_width * 0.75
    intervals: list[tuple[float, float]] = []
    for b in blocks:
        bw = b[2] - b[0]
        if bw < min_col_width or bw > max_col_width:
            continue
        x0 = max(b[0], search_start)
        x1 = min(b[2], search_end)
        if x1 > x0:
            intervals.append((x0, x1))

    if not intervals:
        return None

    intervals.sort()
    covered = search_start
    best_gap: tuple[float, float] | None = None
    best_width = 0.0

    for x0, x1 in intervals:
        if x0 > covered + 5:  # gap de ao menos 5 pt
            gap_w = x0 - covered
            if gap_w > best_width:
                best_width = gap_w
                best_gap = (covered, x0)
        covered = max(covered, x1)

    if best_gap and best_width > 8:
        return (best_gap[0] + best_gap[1]) / 2

    return None


def _order_page_blocks(blocks: list, page_width: float) -> list:
    """
    Retorna os blocos da página na ordem de leitura correta.
    - Coluna única: ordena por y0.
    - Dupla coluna: blocos da coluna esquerda primeiro (por y0), depois da direita,
      com blocos que atravessam o gap intercalados na posição correta.
    """
    if not blocks:
        return []

    gap_x = _find_column_gap(blocks, page_width)

    if gap_x is None:
        # Coluna única: ordena por y0 e, em caso de empate, por x0 (esquerda primeiro)
        return sorted(blocks, key=lambda b: (b[1], b[0]))

    # Classifica os blocos
    left_col = sorted([b for b in blocks if b[2] <= gap_x], key=lambda b: b[1])
    right_col = sorted([b for b in blocks if b[0] >= gap_x], key=lambda b: b[1])
    # Blocos que atravessam o gap (largura total, títulos de seção, figuras)
    full_w = sorted(
        [b for b in blocks if b[0] < gap_x and b[2] > gap_x],
        key=lambda b: b[1],
    )

    if not left_col or not right_col:
        # Fallback: sem colunas distintas, ordena por posição
        return sorted(blocks, key=lambda b: (b[1], b[0]))

    # Intercala blocos full-width como barreiras entre segmentos de coluna
    result: list = []
    barriers = [(b[1], b) for b in full_w] + [(float("inf"), None)]
    prev_y = -float("inf")

    for barrier_y, barrier_block in barriers:
        l_seg = [b for b in left_col if prev_y <= b[1] < barrier_y]
        r_seg = [b for b in right_col if prev_y <= b[1] < barrier_y]
        result.extend(sorted(l_seg, key=lambda b: b[1]))
        result.extend(sorted(r_seg, key=lambda b: b[1]))
        if barrier_block is not None:
            result.append(barrier_block)
            prev_y = barrier_y

    return result


def _read_with_fitz(path: Path) -> str:
    doc = fitz.open(path)
    textos_ordenados = []

    for page in doc:
        pw = page.rect.width
        blocos = page.get_text("blocks")
        blocos_texto = [b for b in blocos if b[6] == 0 and b[4].strip()]

        for b in _order_page_blocks(blocos_texto, pw):
            texto_do_bloco = b[4].strip()
            if texto_do_bloco:
                textos_ordenados.append(texto_do_bloco)

    doc.close()

    return "\n\n".join(textos_ordenados)


def read_pdf(path: Path) -> str:
    """
    1. Extrai texto de um PDF usando fitz.
    2. Aplica pós-processamento (hifens, páginas, espaços).
    3. Retorna o texto.
    """

    extracted_text = ""

    try:
        extracted_text = _read_with_fitz(path)
    except Exception as e:
        print(f"    [fitz] erro: {e}")

    result = _clean_extracted_text(extracted_text)
    if not result:
        print(f"    [AVISO] Nenhum texto extraído de {path.name}")
    else:
        print(f"    extrator: fitz  |  chars: {len(result)}")

    return result


def load_all_pdfs(directory: Path) -> dict[str, str]:
    """Lê todos os PDFs de um diretório. Retorna {nome_arquivo: texto_bruto}."""
    pdfs = {}
    files = sorted(directory.glob("*.pdf"))
    if not files:
        print(f"[AVISO] Nenhum PDF encontrado em '{directory}/'.")
        return pdfs
    for f in files:
        print(f"  Lendo: {f.name}")
        pdfs[f.stem] = read_pdf(f)
    print(f"  → {len(pdfs)} artigo(s) carregado(s).\n")
    return pdfs


# ──────────────────────────────────────────────────────────────────────────────
# 2.  SEPARAÇÃO CORPO vs. REFERÊNCIAS
# ──────────────────────────────────────────────────────────────────────────────
_REF_PATTERNS = [
    r"\breferences\b",
    r"\bbibliography\b",
    r"\bbibliographies\b",
    r"\bworks cited\b",
]


def split_body_references(text: str) -> tuple[str, str]:
    """
    Divide o texto em (corpo, referências).
    Procura a última ocorrência de 'References' (case-insensitive) como cabeçalho.
    """
    best_idx = -1
    for pat in _REF_PATTERNS:
        for m in re.finditer(pat, text, re.IGNORECASE):
            # só conta se estiver no início de uma linha (cabeçalho de seção)
            start = m.start()
            line_start = text.rfind("\n", 0, start)
            prefix = text[line_start + 1 : start].strip()
            if len(prefix) < 5:  # quase início de linha
                best_idx = max(best_idx, start)
    if best_idx == -1:
        return text, ""
    return text[:best_idx], text[best_idx:]


# ──────────────────────────────────────────────────────────────────────────────
# 3.  PRÉ-PROCESSAMENTO
# ──────────────────────────────────────────────────────────────────────────────
def preprocess(text: str, do_stem: bool = False, do_lemma: bool = True) -> list[str]:
    """
    Tokeniza, remove pontuação/números/stopwords, aplica lematização/stemming.
    Retorna lista de tokens limpos.
    """
    text = text.lower()
    text = re.sub(r"\d+", " ", text)
    text = text.translate(
        str.maketrans(string.punctuation, " " * len(string.punctuation))
    )
    tokens = word_tokenize(text)
    clean = []
    for tok in tokens:
        if len(tok) < 3:
            continue
        if tok in STOP_ALL:
            continue
        if do_lemma:
            tok = lemmatizer.lemmatize(tok)
        if do_stem:
            tok = stemmer.stem(tok)
        if tok not in STOP_ALL and len(tok) >= 3:
            clean.append(tok)
    return clean


# ──────────────────────────────────────────────────────────────────────────────
# 4.  BAG-OF-WORDS  e  N-GRAMAS
# ──────────────────────────────────────────────────────────────────────────────
def bag_of_words(tokens: list[str]) -> Counter:
    return Counter(tokens)


def ngrams(tokens: list[str], n: int) -> Counter:
    grams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    return Counter(grams)


def top_terms(tokens: list[str], n: int = 10) -> list[tuple[str, int]]:
    return bag_of_words(tokens).most_common(n)


# ──────────────────────────────────────────────────────────────────────────────
# 5.  EXTRAÇÃO DE REFERÊNCIAS BIBLIOGRÁFICAS
# ──────────────────────────────────────────────────────────────────────────────

# Linhas que indicam fim da seção de refs (tudo a partir daqui é descartado).
# \W? cobre tanto a aspa reta (U+0027) quanto a aspa tipográfica (U+2019).
_REF_STOP_RE = re.compile(
    r"(?i)^(?:publisher\W?s?\s+note|authors?\s+and\s+affiliations?)\b"
    r"|^\s*©",  # bloco de copyright/licença começa aqui
)

# Linhas de ruído dentro da seção de refs (ignoradas individualmente)
_REF_NOISE_LINE_RE = re.compile(
    r"\(\d{4}\)\s+\d+:\d+\s+[Pp]age\s+\d+\s+of\s+\d+"  # "J. Title (2026) 15:48 Page 3 of 22"
    r"|\b[Pp]age\s+\d+\s+of\s+\d+\b"  # "Page 3 of 22" standalone
    # Cabeçalhos de página Springer: "Autor and Autor Journal (Year) Vol:Art"
    # ou "Autor et al. Journal (Year) Vol:Art" — sem número de ref no início
    r"|^[A-Z][a-z]+\s+(?:and\s+[A-Z][a-zA-Z]*\s+|et\s+al\.\s+)\w+.*\(\d{4}\)\s+\d+:\d+",
    re.IGNORECASE,
)

# Cabeçalhos/rodapés de página que NÃO casam o padrão acima (não começam por autor),
# verificados com re.match (case-sensitive). São linhas isoladas, intercaladas entre
# referências por quebra de página, que de outro modo seriam grudadas na ref anterior.
_REF_RUNHEADER_RES = (
    # Periódico Springer terminando em "(Ano) vol:art" — ex.:
    # "International Journal of Data Science and Analytics  (2026) 22:90".
    # Não inicia com dígito/colchete (não é referência numerada) e o ano-paren+vol:art
    # fica no FIM da linha (em referências reais o ano vem logo após os autores).
    re.compile(r"^(?![\[\d]).*\([12]\d{3}\)\s*\d+\s*:\s*\d+\s*$"),
    # Cabeçalho IEEE em caixa-alta — ex.:
    # "IEEE COMMUNICATIONS SURVEYS & TUTORIALS, VOL. 18, NO. 2, SECOND QUARTER 2016".
    re.compile(r"^[A-Z][A-Z &/]{4,}.*\bVOL\."),
    # Rodapé de página com a autoria corrente — ex.: "M. Alawida et al.",
    # "M. E. Oka et al." (com número de página opcional nas pontas).
    re.compile(
        r"^\d{0,3}\s*(?:[A-Z]\.\s*){1,3}[A-Z][A-Za-zÀ-ÿ'\-]+\s+et\s+al\.?\s*\d{0,3}$"
    ),
)


def _is_allcaps_header(line: str) -> bool:
    """Cabeçalho/título corrente em CAIXA-ALTA — ex.: o próprio título do artigo
    repetido no topo da página ('A SURVEY OF DATA MINING AND MACHINE LEARNING METHODS')."""
    letters = [c for c in line if c.isalpha()]
    if len(letters) < 15 or len(line.split()) < 4:
        return False
    return sum(c.isupper() for c in letters) / len(letters) >= 0.9


def _preclean_ref_text(text: str) -> str:
    """Remove page headers, publisher notes, bios and copyright from ref section text."""
    lines = text.split("\n")
    cleaned: list[str] = []
    for line in lines:
        stripped = line.strip()
        if _REF_STOP_RE.match(stripped):
            break  # trunca tudo a partir daqui
        if re.search(r"photograph\s+and\s+biograph", stripped, re.IGNORECASE):
            break  # bios de autores (IEEE) vêm após todas as referências
        if _REF_NOISE_LINE_RE.search(line):
            continue  # pula esta linha de ruído
        if any(rx.match(stripped) for rx in _REF_RUNHEADER_RES):
            continue  # cabeçalho/rodapé de periódico intercalado
        if _is_allcaps_header(stripped):
            continue  # título corrente em caixa-alta
        cleaned.append(line)
    return "\n".join(cleaned)


# --- PADRÕES DE INÍCIO DE REFERÊNCIA ---
# Numérico (IEEE/Vancouver): [1], [ 1 ], (1), 1., 12.
#  • "(\d{1,3})" limita o marcador entre parênteses a 1–3 dígitos: evita que
#    anos no início de continuações ("(2008)", "(2023)") sejam tomados como
#    início de referência e partam a entrada ao meio.
#  • o número seguido de ponto exige espaço/fim depois (descarta decimais
#    "3.5 GHz" e anos "2024.") e NÃO pode preceder uma URL — fragmentos de DOI
#    quebrados ("94. https://…", "9. https://…") não são novas referências.
_RE_REF_NUM = re.compile(
    r"^(?:\[\s*\d+\s*\]|\(\d{1,3}\)|\d{1,3}\.(?=\s|$)(?!\s*https?://))"
)

# Autor-Data (APA/Harvard/Springer): nome capitalizado seguido de ano entre
# parênteses. O grupo opcional (?:...) admite UM parêntese não-ano antes do ano
# (autores organizacionais: "...Agency (ENISA) (2021)", "Gavriluță (Vatamanu) AF ... (2018)").
# Ancorar no ano-entre-parênteses é robusto: continuações usam (2)/(3) de vol/página,
# quase nunca (2024).
_RE_REF_AUTHOR_YEAR = re.compile(
    r"^[A-Z][A-Za-zÀ-ÿ'’\-]+[,.\s](?:[^()]*\([^)]*\))?[^()]*?\(\d{4}[a-z]?\)"
)

# IEEE com inicial primeiro: "A. Abbasi, J. Wetzels, ..." (inicial, sobrenome, vírgula).
# Exige a vírgula da lista de autores para não casar início de frase ("A. Test was...").
_RE_REF_IEEE = re.compile(r"^[A-Z]\.\s*[A-Z][A-Za-zÀ-ÿ'’\-]+,")

# Caracteres de largura zero que o PyMuPDF intercala em URLs ("h​t​t​p​s") e afins.
_ZERO_WIDTH_RE = re.compile(r"[​‌‍﻿]")


def _normalize_ref(text: str) -> str:
    """Remove caracteres de largura zero (reconstrói URLs) e colapsa espaços/tabs."""
    text = _ZERO_WIDTH_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def _detect_ref_style(lines: list[str]) -> str:
    """
    Decide o estilo dominante da seção de referências.

    'numbered'    → cada entrada começa com um número ([1], 1.).  É o sinal mais
                    confiável; nestes artigos NÃO se detecta autor (continuações
                    costumam trazer "Journal. (2024)" no fim e gerariam falsos inícios).
    'author_year' → entradas iniciam por autor seguido de ano (APA/Springer/IEEE).
    """
    num = sum(1 for l in lines if _RE_REF_NUM.match(l))
    ay = sum(
        1 for l in lines if _RE_REF_AUTHOR_YEAR.match(l) and not _RE_REF_NUM.match(l)
    )
    if num >= 5 and num >= ay:
        return "numbered"
    return "author_year"


def extract_references(ref_text: str) -> list[str]:
    print(ref_text)
    """
    Extrai referências respeitando o estilo dominante do artigo.

    Uma nova referência só é iniciada quando a linha casa o detector do estilo;
    linhas em branco NÃO encerram uma referência (quebras de página/coluna inserem
    linhas vazias no meio de uma mesma referência), e continuações são acumuladas.
    """
    # Remove cabeçalho de seção que pode estar colado ao texto
    # (ex: "References A. Abbasi..." ou "References\nA. Abbasi...")
    ref_text = re.sub(
        r"(?i)^\s*(?:references|bibliography|bibliographies|works\s+cited)\s*",
        "",
        ref_text.lstrip(),
        count=1,
    )
    print(ref_text)

    # Remove ruído (headers de página, publisher notes, bios, copyright)
    ref_text = _preclean_ref_text(ref_text)

    # Normaliza largura-zero por linha e descarta linhas vazias
    lines = [_ZERO_WIDTH_RE.sub("", ln).strip() for ln in ref_text.split("\n")]
    lines = [ln for ln in lines if ln]

    style = _detect_ref_style(lines)
    if style == "numbered":

        def is_start(ln: str) -> bool:
            return bool(_RE_REF_NUM.match(ln))
    else:

        def is_start(ln: str) -> bool:
            return bool(_RE_REF_AUTHOR_YEAR.match(ln) or _RE_REF_IEEE.match(ln))

    refs: list[str] = []
    current = ""
    for line in lines:
        if is_start(line):
            if current:
                refs.append(current)
            current = line
        elif current:
            current += " " + line  # continuação da referência atual
        else:
            current = line  # fallback: 1ª linha não casou o detector

    if current:
        refs.append(current)

    # Normaliza cada referência e descarta fragmentos curtos (lixo de conversão)
    refs = [_normalize_ref(r) for r in refs]
    print("-------------------------------------------------------")
    print(refs)
    print(len(refs))
    return [r for r in refs if len(r) > 20]


# ──────────────────────────────────────────────────────────────────────────────
# 5b. EXTRAÇÃO DE ANO DO ARTIGO
# ──────────────────────────────────────────────────────────────────────────────
def extract_year(text: str, filename: str = "") -> int | None:
    """
    Tenta extrair o ano de publicação do artigo.
    Estratégia: procura padrões como © 2021, Published 2020, (2019), etc.
    nas primeiras linhas do texto. Cai no nome do arquivo como último recurso.
    """
    # nas primeiras 1500 chars (cabeçalho / abstract)
    header = text[:1500]

    patterns = [
        r"©\s*(20\d{2})",  # © 2021
        r"[Cc]opyright\s+(20\d{2})",  # Copyright 2021
        r"[Pp]ublished\s+(?:in\s+)?(\d{4})",  # Published 2021
        r"[Rr]eceived\s+\w+\s+\d+,?\s+(20\d{2})",  # Received Jan 2021
        r"[Aa]ccepted\s+\w+\s+\d+,?\s+(20\d{2})",
        r"\b(20[12]\d)\b",  # qualquer 2010-2029
    ]
    for pat in patterns:
        m = re.search(pat, header)
        if m:
            return int(m.group(1))

    # tenta no nome do arquivo: ex. article01_2021_ransomware
    m = re.search(r"(20\d{2})", filename)
    if m:
        return int(m.group(1))

    return None


# ──────────────────────────────────────────────────────────────────────────────
# 5c. DETECÇÃO DE TÉCNICAS/METODOLOGIAS MENCIONADAS
# ──────────────────────────────────────────────────────────────────────────────

# Dicionário de técnicas de Segurança Cibernética com variações
TECHNIQUES: dict[str, list[str]] = {
    "Intrusion Detection": [r"\bintrusion detection\b", r"\bids\b", r"\bips\b"],
    "Cryptography": [
        r"\bcryptograph\w*\b",
        r"\bencryption\b",
        r"\bdecryption\b",
        r"\bcipher\b",
    ],
    "Firewall": [r"\bfirewall\b"],
    "Machine Learning": [r"\bmachine learning\b", r"\bclassif\w+\b"],
    "Deep Learning": [
        r"\bdeep learning\b",
        r"\bneural network\b",
        r"\bcnn\b",
        r"\brnn\b",
    ],
    "Static Analysis": [r"\bstatic analysis\b", r"\bstatic analys\w+\b"],
    "Dynamic Analysis": [r"\bdynamic analysis\b", r"\bsandbox\b"],
    "Penetration Testing": [r"\bpenetration test\w*\b", r"\bpentest\w*\b"],
    "Vulnerability Scanning": [r"\bvulnerability scan\w*\b", r"\bcve\b"],
    "Digital Forensics": [r"\bdigital forensic\w*\b", r"\bforensic\w*\b"],
    "Zero Trust": [r"\bzero trust\b", r"\bzta\b"],
    "Threat Intelligence": [
        r"\bthreat intelligence\b",
        r"\bcti\b",
        r"\bstix\b",
        r"\btaxii\b",
    ],
    "Network Monitoring": [
        r"\bnetwork monitor\w*\b",
        r"\btraffic analys\w+\b",
        r"\bpacket\b",
    ],
    "Malware Analysis": [
        r"\bmalware\b",
        r"\bransomware\b",
        r"\bvirus\b",
        r"\btrojan\b",
    ],
    "Phishing Detection": [r"\bphishing\b", r"\bspear.?phishing\b"],
    "Access Control": [
        r"\baccess control\b",
        r"\brbac\b",
        r"\babac\b",
        r"\bauthoriz\w+\b",
    ],
    "Social Engineering": [r"\bsocial engineering\b", r"\bpretexting\b"],
    "DDoS Mitigation": [r"\bddos\b", r"\bdenial.of.service\b"],
    "Post-Quantum Crypto": [
        r"\bpost.quantum\b",
        r"\bquantum.resistant\b",
        r"\blattice.based\b",
    ],
    "Anomaly Detection": [r"\banomaly detection\b", r"\boutlier\b"],
}


def detect_techniques(body_text: str) -> list[str]:
    """
    Retorna lista de técnicas mencionadas no corpo do artigo.
    """
    text_lower = body_text.lower()
    found = []
    for technique, pats in TECHNIQUES.items():
        for pat in pats:
            if re.search(pat, text_lower):
                found.append(technique)
                break
    return found


# ──────────────────────────────────────────────────────────────────────────────
# 6.  ETAPA 2 – EXTRAÇÃO DE INFORMAÇÕES ESTRUTURADAS
# ──────────────────────────────────────────────────────────────────────────────
# Prefixos de seção que o PDF cola ao início da primeira frase do parágrafo
_SECTION_HEADER_RE = re.compile(
    r"(?i)^(?:\d+[\s.]+)?(?:conclusions?(?:\s+and\s+(?:future\s+)?work)?|"
    r"author\s+contributions?|contributions?|introduction|abstract|summary|"
    r"related\s+work|methodology|results?|discussion|problem\s+formulation)\s+",
)


def _clean_candidate(sent: str) -> str | None:
    """Normaliza espaços e descarta sentenças que são ruído de extração de PDF.

    Retorna a sentença limpa (com prefixo de título de seção removido) ou None
    se a sentença deve ser descartada (cabeçalho, afiliação, keywords, etc.).
    """
    s = re.sub(r"\s+", " ", sent).strip()
    # (a) URL / DOI  (b) início minúsculo (afiliação)  (c) cabeçalho de periódico
    if re.search(r"https?://|doi\.org", s, re.IGNORECASE):
        return None
    if re.match(r"^[a-z]", s):
        return None
    if re.match(r"^(?:[A-Z]\w*\s+){1,6}\(20\d{2}\)", s):
        return None
    if re.match(r"^\[\d+\]", s):  # (d) referência "[33] ..."
        return None
    if re.match(r"^keywords?\b", s, re.IGNORECASE):  # (e) linha de keywords
        return None
    if s.count("·") >= 2:  # (f) lista de keywords Springer
        return None
    if re.match(r"^\d+,\s+(?:FIRST|SECOND|THIRD|FOURTH)\s+QUARTER", s):  # (g)
        return None
    if re.match(r"^\(\d{4}\)", s):  # (h) "(2024) Our Study..."
        return None
    if re.search(r"\(20\d{2}\)\s+\d+:\d+", s):  # (i) cabeçalho vol:página
        return None
    s = _SECTION_HEADER_RE.sub("", s).strip()  # remove título de seção grudado
    if not s or re.match(r"^[a-z]", s):
        return None
    return s


# Aberturas de lista cujo conteúdo real vem em linhas seguintes (não capturadas):
# penalizadas para não serem escolhidas no lugar de uma frase completa.
_LIST_OPENER_RE = re.compile(
    r"(?:as\s+follows|in\s+the\s+following|following\s+points?|below)\s*[:.]?\s*$"
    r"|:\s*\d{0,2}\W*$",
    re.IGNORECASE,
)


# TODO: REVISAR PARA MELHORAR A EXTRAÇÃO DOS OBJETIVOS, CONSIDERAR MOFICAR A FORMA COMO O CALCULO É FEITO
def _score_candidates(
    text: str,
    weighted_patterns: list[tuple[str, float]],
    region_bonus: float = 0.0,
    merge_lists: bool = False,
) -> list[tuple[str, float]]:
    """Pontua cada sentença que casa algum padrão.

    score = (maior peso de padrão casado) + region_bonus - penalidades.
    Penaliza fragmentos curtos e aberturas de lista sem conteúdo, para que a
    seleção priorize frases completas e específicas (e não a 1ª que casar).

    merge_lists=True: quando a frase é uma abertura de lista (ex.: "Our
    contributions are: (1)..."), anexa as sentenças seguintes (os itens), pois
    o conteúdo da contribuição costuma ficar nelas. Usado para contribuições.
    """
    cleaned = [_clean_candidate(s) for s in sent_tokenize(text)]
    out: list[tuple[str, float]] = []
    for i, s in enumerate(cleaned):
        if not s:
            continue
        best = 0.0
        for pat, w in weighted_patterns:
            if w > best and re.search(pat, s, re.IGNORECASE):
                best = w
        if best <= 0:
            continue
        pen = 0.0
        text_s = s
        if _LIST_OPENER_RE.search(s):
            anexados = ""
            if merge_lists:
                for nxt in cleaned[i + 1 : i + 4]:
                    if not nxt:
                        continue
                    anexados += " " + nxt
                    if len(anexados) > 220:
                        break
            if anexados:
                text_s = (s + anexados).strip()
            else:
                pen += 1.5
        elif len(s) < 60:
            pen += 1.0
        out.append((text_s, best + region_bonus - pen))
    return out


# Score mínimo para aceitar qualquer candidato. Filtra frases que casam apenas
# padrões genéricos em regiões de baixa prioridade (ex.: "challenge" no intro
# com bonus=0.5 → score 1.0). Valor de referência:
#   abstract bonus=2.0 + peso 0.5 (genérico) = 2.5  → aceito
#   intro    bonus=0.5 + peso 1.0              = 1.5  → rejeitado
#   methods_section bonus=1.0 + peso 1.0       = 2.0  → aceito
_MIN_CANDIDATE_SCORE = 2.0


def _select(*candidate_lists: list[tuple[str, float]], n: int) -> list[str]:
    """Une candidatos de várias regiões, deduplica (mantendo maior score) e
    devolve as `n` melhores sentenças que atingem _MIN_CANDIDATE_SCORE."""
    merged: dict[str, tuple[str, float]] = {}
    for lst in candidate_lists:
        for s, score in lst:
            key = re.sub(r"\s+", " ", s).strip().lower()
            if key not in merged or score > merged[key][1]:
                merged[key] = (s, score)
    ranked = sorted(merged.values(), key=lambda x: -x[1])
    return [s for s, score in ranked if score >= _MIN_CANDIDATE_SCORE][:n]


# ── Nouns e verbos reutilizados nos padrões ────────────────────────────────
_PAPER_NOUNS = (
    r"paper|study|article|work|analysis|review|survey|"
    r"investigation|assessment|evaluation|research"
)
_OBJ_VERBS = (
    r"aims?|seeks?|intends?|proposes?|presents?|investigates?|examines?|"
    r"focuses?|addresses?|describes?|analyzes?|analyses?|reviews?|assesses?|"
    r"evaluates?|surveys?|explores?|identifies?|provides?|reports?|studies|"
    r"conducts?"
)

# Padrões PONDERADOS (regex, peso). Peso maior = marcador mais específico do
# campo; padrões genéricos recebem peso baixo e só vencem na ausência de algo melhor.
_OBJ_PATS = [
    (
        rf"\bthe\s+(aim|goal|objective|purpose|intent)\s+of\s+(this|the|our)\s+(?:\w+\s+)?({_PAPER_NOUNS})\b",
        3.0,
    ),
    (rf"\bthe\s+({_PAPER_NOUNS})\s+(aims?|seeks?|intends?)\s+to\b", 3.0),
    (rf"\bthis\s+(?:\w+\s+)?({_PAPER_NOUNS})\s+(?:\w+\s+)?({_OBJ_VERBS})\b", 2.5),
    (rf"\bin\s+this\s+({_PAPER_NOUNS}),?\s+we\b", 2.0),
    (
        r"\bwe\s+(aim|seek|propose|present|investigate|examine|address|focus|develop|design|introduce|build)\b",
        2.0,
    ),
    (r"\bthe\s+(main\s+|primary\s+|overall\s+)?(aim|goal|objective|purpose)\b", 2.0),
    (r"\bour\s+(aim|goal|objective|purpose)\b", 1.5),
]

_PROB_PATS = [
    # gap/limitação explicitamente ligados à pesquisa (alta especificidade)
    (
        r"\b(research\s+gap|gap\s+in\s+the\s+literature|significant\s+gap|critical\s+gap)\b",
        3.0,
    ),
    (
        r"\bhowever\b.{0,90}\b(lack|fail|limit|problem|issue|challenge|gap|absence|scarce|few|no)\b",
        3.0,
    ),
    (
        r"\bexisting\s+(approaches?|methods?|solutions?|systems?|frameworks?|models?)\s+(fail|lack|do\s+not|are\s+not|cannot|struggle)\b",
        3.0,
    ),
    # limitação de técnicas: "misuse-based techniques cannot detect ...", "models fail to ..."
    (
        r"\b(techniques?|methods?|approaches?|solutions?|systems?|models?|algorithms?|tools?|detectors?)\s+"
        r"(cannot|can\s?not|are\s+unable|fail\s+to|do\s+not|are\s+not\s+able|struggle)\b",
        2.5,
    ),
    (
        r"\b(remains?|stays?|is|are)\s+(challenging|difficult|limited|underexplored|under-explored|open|unclear|unsolved|insufficient)\b",
        2.5,
    ),
    (
        r"\b(few|no|little|limited)\s+(studies|works?|research|attention|comprehensive)\b",
        2.5,
    ),
    (
        r"\bnot\s+(yet|fully|adequately|well)\s+(addressed|solved|studied|explored|understood)\b",
        2.5,
    ),
    (r"\bstill\s+(remains?|lacks?|suffers?|faces?)\b", 2.5),
    (r"\b(lack|absence)\s+of\b", 2.0),
    (
        r"\bto\s+(address|tackle|solve|overcome|mitigate|bridge)\b.{0,40}\b(gap|challenge|problem|limitation|issue)\b",
        2.0,
    ),
    # "present/pose significant challenges/risks" — aljaradat e similares
    (
        r"\b(present|pose|represent|create)\s+(significant|major|serious|growing|critical)\s+"
        r"(challenge|risk|threat|issue|concern)\w*\b",
        2.0,
    ),
    (r"\bunfortunately\b", 1.5),
    # menção genérica de problema — peso reduzido para não vencer padrões
    # específicos no intro; ainda serve de fallback no abstract (bonus=2.0)
    (
        r"\b(challenge|problem|issue|limitation|gap|drawback|shortcoming|barrier|"
        r"difficulty|obstacle|disadvantage|weakness|vulnerabilit)\w*\b",
        0.5,
    ),
]

_METH_PATS = [
    # desenho metodológico explícito
    (
        r"\b(quantitative|qualitative|mixed.?method)\s+(research|approach|design|analysis|study)\b",
        3.0,
    ),
    (r"\bdata\s+(were|was|are|is)\s+(collected|gathered|obtained|retrieved)\b", 3.0),
    # métodos de pesquisa nomeados (revisão/bibliometria/estatística/PLN)
    (
        r"\b(scientometric|bibliometric|systematic\s+(literature\s+)?review|literature\s+review|"
        r"content\s+analysis|document\s+analysis|thematic\s+analysis|co-?occurrence|"
        r"bibliographic\s+coupling|meta-?analysis|case\s+study|grounded\s+theory|"
        r"structural\s+equation\s+model|regression|partial\s+least\s+squares|"
        r"natural\s+language\s+processing|tf-?idf|topic\s+model|coding)\b",
        2.5,
    ),
    (
        r"\bwe\s+(conduct|perform|carry\s+out|implement|develop|design|evaluate|train|test|collect|analyz|appl|employ|gather|compile|select)\w*\b",
        2.0,
    ),
    (r"\bthe\s+proposed\s+(method|approach|system|framework|algorithm|model)\b", 2.0),
    # voz passiva metodológica — molina2017 e papers de comparação de ferramentas
    (
        r"\bthe\s+(study|paper|work|analysis)\s+(is\s+focused\s+on|focuses\s+on|compares?|analyses?|analyzes?|evaluates?)\b",
        2.0,
    ),
    (
        r"\b(properties|metrics|strategies|characteristics|features)\s+(are|were)\s+(analysed?|analyzed?|evaluated?|compared?|assessed?)\b",
        2.0,
    ),
    (
        r"\b(using|based\s+on|by\s+means\s+of)\b.{0,60}(method|technique|approach|algorithm|model|framework|data|survey|interview)\b",
        1.5,
    ),
    # menção genérica de método (peso baixo)
    (
        r"\b(methodology|approach|technique|algorithm|framework|protocol|experiment|"
        r"survey|interview|dataset|benchmark|simulation|testbed)\b",
        1.0,
    ),
]

_CONTRIB_PATS = [
    (
        r"\b(our|the)\s+(main|key|major|primary|principal)\s+(contribution|contributions)\b",
        3.0,
    ),
    (
        r"\b(main|key|major|primary)\s+(contribution|contributions)\s+(of|are|include|can)\b",
        3.0,
    ),
    (
        rf"\bthis\s+(?:\w+\s+)?({_PAPER_NOUNS})\s+(?:\w+\s+)?(presents?|provides?|introduces?|proposes?|offers?|makes?)\s+(?:the\s+|a\s+|an\s+|several\s+|key\s+|its\s+)*(first|novel|main|key|comprehensive|contribution)",
        2.5,
    ),
    (
        rf"\b(?:(?:this|the|our)\s+(?:\w+\s+)?(?:{_PAPER_NOUNS})|we|our\s+(?:work|research))\s+(?:\w+\s+)*contribut",
        2.0,
    ),
    (
        r"\bwe\s+(contribute|propose|introduce|present|provide|develop|design|identify|reveal)\b",
        1.5,
    ),
    (r"\bnovel\s+(contribution|approach|framework|method|technique|algorithm)\b", 1.5),
    (r"\bfirst\s+(paper|study|work|comprehensive)\b", 1.5),
    (r"\bto\s+the\s+best\s+of\s+(our|the\s+authors'?)\s+knowledge\b", 1.5),
    # contribuição típica de survey/review: "the paper provides a taxonomy/
    # comparison criteria/recommendations/overview..."
    (
        r"\b(paper|survey|study|review|article|work)\s+(provides?|presents?|offers?|gives?)\s+"
        r"(?:a\s+|an\s+|the\s+)?(set\s+of|taxonomy|comprehensive|comparison|overview|"
        r"recommendation|criteria|guideline|tutorial)",
        1.5,
    ),
    # contribui genericamente "para o campo/literatura" (peso baixo)
    (
        r"\bcontribut(?:e|es|ing|ion|ions)\s+to\s+(?:the\s+)?(?:field|literature|knowledge|understanding|body|discourse|theory|practice|research|community|discussion|debate)\b",
        1.0,
    ),
]

# Sinais de que a frase é uma limitação, não uma contribuição
_LIMITATION_RE = re.compile(
    r"\b(?:limitations?|caveats?|trade-?offs?\s+merit|should\s+be\s+interpreted\b|"
    r"future\s+(?:work|research|stud)|"
    r"did\s+not\s+(?:use|include|consider|test|cover)|"
    r"relies?\s+on\s+cross.sectional|"
    r"cannot\s+(?:be\s+)?(?:generalized|extended|applied)|"
    r"potential\s+bias|not\s+represent(?:ative)?)\b",
    re.IGNORECASE,
)
# Exceção: superar/endereçar limitações de OUTROS é contribuição, não limitação
# própria — ex.: "to overcome limitations of traditional models".
_OVERCOME_RE = re.compile(
    r"\b(overcome|overcomes|overcoming|address(?:es|ing)?|tackl\w+|mitigat\w+|"
    r"solv\w+|surpass\w*|bridg\w+)\s+(?:the\s+|these\s+|key\s+|several\s+|major\s+|"
    r"existing\s+|current\s+)*(?:limitations?|challenges?|gaps?|drawbacks?|shortcomings?)",
    re.IGNORECASE,
)


def _is_limitation(s: str) -> bool:
    """True se a frase descreve uma limitação do PRÓPRIO estudo (e não a
    superação de limitações alheias, que é contribuição)."""
    return bool(_LIMITATION_RE.search(s)) and not _OVERCOME_RE.search(s)


# Padrões de contribuição complementares — "this paper extends/offers..." são
# afirmações de contribuição válidas. Padrões de resultado ("our findings show",
# "we found") foram removidos daqui: eles capturavam achados como contribuições.
_CONCL_PATS = [
    (
        rf"\bthis\s+(?:\w+\s+)?({_PAPER_NOUNS})\s+"
        r"(complements?|extends?|builds?\s+(?:on|upon)|advances?|improves?\s+upon|enriches?)\b",
        1.5,
    ),
    (
        rf"\bthis\s+(?:\w+\s+)?({_PAPER_NOUNS})\s+(offers?|provides?|fills?|addresses?)\b",
        1.5,
    ),
]


def _extract_abstract(body_text: str) -> str:
    """Retorna o texto do resumo (entre 'Abstract' e 'Keywords'/'Introduction').

    O resumo é a fonte mais densa e limpa de objetivo/problema/metodologia/
    contribuição. Tolera cabeçalhos espaçados ('A B S T R A C T'). Em caso de
    não localizar fronteiras, usa uma janela generosa após o cabeçalho ou o
    topo do artigo como fallback.
    """
    head = body_text[:9000]
    m = re.search(r"(?i)\bA\s*b\s*s\s*t\s*r\s*a\s*c\s*t\b", head)
    if not m:
        return body_text[:1800]  # fallback: topo do artigo
    region = body_text[m.end() : m.end() + 3000]
    fim = re.search(
        r"(?i)\b(K\s*e\s*y\s*w\s*o\s*r\s*d|"
        r"I\s*n\s*t\s*r\s*o\s*d\s*u\s*c\s*t\s*i\s*o\s*n|"
        r"\d\s*\.?\s*Introduction)\b",
        region,
    )
    if fim:
        region = region[: fim.start()]
    return region.strip()


def _extract_conclusion_section(body_text: str) -> str:
    """Retorna o texto da seção de conclusão (ou os últimos 8000 chars).

    Usa a PRIMEIRA ocorrência do cabeçalho de conclusão na segunda metade do
    artigo para não cortar seções internas (ex: subsection 'Implications').
    """
    pattern = re.compile(
        r"\b(conclusions?(\s+and\s+\w+)?|concluding\s+remarks?|"
        r"summary\s+and\s+conclusions?|discussion\s+and\s+conclusions?)\b",
        re.IGNORECASE,
    )
    threshold = len(body_text) * 0.5
    for m in pattern.finditer(body_text):
        if m.start() > threshold:
            return body_text[m.start() :]
    # Fallback: últimos 8000 chars
    return body_text[-8000:]


def _extract_methods_section(body_text: str) -> str:
    """Retorna a seção de metodologia (ou o corpo central como fallback).

    Procura cabeçalhos de metodologia entre 10 % e 70 % do artigo para
    não confundir com a intro ou a conclusão. Pega até 6000 chars a partir
    do cabeçalho encontrado.
    """
    _HDR = re.compile(
        r"\b((?:research\s+)?methodolog(?:y|ies)|materials?\s+and\s+methods?|"
        r"proposed\s+(?:method|approach|framework|system|model)|"
        r"experimental\s+setup|research\s+design|system\s+design)\b",
        re.IGNORECASE,
    )
    n = len(body_text)
    seg_start = int(n * 0.10)
    seg_end = int(n * 0.70)
    for m in _HDR.finditer(body_text, seg_start, seg_end):
        line_start = body_text.rfind("\n", 0, m.start()) + 1
        prefix = body_text[line_start : m.start()].strip()
        if len(prefix) < 10:  # pouco texto antes na linha → cabeçalho
            return body_text[m.start() : m.start() + 10000]
    # Fallback: corpo central (evita intro e conclusão)
    return body_text[seg_start:seg_end]


def extract_structured_info(body_text: str) -> dict:
    """
    Extrai objetivo, problema, metodologia e contribuições do corpo do artigo.

    Estratégia: buscar candidatos em várias regiões (resumo, introdução,
    conclusão, corpo) e RANQUEAR por especificidade do padrão + prioridade da
    região, em vez de pegar a 1ª frase que casa. O resumo recebe bônus alto por
    ser a fonte mais limpa dos quatro campos.
    """
    abstract = _extract_abstract(body_text)
    intro = body_text[:12000]
    full = body_text
    conclusion = _extract_conclusion_section(body_text)
    methods_section = _extract_methods_section(body_text)

    objectives = _select(
        _score_candidates(abstract, _OBJ_PATS, region_bonus=2.0),
        _score_candidates(intro, _OBJ_PATS, region_bonus=0.5),
        n=2,
    )
    problems = _select(
        _score_candidates(abstract, _PROB_PATS, region_bonus=2.0),
        _score_candidates(intro, _PROB_PATS, region_bonus=0.5),
        n=3,
    )
    methods = _select(
        _score_candidates(abstract, _METH_PATS, region_bonus=2.0),
        _score_candidates(intro, _METH_PATS, region_bonus=0.5),
        _score_candidates(methods_section, _METH_PATS, region_bonus=1.0),
        n=4,
    )

    # Contribuições: resumo (contrib + achados) + conclusão + corpo (explícitas).
    contrib_cands = (
        _score_candidates(
            abstract, _CONTRIB_PATS + _CONCL_PATS, region_bonus=1.5, merge_lists=True
        )
        + _score_candidates(
            conclusion, _CONTRIB_PATS + _CONCL_PATS, region_bonus=1.0, merge_lists=True
        )
        + _score_candidates(full, _CONTRIB_PATS, region_bonus=0.0, merge_lists=True)
    )
    # Descarta contribuições que falam de OUTRO trabalho, são limitações ou já
    # foram capturadas como objetivo; mantém a de maior score.
    _ref_other = re.compile(
        r"\b(that|those)\s+(paper|study|article|work)\b"
        r"|\b(their|those\s+authors?)\s+(results?|findings?|contributions?|work|approach)\b",
        re.IGNORECASE,
    )
    obj_keys = {re.sub(r"\s+", " ", s).strip().lower() for s in objectives}
    seen: set[str] = set()
    ranked: list[tuple[str, float]] = []
    for s, score in sorted(contrib_cands, key=lambda x: -x[1]):
        key = re.sub(r"\s+", " ", s).strip().lower()
        if key in seen or key in obj_keys:
            continue
        if _ref_other.search(s) or _is_limitation(s):
            continue
        seen.add(key)
        ranked.append((s, score))
    contributions = [s for s, _ in ranked[:4]]

    return {
        "objectives": objectives,
        "problems": problems,
        "methods": methods,
        "contributions": contributions,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 7.  PIPELINE PRINCIPAL
# ──────────────────────────────────────────────────────────────────────────────
def run_pipeline(pdf_dir: Path = PDF_DIR) -> dict:
    print("=" * 60)
    print("  PIPELINE – Análise de Artigos Científicos")
    print("  Tema: Segurança Cibernética")
    print("=" * 60)

    # --- leitura ---
    print("\n[1] Carregando PDFs...")
    raw_texts = load_all_pdfs(pdf_dir)
    if not raw_texts:
        print("Nenhum artigo encontrado. Coloque os PDFs na pasta 'artigos/'.")
        return {}

    all_results = {}
    global_body_tokens: list[str] = []

    for name, raw in raw_texts.items():
        print(f"\n--- Processando: {name} ---")

        # separa corpo e referências
        body, refs_text = split_body_references(raw)

        # pré-processamento
        tokens = preprocess(body)
        global_body_tokens.extend(tokens)

        # top 10 termos
        top10 = top_terms(tokens, 10)

        # bigramas mais comuns
        bi = ngrams(tokens, 2).most_common(10)

        # referências
        refs = extract_references(refs_text)

        # extração estruturada (Etapa 2)
        info = extract_structured_info(body)

        # ano de publicação
        year = extract_year(raw, filename=name)

        # técnicas mencionadas
        techniques = detect_techniques(body)

        all_results[name] = {
            "top10_terms": top10,
            "top10_bigrams": [(" ".join(g), c) for g, c in bi],
            "references": refs,
            "structured_info": info,
            "year": year,
            "techniques": techniques,
        }

        print(f"  Tokens limpos : {len(tokens)}")
        print(f"  Top-3 termos  : {[t for t, _ in top10[:3]]}")
        print(f"  Referências   : {len(refs)}")
        print(f"  Ano           : {year or 'não encontrado'}")
        print(f"  Técnicas      : {techniques[:4]}")

    # --- termos globais (todos os artigos juntos) ---
    print("\n[2] Top 10 termos globais (sem referências):")
    global_top10 = top_terms(global_body_tokens, 10)
    for term, count in global_top10:
        print(f"  {term:25s} {count}")

    all_results["__global__"] = {
        "top10_terms": global_top10,
        "bigrams_top10": [
            (" ".join(g), c) for g, c in ngrams(global_body_tokens, 2).most_common(10)
        ],
        "trigrams_top10": [
            (" ".join(g), c) for g, c in ngrams(global_body_tokens, 3).most_common(10)
        ],
        "total_tokens": len(global_body_tokens),
        "num_articles": len(raw_texts),
    }

    # salva JSON
    out_path = OUT_DIR / "resultados_pipeline.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n[✓] Resultados salvos em: {out_path}")

    return all_results


if __name__ == "__main__":
    run_pipeline()
