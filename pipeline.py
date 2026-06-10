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

    # Intervalos x dos blocos significativos que sobrepõem a zona de busca
    intervals: list[tuple[float, float]] = []
    for b in blocks:
        if (b[2] - b[0]) < min_col_width:
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
def extract_references(ref_text: str) -> list[str]:
    """
    Heurística expandida: detecta entradas numeradas (IEEE/Vancouver)
    e tenta detectar padrões de Autor-Data (APA/Harvard).
    """
    refs = []
    lines = ref_text.split("\n")
    current = ""

    # --- PADRÕES DE INÍCIO DE REFERÊNCIA ---
    # Numérico: [1], [ 1 ], 1., 12., (1)
    pat_num = r"^\[\s*\d+\s*\]|^\(\d+\)|^\d{1,3}\."

    # Autor-Data (APA/Harvard): Começa com Letra Maiúscula, minúsculas, vírgula, iniciais.
    # Ex: "Smith, J.", "O'Connor, A.", "Silva, M. T."
    pat_author = r"^[A-Z][a-zA-ZÀ-ÿ\'-]+,\s*[A-Z]\."

    for line in lines:
        line = line.strip()

        if not line:
            if current:
                refs.append(current.strip())
                current = ""
            continue

        # Testa se a linha atual "tem cara" de ser o começo de uma nova referência
        is_numbered = re.match(pat_num, line)
        is_author = re.match(pat_author, line)

        if is_numbered or is_author:
            # Salva a referência anterior (se houver) e começa uma nova
            if current:
                refs.append(current.strip())
            current = line

        elif current:
            # Se não parece um início, é continuação da referência atual
            current += " " + line

        else:
            # Fallback seguro: se não temos 'current' e não bateu com o regex,
            # forçamos o início de uma. (Evita que a 1ª ref seja ignorada se fugir do padrão)
            current = line

    # Salva a última referência processada
    if current:
        refs.append(current.strip())

    # Limpa referências muito curtas (geralmente lixo de conversão ou números de página isolados)
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
def _find_sentences_by_patterns(
    text: str, patterns: list[str], context_chars: int = 600
) -> list[str]:
    """Retorna trechos do texto que casam com qualquer dos padrões.

    O match é feito sobre a frase com espaços normalizados para evitar falsos
    negativos causados por quebras de linha no interior de sentenças extraídas
    do PDF (ex: "The\\naim of..." não casaria sem normalização).
    Sentenças com URLs (cabeçalhos de página concatenados) são descartadas.
    """
    results = []
    sentences = sent_tokenize(text)
    for sent in sentences:
        sent_norm = re.sub(r"\s+", " ", sent).strip()
        # Descarta blocos de cabeçalho de artigo:
        # (a) contêm URL / DOI
        if re.search(r"https?://|doi\.org", sent_norm, re.IGNORECASE):
            continue
        # (b) começam com letra minúscula — sinal de bloco de afiliação/endereço
        if re.match(r"^[a-z]", sent_norm):
            continue
        # (c) cabeçalho de página tipo "Journal Name (2026) 6:41 ..."
        if re.match(r"^(?:[A-Z]\w*\s+){1,6}\(20\d{2}\)", sent_norm):
            continue
        # (d) frase começa com referência bibliográfica "[33] proposed..."
        if re.match(r"^\[\d+\]", sent_norm):
            continue
        for pat in patterns:
            if re.search(pat, sent_norm, re.IGNORECASE):
                results.append(sent.strip())
                break
    return results


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

_OBJ_PATS = [
    # "The aim/goal/objective of this/the/our [current] paper/study is..."
    rf"\bthe\s+(aim|goal|objective|purpose|intent)\s+of\s+(this|the|our)\s+(?:\w+\s+)?({_PAPER_NOUNS})\b",
    # "This [optional_mod] paper/study [optional_adv] aims/presents/studies..."
    # Usa (?:...) para que o modificador opcional não consuma o noun
    rf"\bthis\s+(?:\w+\s+)?({_PAPER_NOUNS})\s+(?:\w+\s+)?({_OBJ_VERBS})\b",
    # "We aim / We propose / We present..."
    r"\bwe\s+(aim|seek|propose|present|investigate|examine|address|focus|contribute|develop|design|introduce|build)\b",
    # "The (main|primary) aim/goal..."
    r"\bthe\s+(main\s+|primary\s+|overall\s+)?(aim|goal|objective|purpose)\b",
    # "In this paper/study, we..."
    rf"\bin\s+this\s+({_PAPER_NOUNS}),?\s+we\b",
    # "Our aim/goal/objective..."
    r"\bour\s+(aim|goal|objective|purpose|approach)\b",
    # "The paper/study aims to..."
    rf"\bthe\s+({_PAPER_NOUNS})\s+(aims?|seeks?|intends?)\s+to\b",
]

_PROB_PATS = [
    r"\b(challenge|problem|issue|limitation|gap|drawback|shortcoming|concern|barrier|difficulty|obstacle)\b",
    r"\bstill\s+(remains?|lacks?|suffers?|faces?)\b",
    r"\bexisting\s+(approaches?|methods?|solutions?|systems?)\s+(fail|lack|do\s+not|are\s+not)\b",
    r"\bovercom(e|ing)\b",
    r"\bhowever.{0,80}(lack|fail|limit|problem|issue|challenge)\b",
    r"\bunforutnately\b",  # typo original mantido por compatibilidade
    r"\bunfortunately\b",
    r"\bto\s+(address|tackle|solve|overcome|mitigate|handle)\b",
    r"\b(lack|absence)\s+of\b",
    r"\bnot\s+(yet|fully|adequately|well)\s+(addressed|solved|studied|explored)\b",
]

_METH_PATS = [
    r"\b(methodology|method|approach|technique|algorithm|framework|protocol|experiment|evaluation|survey|interview|dataset|benchmark|simulation|testbed)\b",
    r"\bwe\s+(conduct|perform|carry\s+out|implement|develop|design|evaluate|train|test|collect|use|apply|employ)\b",
    r"\b(using|based\s+on|by\s+means\s+of).{0,60}(method|technique|approach|algorithm|model|framework)\b",
    r"\bthe\s+proposed\s+(method|approach|system|framework|algorithm|model)\b",
    r"\bdata\s+(were|was|are|is)\s+(collected|gathered|obtained|retrieved)\b",
    r"\b(quantitative|qualitative|mixed.?method)\s+(research|approach|design|analysis)\b",
]

_CONTRIB_PATS = [
    r"\bcontribut(e|es|ing|ion|ions)\s+to\b",
    r"\b(our|the\s+(?:main|key|major|primary))\s+(contribution|contributions)\b",
    # sem "the" antes de "main" (ex: "Main contributions of this paper are...")
    r"\b(main|key|major|primary)\s+(contribution|contributions)\b",
    r"\bwe\s+contribute\b",
    r"\bnovel\s+(contribution|approach|framework|method|technique|algorithm)\b",
    r"\bfirst\s+(paper|study|work)\s+to\b",
    r"\bto\s+the\s+best\s+of\s+(our|the\s+authors'?)\s+knowledge\b",
]

# Padrões de "accomplishment" — usados APENAS na seção de conclusão,
# onde as mesmas frases que seriam objetivo (introdução) significam realização.
_CONCL_PATS = [
    # "The findings/results suggest/indicate/reveal/establish..."
    r"\bthe\s+(findings?|results?|analysis|study|research|paper)\s+"
    r"(suggests?|indicates?|reveals?|demonstrates?|shows?|establishes?|highlights?|confirms?|identifies?|provides?)\b",
    # "This study/paper extends/complements/builds upon/advances..."
    rf"\bthis\s+(?:\w+\s+)?({_PAPER_NOUNS})\s+"
    r"(complements?|extends?|builds?\s+(?:on|upon)|advances?|improves?\s+upon|enriches?)\b",
    # "This study/paper also identificou/found/revealed..." — achievments em conclusão
    rf"\bthis\s+(?:\w+\s+)?({_PAPER_NOUNS})\s+(?:also\s+)?(?:\w+\s+)?"
    rf"({_OBJ_VERBS})\b",
    # "We found/identified/established/demonstrated..."
    r"\bwe\s+(found|identified|established|demonstrated|revealed|showed|confirmed|developed|built|created|designed)\b",
    # "The study/paper offers/provides/fills a gap..."
    rf"\bthe\s+(?:\w+\s+)?({_PAPER_NOUNS})\s+(offers?|provides?|fills?|addresses?)\b",
]


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
            return body_text[m.start():]
    # Fallback: últimos 8000 chars
    return body_text[-8000:]


def extract_structured_info(body_text: str) -> dict:
    """
    Extrai objetivo, problema, metodologia e contribuições do corpo do artigo.
    """
    # Usa os primeiros ~12 000 chars para objetivo e problema — artigos
    # Springer/MDPI têm cabeçalho longo + motivação antes da introdução
    intro = body_text[:12000]
    full = body_text
    conclusion = _extract_conclusion_section(body_text)

    objectives = _find_sentences_by_patterns(intro, _OBJ_PATS)
    problems = _find_sentences_by_patterns(intro, _PROB_PATS)
    methods = _find_sentences_by_patterns(full, _METH_PATS)

    # Contribuições explícitas (corpo todo) + accomplishments na conclusão
    contrib_explicit = _find_sentences_by_patterns(full, _CONTRIB_PATS)
    contrib_conclusion = _find_sentences_by_patterns(conclusion, _CONCL_PATS)

    # Une, remove duplicatas e evita overlap com objetivos já capturados.
    # Também descarta contribuições que claramente falam de OUTRO trabalho.
    _ref_other = re.compile(
        r"\b(that|those)\s+(paper|study|article|work)\b", re.IGNORECASE
    )
    obj_texts = {re.sub(r"\s+", " ", s).strip() for s in objectives}
    seen: set[str] = set()
    contributions: list[str] = []
    for c in contrib_explicit + contrib_conclusion:
        key = re.sub(r"\s+", " ", c).strip()
        if key in seen or key in obj_texts:
            continue
        if _ref_other.search(key):
            continue
        seen.add(key)
        contributions.append(c)

    # limita quantidade
    return {
        "objectives": objectives[:3],
        "problems": problems[:3],
        "methods": methods[:4],
        "contributions": contributions[:4],
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
