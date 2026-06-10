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
# 1.  LEITURA DE PDFs  –  suporte a múltiplos formatos e layouts - talvez definir somente uma formato para simplificar a implementação
# ──────────────────────────────────────────────────────────────────────────────

# Largura mínima (pts) de uma coluna para considerar layout de 2 colunas.
# 1 ponto ≈ 0,35 mm.  Colunas de artigos IEEE/ACM têm ~200-250 pts cada.
_MIN_COLUMN_WIDTH = 200


# Rever essa implementação - retornar o texto mais estruturado
def _clean_extracted_text(input_text: str) -> str:
    """
    Pós-processamento aplicado ao texto extraído do PDF:
      • reconecta palavras hifenizadas no final da linha (ignorando espaços/tabs fantasmas)
      • limpa espaços desnecessários nas pontas de cada linha
      • remove cabeçalhos/rodapés repetitivos (números de página isolados)
      • colapsa espaços múltiplos e linhas em branco excessivas
    """

    # Hifenização: O \s* garante que ignoraremos espaços/tabs antes ou depois da quebra de linha
    input_text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", input_text)

    # Limpeza de linhas e remoção de páginas
    linhas_limpas = []
    for ln in input_text.splitlines():
        # O strip() remove espaços no começo e fim. Linhas só com espaços viram strings vazias "".
        ln_stripped = ln.strip()

        # Se a linha limpa for apenas um número de até 4 dígitos (página), ignoramos.
        if re.fullmatch(r"\d{1,4}", ln_stripped):
            continue

        linhas_limpas.append(ln_stripped)

    input_text = "\n".join(linhas_limpas)

    # Colapsa 3+ quebras de linha em apenas 2.
    # Agora isso funciona perfeitamente porque garantimos com o strip() que as linhas vazias não têm espaços
    input_text = re.sub(r"\n{3,}", "\n\n", input_text)

    # Remove caracteres de controle exceto newline/tab
    input_text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", input_text)

    return input_text.strip()


def _read_with_fitz(path: Path) -> str:
    doc = fitz.open(path)
    textos_ordenados = []

    for page in doc:
        # Extrai a página como uma lista de blocos.
        # Cada bloco é uma tupla: (x0, y0, x1, y1, texto, numero_do_bloco, tipo_do_bloco)
        blocos = page.get_text("blocks")

        # Filtra para pegar apenas blocos que são texto (tipo 0). Imagens são tipo 1.
        blocos_texto = [b for b in blocos if b[6] == 0]

        # Ordenamos os blocos primeiro pela coordenada Horizontal (x0), depois pela Vertical (y0).
        # O round(x0 / 50) agrupa blocos que estão na mesma coluna, mesmo que levemente tortos.
        blocos_texto.sort(
            key=lambda b: (round(int(b[0]) / 50), b[1])
        )  # TODO: Revisar essa implementação!

        for b in blocos_texto:
            texto_do_bloco = b[4].strip()
            if texto_do_bloco:
                textos_ordenados.append(texto_do_bloco)

    doc.close()

    # Junta os blocos com uma quebra de linha dupla para isolar bem cada parágrafo/referência
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
    print("*" * 50)
    print("TEXTO DE REFERÊNCIA SENDO PROCESSADO")
    print("*" * 50)
    print(ref_text)

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
    """Retorna trechos do texto que casam com qualquer dos padrões."""
    results = []
    sentences = sent_tokenize(text)
    for sent in sentences:
        for pat in patterns:
            if re.search(pat, sent, re.IGNORECASE):
                results.append(sent.strip())
                break
    return results


# Padrões para cada campo
_OBJ_PATS = [
    r"\bthe (aim|goal|objective|purpose|intent) of this (paper|study|article|work)\b",
    r"\bthis (paper|study|article|work) (aims|seeks|intends|proposes|presents|investigates|examines|focuses|addresses)\b",
    r"\bwe (aim|seek|propose|present|investigate|examine|address|focus)\b",
    r"\bthe (main |primary )?(aim|goal|objective|purpose)\b",
    r"\bin this (paper|study|article|work),?\s+we\b",
    r"\bour (aim|goal|objective|purpose)\b",
]

_PROB_PATS = [
    r"\b(challenge|problem|issue|limitation|gap|drawback|shortcoming|concern)\b",
    r"\bstill (remains|lacks|suffers|faces)\b",
    r"\bexisting (approaches|methods|solutions|systems) (fail|lack|do not|are not)\b",
    r"\bovercom(e|ing)\b",
    r"\bhowever[,\s]+.{0,80}(lack|fail|limit|problem|issue|challenge)\b",
    r"\bunforutnately\b",
    r"\bto (address|tackle|solve|overcome)\b",
]

_METH_PATS = [
    r"\b(methodology|method|approach|technique|algorithm|framework|protocol|experiment|evaluation|survey|interview|analysis|dataset|benchmark|simulation|testbed)\b",
    r"\bwe (conduct|perform|carry out|implement|develop|design|evaluate|train|test)\b",
    r"\b(using|based on|by means of)\b.{0,60}(method|technique|approach|algorithm|model|framework)\b",
    r"\bthe proposed (method|approach|system|framework|algorithm)\b",
]

_CONTRIB_PATS = [
    r"\bcontribut(e|es|ing|ion|ions) to\b",
    r"\bour (contribution|contributions)\b",
    r"\bthe (main |key |major |primary )?(contribution|contributions)\b",
    r"\bwe contribute\b",
    r"\bnovel (contribution|approach|framework|method|technique)\b",
    r"\bfirst (paper|study|work) to\b",
]


def extract_structured_info(body_text: str) -> dict:
    """
    Extrai objetivo, problema, metodologia e contribuições do corpo do artigo.
    """
    # foca na introdução (primeiros ~4000 chars) para objetivo e problema
    intro = body_text[:4000]
    full = body_text

    objectives = _find_sentences_by_patterns(intro, _OBJ_PATS)
    problems = _find_sentences_by_patterns(intro, _PROB_PATS)
    methods = _find_sentences_by_patterns(full, _METH_PATS)
    contributions = _find_sentences_by_patterns(full, _CONTRIB_PATS)

    # elimina contribuições que são na verdade objetivos
    contributions = [
        c
        for c in contributions
        if not any(re.search(p, c, re.IGNORECASE) for p in _OBJ_PATS)
    ]

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
        print(
            "debug------------------------------ REFERÊNCIAS --------------------------------"
        )
        print(refs_text)

        print(
            "debug------------------------------ BODY --------------------------------"
        )
        print(body)

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
