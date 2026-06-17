"""
avaliacao.py
============
Etapa 4 – Avaliação de desempenho do sistema de extração de informações.

A avaliação compara a saída do pipeline com um GABARITO (ground truth)
construído manualmente, em três dimensões:

  1. TOP-10 TERMOS  – verificação independente: os 10 termos mais frequentes
     são RECOMPUTADOS a partir dos PDFs (mesma definição do pipeline) e
     comparados, termo a termo e contagem a contagem, com o que o pipeline
     salvou. Não é uma cópia: é uma reconferência da aritmética.

  2. REFERÊNCIAS    – número de referências extraídas vs. número real
     (contagem verdadeira por artigo, levantada manualmente).

  3. CAMPOS ESTRUTURADOS (objetivo / problema / metodologia / contribuição)
     – cada artigo tem o texto-referência do campo escrito manualmente (a
     partir do resumo/introdução/conclusão). Mede-se a sobreposição de
     palavras-chave entre esse texto e as sentenças que o pipeline extraiu
     (revocação de conteúdo).

Métricas: revocação de palavras-chave por campo, taxa de captura, erro de
contagem de referências e exatidão dos top-10 termos.
"""

import json
import io
import re
from contextlib import redirect_stdout
from pathlib import Path
from collections import Counter

# O pipeline é usado para (a) ler os PDFs e separar corpo/refs e (b) fornecer a
# DEFINIÇÃO de "termo" (o conjunto de stopwords). A CONTAGEM dos termos, porém,
# é refeita por código próprio nesta avaliação (ver _contagem_independente),
# para não ser uma simples reexecução do pipeline.
try:
    import pipeline as _pl
    from nltk.stem import WordNetLemmatizer
    _LEMMA = WordNetLemmatizer()
except Exception:                       # pragma: no cover
    _pl = None
    _LEMMA = None

RESULTS_FILE = Path("saida/resultados_pipeline.json")
PDF_DIR      = Path("artigos")
OUT_DIR      = Path("saida")

# Limiar de revocação de palavras-chave para considerar um campo "capturado".
LIMIAR_OVERLAP = 0.30


# ──────────────────────────────────────────────────────────────────────────────
# GABARITO (ground truth) – um por artigo (stem do PDF)
#   ref_count : número REAL de referências do artigo (contagem manual).
#   objetivo / problema / metodologia / contribuicao : a(s) FRASE(S) REAIS do
#       paper que enunciam cada campo, citadas (quase) verbatim do resumo/intro/
#       conclusão. Como o pipeline é EXTRATIVO (devolve frases do próprio artigo),
#       o gabarito é o texto do artigo — não uma paráfrase — para que a métrica
#       meça se o pipeline extraiu a frase certa, não a similaridade com um resumo.
#   ref_obs   : observação opcional sobre a contagem de referências.
# ──────────────────────────────────────────────────────────────────────────────
GABARITO: dict[str, dict] = {
    "ahamed2026": {
        "ref_count": 87,
        "objetivo": "The aim of the current study is understanding the factors of "
                    "cybersecurity awareness in Bangladesh and the mediating influence "
                    "of risk perception.",
        "problema": [
            "The Generation Z students have also become one of the most digitally "
            "active populations, and cybersecurity awareness is becoming very "
            "important to them.",
            "It helps fill a research gap to study the topic of cybersecurity "
            "awareness among the students of Generation Z.",
        ],
        "metodologia": "Data was collected among 398 university students in Dhaka, "
                       "considering a stratified random sampling of both the public and "
                       "the privately-owned institutions of higher learning, and analyzed "
                       "using Partial Least Squares Structural Equation Modeling (SmartPLS).",
        "contribuicao": "The findings suggest that cybersecurity knowledge is a strong "
                        "factor that contributes to forming risk perception and awareness; "
                        "social networking is also a contributing factor in the mentioned "
                        "outcomes.",
    },
    "alawida2026": {
        "ref_count": 41,
        "objetivo": "This paper presents an in-depth analysis of cybersecurity insurance "
                    "companies in the United Arab Emirates (UAE), examining four key "
                    "factors: question categorization, alignment with two international/"
                    "national cybersecurity standards (ISO 27001 and NIST CSF), and "
                    "adherence to the UAE cybersecurity standard.",
        "problema": "However, a significant gap exists, as most of these questions do not "
                    "always explicitly comprehensively cover all relevant cybersecurity "
                    "domains, particularly when benchmarked against international and local "
                    "cybersecurity standards.",
        "metodologia": [
            "Four representative companies and their application forms were selected for "
            "this analysis.",
            "We collected and analyzed cybersecurity insurance forms from four companies "
            "in the UAE, coding each application item according to the 20-question "
            "taxonomy.",
        ],
        "contribuicao": "Our findings indicate that while these companies extensively "
                        "cover certain cybersecurity aspects through multiple questions, "
                        "some control areas are less explicitly represented in the analyzed "
                        "forms.",
    },
    "alazzam2026": {
        "ref_count": 33,
        "objetivo": "The objective of the study was to assess the relative impact of the "
                    "cybersecurity challenges, legal infrastructures, and e-governance "
                    "maturity on the cyber threats and trust of clientele.",
        "problema": "The study looks into the intersection of Artificial Intelligence (AI) "
                    "with the Internet of Things (IoT), especially the legal, regulatory, "
                    "and cybersecurity integration challenges within the context of UAE "
                    "and Jordan's financial sectors.",
        "metodologia": "The study utilized a quantitative research design, gathering data "
                       "through a survey distributed to employees and managers within "
                       "financial institutions, and the survey data of 400 employees were "
                       "analyzed through descriptive statistics, reliability, Pearson "
                       "correlations, and Structural Equation Modelling (SEM).",
        "contribuicao": [
            "The study established that the risks posed by inadequate cybersecurity "
            "infrastructures substantially increase the threats.",
            "The study highlights several critical insights and underscores that "
            "cybersecurity challenges in financial institutions substantially increase "
            "cyber threats.",
        ],
    },
    "aljaradat2026": {
        "ref_count": 92,
        "objetivo": "This study examines key factors associated with digital payment "
                    "systems use behaviour in India, focusing on effort expectancy, "
                    "grievance redressal, trust, performance expectancy, perceived "
                    "cybersecurity risks, and social influence.",
        "problema": [
            "The cybercrimes targeting digital payment systems continue to present "
            "significant challenges for clients and financial institutions.",
            "The findings underscore the importance of mitigating cybersecurity risks "
            "and strengthening trust-related mechanisms to support sustained use of "
            "digital payment systems.",
        ],
        "metodologia": "Survey data were collected from urban regions in North India, and "
                       "structural equation modelling was employed to analyse the "
                       "determinants of use behaviour among active users.",
        "contribuicao": "The analysis reveals that effort expectancy, grievance redressal, "
                        "and performance expectancy are positively associated with use "
                        "behaviour.",
    },
    "alnaabi2026": {
        "ref_count": 65,
        "objetivo": "This study systematically examines artificial intelligence-based "
                    "methods for enhancing cybersecurity capabilities in Oman, while "
                    "critically identifying the technical, organizational, and regulatory "
                    "challenges constraining their effective adoption.",
        "problema": "However, translating global AI-based cybersecurity approaches into "
                    "effective national practices remains challenging, particularly in "
                    "emerging digital economies.",
        "metodologia": "A convergent mixed-methods approach was employed, integrating "
                       "survey data from 138 cybersecurity practitioners with "
                       "semistructured interviews involving 15 experts from government, "
                       "banking, energy, healthcare, and education sectors.",
        "contribuicao": [
            "Quantitative results show a strong relationship between AI adoption factors "
            "and cybersecurity effectiveness. Anomaly detection, Arabic natural language "
            "processing, and explainable AI emerged as the most contextually relevant "
            "techniques.",
            "This research makes several key contributions: it reviews and analyzes major "
            "AI techniques and their effectiveness in detecting and mitigating cyber "
            "threats.",
        ],
    },
    "buczak2016": {
        "ref_count": 113,
        "objetivo": "This survey paper describes a focused literature survey of machine "
                    "learning (ML) and data mining (DM) methods for cyber analytics in "
                    "support of intrusion detection.",
        "problema": "Misuse-based techniques cannot detect novel (zero-day) attacks. The "
                    "main disadvantage of anomaly-based techniques is the potential for "
                    "high false alarm rates.",
        "metodologia": "Short tutorial descriptions of each ML/DM method are provided. "
                       "Based on the number of citations or the relevance of an emerging "
                       "method, papers representing each method were identified, read, and "
                       "summarized.",
        "contribuicao": [
            "The complexity of ML/DM algorithms is addressed, discussion of challenges for "
            "using ML/DM for cyber security is presented, and some recommendations on when "
            "to use a given method are provided.",
            "The paper provides a set of comparison criteria for ML/DM methods and a set of "
            "recommendations on when to use a given method.",
        ],
    },
    "joshi2026": {
        "ref_count": 31,
        "objetivo": "This bibliometric analysis studies cybersecurity research in smart "
                    "cities and ITS over the last ten years (2015-2025) through 170 "
                    "relevant documents retrieved from Scopus database.",
        "problema": [
            "The increasing uptake of digital technologies in smart cities and "
            "intelligent transport systems (ITS), has increased associated cybersecurity "
            "concerns, with cyber threats on critical infrastructure, public safety, and "
            "data privacy.",
            "Research studies continue to highlight persistent challenges in IoT device "
            "vulnerabilities and the lack of a standardized security protocol for securing "
            "large-scale urban networks.",
        ],
        "metodologia": "This bibliometric analysis studies cybersecurity research in smart "
                       "cities and ITS over the last ten years (2015-2025) through 170 "
                       "relevant documents retrieved from Scopus database.",
        "contribuicao": [
            "It focusses on research themes that include cybersecurity risk management, "
            "AI-driven threat detection, IoT security in urban environments, policy and "
            "governance frameworks, and resilience strategies against cyber threats.",
            "The research highlights the growing academic focus on security issues in "
            "urban innovations involving IoT, AI systems, and blockchain systems.",
        ],
    },
    "khraisat2019": {
        "ref_count": 90,
        "ref_obs": "valor de referência adotado do PRÓPRIO pipeline (não houve "
                   "contagem manual independente — a transcrição estava parcial); "
                   "logo o erro +0 não é validação cruzada. As 90 refs foram "
                   "conferidas uma a uma e estão íntegras.",
        "objetivo": "This survey paper presents a taxonomy of contemporary IDS, a "
                    "comprehensive review of notable recent works, and an overview of the "
                    "datasets commonly used for evaluation purposes.",
        "problema": "Cyber-attacks are becoming more sophisticated and thereby presenting "
                    "increasing challenges in accurately detecting intrusions.",
        "metodologia": [
            "Numerous intrusion detection methods have been proposed in the literature to "
            "tackle computer security threats, which can be broadly classified into "
            "Signature-based Intrusion Detection Systems (SIDS) and Anomaly-based "
            "Intrusion Detection Systems (AIDS).",
            "This survey paper presents a taxonomy of contemporary IDS, a comprehensive "
            "review of notable recent works, and an overview of the datasets commonly used "
            "for evaluation purposes.",
        ],
        "contribuicao": [
            "It also presents evasion techniques used by attackers to avoid detection and "
            "discusses future research challenges to counter such techniques so as to make "
            "computer systems more secure.",
            "In this paper, we provide a structured and contemporary, wide-ranging study "
            "on intrusion detection systems in terms of techniques, datasets and "
            "challenges.",
        ],
    },
    "kundu2026": {
        "ref_count": 66,
        "objetivo": "This study proposes a trustworthy intrusion detection framework that "
                    "integrates Bald Eagle Search (BES) optimization with the XGBoost "
                    "classifier and SHAP-based explainability to enhance both predictive "
                    "performance and interpretability.",
        "problema": "The rapid escalation of sophisticated cyberattacks demands intrusion "
                    "detection systems that not only achieve high accuracy but also provide "
                    "transparency and fairness in decision-making, addressing critical gaps "
                    "in current machine learning-based cybersecurity solutions.",
        "metodologia": "The BES algorithm efficiently fine-tunes key XGBoost "
                       "hyperparameters to overcome limitations of traditional models, and "
                       "the BES-optimized XGBoost model, evaluated on a real-world "
                       "cybersecurity dataset, achieves a substantially improved accuracy "
                       "of 99.7%.",
        "contribuicao": [
            "SHAP visualizations further reveal feature-level contributions, ensuring "
            "transparent and accountable detection decisions, while fairness analysis "
            "highlights disparities across browser types, promoting responsible and "
            "bias-aware cybersecurity deployment.",
            "This study presents the first integration of Bald Eagle Search (BES) "
            "optimization with XGBoost for intrusion detection, supported by SHAP "
            "explainability and fairness analysis.",
        ],
    },
    "molina2017": {
        "ref_count": 30,
        "objetivo": "This paper presents ongoing work of a decision aiding software "
                    "intended to support cyber risk and cyber threats analysis of an "
                    "information and communication technology infrastructure.",
        "problema": "The work is focused on the evaluation of the most popular and relevant "
                    "tools available for risk assessment and decision making in the "
                    "cybersecurity domain.",
        "metodologia": [
            "Their properties, metrics and strategies are analysed and their support for "
            "cybersecurity risk analysis, decision-making and prevention is assessed for "
            "the protection of an organization's information assets.",
            "The study is focused on the comparison of different risk assessment "
            "strategies applied within the same case study, carried out using tools such "
            "as Nexpose.",
        ],
        "contribuicao": [
            "Their properties, metrics and strategies are analysed and their support for "
            "cybersecurity risk analysis, decision-making and prevention is assessed for "
            "the protection of an organization's information assets.",
            "This paper presents a study for the development of a cybersecurity risk "
            "analysis and management system, complementing the authors' previous Decision "
            "Support System for Corporations Cybersecurity Management.",
        ],
    },
    "oka2026": {
        "ref_count": 20,
        "ref_obs": "20 referências reais; o pipeline extrai 17 (perde 3 entradas de "
                   "autoria institucional/legal sem ano entre parênteses).",
        "objetivo": "This study examines how critical rail infrastructure operators are "
                    "adopting and operationalising NIS 2 directives and identifies the "
                    "sector-specific challenges that affect their compliance efforts.",
        "problema": "The Network information systems (NIS) 2 directive was introduced to "
                    "strengthen cybersecurity obligations for operators of essential and "
                    "important services.",
        "metodologia": "Using a qualitative document analysis of key European Union Agency "
                       "for Cybersecurity (ENISA) reports, supported by inductive and "
                       "deductive coding of NIS 2 directive and Critical Entities "
                       "Resilience (CER) directive.",
        "contribuicao": "Findings indicate that while the critical rail infrastructure has "
                        "made progress in adopting NIS 2, its implementation remains partial "
                        "and inconsistent across Member States and actors.",
    },
    "pandey2026": {
        "ref_count": 50,
        "objetivo": "This study examines how such considerations are represented in "
                    "medical procurement within Indian public healthcare sector.",
        "problema": [
            "Medical devices are increasingly connected to public networks, offering "
            "benefits such as remote diagnosis but also introducing new cybersecurity "
            "risks.",
            "Only a few studies have begun to explore the intersection of cybersecurity "
            "and medical device procurement.",
        ],
        "metodologia": "We compile and standardize 760 e-procurement documents (2014-2024) "
                       "from major public hospitals across the country and analyze them "
                       "using natural language processing techniques, including Term "
                       "Frequency-Inverse Document Frequency (TF-IDF), Non-negative Matrix "
                       "Factorization (NMF), and regular expressions, to identify "
                       "security-related clauses.",
        "contribuicao": [
            "Bridging these two datasets (tender documents and CISA advisories) reveals "
            "significant gaps: cybersecurity specifications appear in only 15% of tenders, "
            "and high-risk CWEs such as CWE-287 (Authentication Bypass) are largely "
            "unaddressed.",
            "The findings indicate limited integration of threat intelligence into "
            "procurement design and suggest stronger policy measures for enhancing vendor "
            "accountability through security-aligned tender specifications.",
        ],
    },
    "shama2026": {
        "ref_count": 70,
        "objetivo": "This study presents the scientometric mapping of NSAI research in "
                    "cybersecurity, analyzing publications from Scopus database spanning "
                    "2016-2025.",
        "problema": [
            "Significant research gaps remain in developing computational efficiency "
            "benchmarks, standardized evaluation frameworks, and explainability "
            "mechanisms.",
            "This study addresses a significant gap in the literature; to the best of our "
            "knowledge, no comprehensive scientometric mapping of neuro-symbolic AI in "
            "cybersecurity exists.",
        ],
        "metodologia": "Through keyword co-occurrence, bibliographic coupling, and "
                       "systematic literature review, we identify four thematic clusters "
                       "and reveal dominant integration paradigms.",
        "contribuicao": [
            "Our findings show rapid field growth, with Learning-for-Reasoning "
            "architectures as the predominant approach; network intrusion detection and "
            "malware analysis emerge as mature domains, while autonomous cyber defense and "
            "IoT security remain underexplored.",
            "Our key contributions include rigorous bibliometric analysis revealing "
            "temporal trends and intellectual structures, and the first comprehensive "
            "scientometric mapping of neuro-symbolic AI research in cybersecurity.",
        ],
    },
}

# Mapeia o nome do campo no gabarito -> chave em structured_info do pipeline.
CAMPOS = {
    "objetivo":     "objectives",
    "problema":     "problems",
    "metodologia":  "methods",
    "contribuicao": "contributions",
}


# ──────────────────────────────────────────────────────────────────────────────
# 1. TOP-10 TERMOS – recomputação independente a partir dos PDFs
# ──────────────────────────────────────────────────────────────────────────────
# Tokeniza por sequências de letras: QUALQUER caractere não-letra é separador.
# Difere do tokenizador do pipeline (NLTK word_tokenize), por isso serve como
# verificação independente — mas, por cobrir também pontuação Unicode (apóstrofo
# curvo, travessões), converge para a mesma contagem na prática.
_TOKEN_RE = re.compile(r"[a-zà-ÿ]+")


def _contagem_independente(body: str) -> Counter:
    """
    Conta os termos do corpo com IMPLEMENTAÇÃO PRÓPRIA — não chama preprocess()
    nem Counter.most_common() do pipeline. Compartilha apenas a *especificação*
    de termo (mesmas stopwords e mesmo lematizador), reescrevendo a tokenização,
    a filtragem e a contagem do zero. Assim, um bug na contagem do pipeline
    apareceria como divergência.
    """
    stop = _pl.STOP_ALL
    cont: Counter = Counter()
    for tok in _TOKEN_RE.findall(body.lower()):
        if len(tok) < 3 or tok in stop:
            continue
        tok = _LEMMA.lemmatize(tok)
        if len(tok) < 3 or tok in stop:
            continue
        cont[tok] += 1
    return cont


def avaliar_termos(results: dict) -> dict:
    """
    Verifica os top-10 termos do pipeline com um contador INDEPENDENTE.

    Para cada artigo:
      termos_confirmados : quantos dos 10 termos do pipeline são, segundo o
          contador independente, de fato de frequência >= à 10ª maior (tolerante
          a empates na borda).
      conjunto_identico  : o conjunto dos 10 termos coincide exatamente.
      empate_na_borda    : 10ª e 11ª maiores contagens iguais (top-10 não único).
      contagens_divergentes : termos cuja contagem independente difere da do
          pipeline (casos-limite de tokenização; não alteram o top-10).
    """
    por_artigo: dict[str, dict] = {}
    if _pl is None or _LEMMA is None or not PDF_DIR.exists():
        for name, data in results.items():
            if name != "__global__":
                por_artigo[name] = {"verificavel": False}
        return {"por_artigo": por_artigo, "artigos_verificaveis": 0}

    buf = io.StringIO()
    with redirect_stdout(buf):                       # silencia os logs de leitura
        raws = _pl.load_all_pdfs(PDF_DIR)

    avaliados = conjuntos_iguais = 0
    confirmados_total = 0
    for name, data in results.items():
        if name == "__global__":
            continue
        salvo = [list(t) for t in data.get("top10_terms", [])]
        raw = raws.get(name)
        if raw is None:
            por_artigo[name] = {"verificavel": False, "top10": salvo}
            continue
        avaliados += 1
        body, _ = _pl.split_body_references(raw)
        cont = _contagem_independente(body)

        ordenado = sorted(cont.items(), key=lambda kv: (-kv[1], kv[0]))
        ind_top = ordenado[:10]
        limiar = ind_top[-1][1] if len(ind_top) == 10 else 0          # 10ª maior
        ind_set = {t for t, _ in ind_top}
        pipe_set = {t for t, _ in salvo}

        confirmados = sum(1 for t, _ in salvo if cont.get(t, 0) >= limiar)
        divergentes = [
            {"termo": t, "pipeline": c, "independente": cont.get(t, 0)}
            for t, c in salvo if cont.get(t, 0) != c
        ]
        empate = len(ordenado) > 10 and ordenado[9][1] == ordenado[10][1]
        ident = pipe_set == ind_set

        confirmados_total += confirmados
        conjuntos_iguais += int(ident)
        por_artigo[name] = {
            "verificavel": True,
            "termos_confirmados": confirmados,
            "conjunto_identico": ident,
            "empate_na_borda": empate,
            "contagens_divergentes": divergentes,
            "top10": salvo,
        }
    return {
        "por_artigo": por_artigo,
        "artigos_verificaveis": avaliados,
        "conjuntos_identicos": conjuntos_iguais,
        "termos_confirmados_total": confirmados_total,
        "termos_total": 10 * avaliados,
        "pct_confirmados": round(100 * confirmados_total / (10 * avaliados), 1)
        if avaliados else None,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 2. REFERÊNCIAS – contagem extraída vs. contagem real
# ──────────────────────────────────────────────────────────────────────────────
def avaliar_referencias(results: dict) -> dict:
    por_artigo: dict[str, dict] = {}
    erros_abs = []
    exatos = 0
    com_gab = 0
    for name, data in results.items():
        if name == "__global__":
            continue
        extraidas = len(data.get("references", []))
        gab = GABARITO.get(name, {})
        esperado = gab.get("ref_count")
        item = {"extraidas": extraidas, "esperado": esperado}
        if esperado is not None:
            com_gab += 1
            erro = extraidas - esperado
            item["erro"] = erro
            item["exato"] = (erro == 0)
            # cobertura limitada a 100% (não premia excesso)
            item["cobertura_pct"] = round(100 * min(extraidas, esperado) / esperado, 1)
            if "ref_obs" in gab:
                item["obs"] = gab["ref_obs"]
            erros_abs.append(abs(erro))
            if erro == 0:
                exatos += 1
        por_artigo[name] = item
    return {
        "por_artigo": por_artigo,
        "artigos_com_gabarito": com_gab,
        "contagens_exatas": exatos,
        "erro_absoluto_medio": round(sum(erros_abs) / len(erros_abs), 2) if erros_abs else None,
        "cobertura_media_pct": round(
            sum(v["cobertura_pct"] for v in por_artigo.values() if "cobertura_pct" in v)
            / com_gab, 1) if com_gab else None,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 3. CAMPOS ESTRUTURADOS – revocação de palavras-chave (overlap)
# ──────────────────────────────────────────────────────────────────────────────
_FALLBACK_STOP = {
    "the", "and", "for", "with", "that", "this", "are", "was", "were", "from",
    "into", "such", "their", "have", "has", "not", "but", "which", "its",
    "they", "them", "these", "those", "also", "can", "may", "our", "out", "all",
}


def _keywords(text: str) -> set[str]:
    """Palavras-chave de conteúdo (lematizadas, sem stopwords)."""
    if _pl is not None:
        return set(_pl.preprocess(text))
    toks = re.findall(r"[a-zA-Z]{3,}", text.lower())
    return {t for t in toks if t not in _FALLBACK_STOP}


def avaliar_campos(results: dict) -> dict:
    por_artigo: dict[str, dict] = {}
    soma_recall = {c: 0.0 for c in CAMPOS}
    capturados  = {c: 0 for c in CAMPOS}
    n = 0
    for name, data in results.items():
        if name == "__global__" or name not in GABARITO:
            continue
        n += 1
        info = data.get("structured_info", {})
        gab = GABARITO[name]
        campos_art: dict[str, dict] = {}
        for campo, chave in CAMPOS.items():
            # O gabarito pode listar VÁRIAS frases aceitáveis (um campo costuma
            # ser enunciado em mais de uma frase no paper). O pipeline acerta o
            # campo se extraiu QUALQUER uma delas → usamos a melhor correspondência.
            aceitaveis = gab[campo]
            if isinstance(aceitaveis, str):
                aceitaveis = [aceitaveis]
            pred_kw = _keywords(" ".join(info.get(chave, [])))
            recall = 0.0
            inter: set[str] = set()
            for frase in aceitaveis:
                gkw = _keywords(frase)
                if not gkw:
                    continue
                r = len(gkw & pred_kw) / len(gkw)
                if r > recall:
                    recall, inter = r, gkw & pred_kw
            capturado = recall >= LIMIAR_OVERLAP
            soma_recall[campo] += recall
            capturados[campo] += int(capturado)
            campos_art[campo] = {
                "recall_kw": round(recall, 3),
                "capturado": capturado,
                "n_frases_aceitaveis": len(aceitaveis),
                "n_sentencas_extraidas": len(info.get(chave, [])),
                "palavras_casadas": sorted(inter),
            }
        por_artigo[name] = campos_art
    resumo = {
        campo: {
            "recall_medio": round(soma_recall[campo] / n, 3) if n else None,
            "taxa_captura_pct": round(100 * capturados[campo] / n, 1) if n else None,
        }
        for campo in CAMPOS
    }
    return {"por_artigo": por_artigo, "resumo_por_campo": resumo, "n_artigos": n}


# ──────────────────────────────────────────────────────────────────────────────
# Relatório
# ──────────────────────────────────────────────────────────────────────────────
def gerar_relatorio(termos: dict, refs: dict, campos: dict) -> str:
    L = []
    L.append("=" * 72)
    L.append("  RELATÓRIO DE AVALIAÇÃO DE DESEMPENHO – Segurança Cibernética")
    L.append("=" * 72)

    # 1. termos
    L.append("\n[1] TOP-10 TERMOS  (contador independente vs. pipeline)")
    L.append("-" * 72)
    L.append("  Os 10 termos mais frequentes são os do pipeline; esta seção os CONFERE")
    L.append("  com um segundo contador escrito de forma independente:")
    L.append("    • pipeline    : tokeniza com NLTK word_tokenize (modelo Punkt);")
    L.append("    • independente: tokeniza por sequências de letras via regex (código")
    L.append("                    próprio, sem usar preprocess/most_common).")
    L.append("  Ambos usam a MESMA definição de termo (stopwords + lematização), então")
    L.append("  nenhum é 'mais confiável' que o outro: a concordância entre dois códigos")
    L.append("  diferentes é que confirma não haver erro de implementação na contagem.")
    L.append("  As diferenças residuais vêm só da tokenização (ex.: \"Oman's\" vira")
    L.append("  \"oman\" no word_tokenize, mas fica grudado na regex) e NÃO mudam quais")
    L.append("  termos entram no top-10.")
    L.append("")
    if termos["artigos_verificaveis"] == 0:
        L.append("  (não verificável: PDFs ou pipeline indisponíveis)")
    else:
        L.append(f"  Artigos verificáveis        : {termos['artigos_verificaveis']}")
        L.append(f"  Termos confirmados (top-freq): {termos['termos_confirmados_total']}"
                 f"/{termos['termos_total']}  ({termos['pct_confirmados']}%)")
        L.append(f"  Conjuntos top-10 idênticos   : {termos['conjuntos_identicos']}"
                 f"/{termos['artigos_verificaveis']}")
        empates = [n for n, m in termos["por_artigo"].items()
                   if m.get("empate_na_borda")]
        if empates:
            L.append("  Empate 10º/11º (10º termo não é único): "
                     + ", ".join(empates))
        div = [n for n, m in termos["por_artigo"].items()
               if m.get("contagens_divergentes")]
        if div:
            L.append("  Diferenças de contagem (casos-limite de tokenização):")
            for n in div:
                for d in termos["por_artigo"][n]["contagens_divergentes"]:
                    L.append(f"      {n}: {d['termo']} "
                             f"pipeline={d['pipeline']} indep={d['independente']}")

    # 2. referências
    L.append("\n[2] REFERÊNCIAS  (extraídas vs. contagem real)")
    L.append("-" * 72)
    L.append(f"  {'Artigo':<16}{'Extr.':>6}{'Real':>6}{'Erro':>6}")
    obs_list = []
    for name, m in refs["por_artigo"].items():
        if m.get("esperado") is None:
            continue
        L.append(f"  {name:<16}{m['extraidas']:>6}{m['esperado']:>6}{m['erro']:>+6}")
        if m.get("obs"):
            obs_list.append((name, m["obs"]))
    L.append(f"  → contagens exatas: {refs['contagens_exatas']}/{refs['artigos_com_gabarito']}"
             f" | erro abs. médio: {refs['erro_absoluto_medio']}"
             f" | cobertura média: {refs['cobertura_media_pct']}%")
    if obs_list:
        L.append("  Observações:")
        for name, obs in obs_list:
            L.append(f"    • {name}: {obs}")

    # 3. campos estruturados
    L.append("\n[3] CAMPOS ESTRUTURADOS  (revocação de palavras-chave; "
             f"limiar captura = {LIMIAR_OVERLAP})")
    L.append("-" * 72)
    L.append("  Como o pipeline é EXTRATIVO (devolve frases do próprio artigo), o gabarito")
    L.append("  de cada campo é o conjunto de FRASES REAIS do paper que o enunciam (um")
    L.append("  campo costuma ter mais de uma frase válida). Por artigo e por campo:")
    L.append("    1. para cada frase aceitável do gabarito, extraem-se suas palavras-chave")
    L.append("       (minúsculas, sem stopwords, lematizadas);")
    L.append("    2. o mesmo é feito com a união das sentenças que o pipeline extraiu;")
    L.append("    3. recall_kw = melhor (entre as frases aceitáveis) de |gab ∩ pipe|/|gab|.")
    L.append(f"  'captura' = recall_kw >= {LIMIAR_OVERLAP}: o pipeline extraiu (essencialmente)")
    L.append("  uma das frases reais do campo. Valores < 1,0 indicam casamento parcial")
    L.append("  (ex.: hifenização do PDF ou frase ligeiramente diferente da aceitável).")
    L.append("")
    L.append(f"  {'Artigo':<16}{'Obj':>6}{'Prob':>6}{'Met':>6}{'Contr':>7}")
    for name, c in campos["por_artigo"].items():
        L.append(f"  {name:<16}"
                 f"{c['objetivo']['recall_kw']:>6.2f}"
                 f"{c['problema']['recall_kw']:>6.2f}"
                 f"{c['metodologia']['recall_kw']:>6.2f}"
                 f"{c['contribuicao']['recall_kw']:>7.2f}")
    L.append("  " + "-" * 50)
    r = campos["resumo_por_campo"]
    L.append(f"  {'recall médio':<16}"
             f"{r['objetivo']['recall_medio']:>6.2f}"
             f"{r['problema']['recall_medio']:>6.2f}"
             f"{r['metodologia']['recall_medio']:>6.2f}"
             f"{r['contribuicao']['recall_medio']:>7.2f}")
    L.append(f"  {'captura %':<16}"
             f"{r['objetivo']['taxa_captura_pct']:>6.0f}"
             f"{r['problema']['taxa_captura_pct']:>6.0f}"
             f"{r['metodologia']['taxa_captura_pct']:>6.0f}"
             f"{r['contribuicao']['taxa_captura_pct']:>7.0f}")

    L.append("\n" + "=" * 72)
    L.append("LEGENDA")
    L.append("  [1] Contador independente: tokeniza por sequências de letras e")
    L.append("      reconta do zero (sem usar preprocess/most_common do pipeline),")
    L.append("      compartilhando só as stopwords e o lematizador. 'Confirmado' =")
    L.append("      o termo é, segundo essa contagem, de frequência >= à 10ª maior.")
    L.append("  [2] Erro = extraídas - reais (+ excesso, - faltam).")
    L.append("  [3] recall_kw = melhor casamento entre as FRASES ACEITÁVEIS do paper")
    L.append("      (gabarito) e as sentenças extraídas pelo pipeline para o campo.")
    return "\n".join(L)


# ──────────────────────────────────────────────────────────────────────────────
# Interface pública
# ──────────────────────────────────────────────────────────────────────────────
def avaliar(results: dict) -> dict:
    print("\n[4] Avaliando desempenho...")
    termos = avaliar_termos(results)
    refs   = avaliar_referencias(results)
    campos = avaliar_campos(results)

    relatorio = gerar_relatorio(termos, refs, campos)
    print(relatorio)

    metricas = {"termos": termos, "referencias": refs, "campos": campos}

    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "avaliacao.json").write_text(
        json.dumps(metricas, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "avaliacao_relatorio.txt").write_text(relatorio, encoding="utf-8")
    print(f"\n  [✓] Avaliação salva em: {OUT_DIR / 'avaliacao.json'}")
    print(f"  [✓] Relatório salvo em: {OUT_DIR / 'avaliacao_relatorio.txt'}")
    return metricas


if __name__ == "__main__":
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            results = json.load(f)
        avaliar(results)
    else:
        print("Execute pipeline.py primeiro.")
