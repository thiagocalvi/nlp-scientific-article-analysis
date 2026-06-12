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
#   objetivo / problema / metodologia / contribuicao : texto-referência em
#       inglês (mesma língua dos artigos, para casar as palavras-chave com as
#       sentenças extraídas), redigido a partir do resumo/intro/conclusão.
#   ref_obs   : observação opcional sobre a contagem de referências.
# ──────────────────────────────────────────────────────────────────────────────
GABARITO: dict[str, dict] = {
    "ahamed2026": {
        "ref_count": 87,
        "objetivo": "understand the factors of cybersecurity awareness in Bangladesh "
                    "and the mediating influence of risk perception among Generation Z "
                    "university students",
        "problema": "Generation Z students are highly digitally active yet their "
                    "cybersecurity awareness is insufficient and increasingly important",
        "metodologia": "data collected from 398 university students in Dhaka using "
                       "stratified random sampling, analyzed with Partial Least Squares "
                       "Structural Equation Modeling (SmartPLS)",
        "contribuicao": "cybersecurity knowledge and social networking strongly "
                        "contribute to forming risk perception and awareness",
    },
    "alawida2026": {
        "ref_count": 41,
        "objetivo": "analyze cybersecurity insurance companies in the UAE, examining "
                    "question categorization and alignment with ISO 27001, NIST CSF and "
                    "the UAE national cybersecurity standard",
        "problema": "insurance application form questions do not explicitly and "
                    "comprehensively cover all relevant cybersecurity domains when "
                    "benchmarked against international and national standards",
        "metodologia": "four representative companies and their application forms were "
                       "selected and analyzed against two international and national "
                       "cybersecurity standards",
        "contribuicao": "shows that while companies extensively cover certain "
                        "cybersecurity aspects, some control areas are less explicitly "
                        "represented, revealing coverage gaps",
    },
    "alazzam2026": {
        "ref_count": 33,
        "objetivo": "assess the relative impact of cybersecurity challenges, legal "
                    "infrastructures and e-governance maturity on cyber threats and "
                    "client trust in UAE and Jordan financial institutions",
        "problema": "integrating artificial intelligence and the internet of things in "
                    "financial institutions raises legal, regulatory and cybersecurity "
                    "challenges",
        "metodologia": "quantitative research design with a survey of 400 employees of "
                       "financial institutions, analyzed with descriptive statistics, "
                       "reliability and Pearson correlation",
        "contribuicao": "identifies how cybersecurity challenges, legal infrastructure "
                        "and e-governance maturity affect cyber threats and customer trust",
    },
    "aljaradat2026": {
        "ref_count": 92,
        "objetivo": "examine key factors associated with digital payment use behaviour "
                    "in India such as effort expectancy, grievance redressal, trust, "
                    "performance expectancy, perceived cybersecurity risks and social "
                    "influence, and their moderating effects",
        "problema": "understanding the determinants and the role of trust and "
                    "cybersecurity in digital payment use behaviour",
        "metodologia": "survey data collected from urban regions in North India, "
                       "analyzed with structural equation modelling",
        "contribuicao": "effort expectancy, grievance redressal and performance "
                        "expectancy are positively associated with use behaviour; trust "
                        "has no direct association but relates to performance expectancy; "
                        "cybercrime experience moderates trust formation",
    },
    "alnaabi2026": {
        "ref_count": 65,
        "objetivo": "systematically examine artificial intelligence based methods for "
                    "enhancing cybersecurity capabilities in Oman and identify the "
                    "technical, organizational and regulatory challenges",
        "problema": "translating global AI-based cybersecurity approaches into effective "
                    "national practices remains challenging in emerging digital economies",
        "metodologia": "convergent mixed-methods approach integrating survey data from "
                       "138 practitioners with semi-structured interviews of 15 experts, "
                       "grounded in the Technology-Organization-Environment framework and "
                       "Diffusion of Innovations theory",
        "contribuicao": "strong relationship between AI adoption factors and "
                        "cybersecurity effectiveness; anomaly detection, Arabic natural "
                        "language processing and explainable AI are the most relevant "
                        "techniques",
    },
    "buczak2016": {
        "ref_count": 113,
        "objetivo": "present a survey of data mining and machine learning methods for "
                    "cyber security intrusion detection",
        "problema": "misuse, anomaly and hybrid intrusion detection techniques have "
                    "limitations such as inability to detect zero-day attacks and high "
                    "false alarm rates",
        "metodologia": "literature review describing and comparing machine learning and "
                       "data mining methods for intrusion detection, including complexity "
                       "and datasets",
        "contribuicao": "systematizes machine learning and data mining methods for "
                        "intrusion detection, discussing their applicability, the datasets "
                        "used and computational complexity",
    },
    "joshi2026": {
        "ref_count": 31,
        "objetivo": "conduct a bibliometric analysis of cybersecurity research in smart "
                    "cities and intelligent transport systems over the last ten years",
        "problema": "the increasing uptake of digital technologies in smart cities and "
                    "intelligent transport systems increases cybersecurity concerns on "
                    "critical infrastructure, public safety and data privacy",
        "metodologia": "bibliometric analysis of 170 documents retrieved from the Scopus "
                       "database",
        "contribuicao": "maps research themes such as risk management, AI-driven threat "
                        "detection, IoT security, governance and resilience, publication "
                        "trends after 2020 and persistent challenges",
    },
    "khraisat2019": {
        "ref_count": 90,
        "ref_obs": "valor de referência adotado do PRÓPRIO pipeline (não houve "
                   "contagem manual independente — a transcrição estava parcial); "
                   "logo o erro +0 não é validação cruzada. As 90 refs foram "
                   "conferidas uma a uma e estão íntegras.",
        "objetivo": "present a taxonomy of contemporary intrusion detection systems, a "
                    "comprehensive review of recent works and an overview of the datasets "
                    "used for evaluation",
        "problema": "cyber-attacks are becoming more sophisticated, presenting increasing "
                    "challenges in accurately detecting intrusions",
        "metodologia": "survey classifying intrusion detection into signature-based and "
                       "anomaly-based systems, reviewing techniques, datasets and evasion "
                       "techniques",
        "contribuicao": "provides a taxonomy of intrusion detection systems, a review of "
                        "notable works, common datasets, attacker evasion techniques and "
                        "future research challenges",
    },
    "kundu2026": {
        "ref_count": 66,
        "objetivo": "propose a trustworthy intrusion detection framework integrating Bald "
                    "Eagle Search optimization with the XGBoost classifier and SHAP-based "
                    "explainability",
        "problema": "intrusion detection systems need not only high accuracy but also "
                    "transparency and fairness, addressing gaps in current machine "
                    "learning cybersecurity solutions",
        "metodologia": "Bald Eagle Search algorithm fine-tunes XGBoost hyperparameters, "
                       "evaluated on a real-world cybersecurity dataset, with SHAP "
                       "explainability and fairness analysis",
        "contribuicao": "achieves high accuracy outperforming baseline XGBoost and GA, "
                        "ACO and PSO optimizers, while providing transparent feature "
                        "contributions and bias-aware detection",
    },
    "molina2017": {
        "ref_count": 30,
        "objetivo": "present decision aiding software to support cyber risk and threat "
                    "analysis and evaluate the most popular and relevant cybersecurity "
                    "risk assessment tools",
        "problema": "need to compare and evaluate cybersecurity risk analysis tools for "
                    "decision-making and prevention to protect organizational information "
                    "assets",
        "metodologia": "evaluation of the most popular risk assessment tools, analyzing "
                       "their properties, metrics and strategies",
        "contribuicao": "comparison of cybersecurity risk analysis tools and their "
                        "support for risk analysis, decision-making and prevention",
    },
    "oka2026": {
        "ref_count": 20,
        "ref_obs": "20 referências reais; o pipeline extrai 17 (perde 3 entradas de "
                   "autoria institucional/legal sem ano entre parênteses).",
        "objetivo": "examine how critical rail infrastructure operators are adopting and "
                    "operationalising the NIS 2 directive and identify the sector-specific "
                    "compliance challenges",
        "problema": "the NIS 2 directive imposes cybersecurity obligations on essential "
                    "services such as critical rail infrastructure, with partial and "
                    "inconsistent adoption across Member States",
        "metodologia": "qualitative document analysis of European Union Agency for "
                       "Cybersecurity (ENISA) reports with inductive and deductive coding "
                       "of the NIS 2 and Critical Entities Resilience directives",
        "contribuicao": "finds that adoption is partial and inconsistent and identifies "
                        "four themes: governance fragmentation, uneven cybersecurity "
                        "maturity, supplier dependency and incident reporting gaps",
    },
    "pandey2026": {
        "ref_count": 50,
        "objetivo": "examine how cybersecurity considerations are represented in medical "
                    "device procurement within the Indian public healthcare sector",
        "problema": "connected medical devices introduce cybersecurity risks and "
                    "procurement requirements need to incorporate cybersecurity terms and "
                    "controls at the design stage",
        "metodologia": "compile and standardize 760 e-procurement documents using natural "
                       "language processing techniques such as TF-IDF, NMF and regular "
                       "expressions, and map 123 CISA advisories to the MITRE CWE framework",
        "contribuicao": "reveals gaps between tender documents and CISA vulnerabilities by "
                        "bridging the two datasets of procurement documents and advisories",
    },
    "shama2026": {
        "ref_count": 70,
        "objetivo": "present a scientometric mapping of neuro-symbolic artificial "
                    "intelligence research in cybersecurity from 2016 to 2025",
        "problema": "neuro-symbolic AI integrates neural learning with symbolic reasoning "
                    "for cybersecurity but the field needs systematic mapping of paradigms "
                    "and research gaps",
        "metodologia": "scientometric analysis using keyword co-occurrence, bibliographic "
                       "coupling and systematic literature review of Scopus publications",
        "contribuicao": "identifies four thematic clusters, the dominant "
                        "Learning-for-Reasoning paradigm, mature versus underexplored "
                        "domains and research gaps in efficiency, evaluation and "
                        "explainability",
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
            gab_kw = _keywords(gab[campo])
            pred_txt = " ".join(info.get(chave, []))
            pred_kw = _keywords(pred_txt)
            inter = gab_kw & pred_kw
            recall = len(inter) / len(gab_kw) if gab_kw else 0.0
            capturado = recall >= LIMIAR_OVERLAP
            soma_recall[campo] += recall
            capturados[campo] += int(capturado)
            campos_art[campo] = {
                "recall_kw": round(recall, 3),
                "capturado": capturado,
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
    L.append("  Mede o quanto as sentenças extraídas pelo pipeline cobrem o conteúdo")
    L.append("  REAL de cada campo. Por artigo e por campo:")
    L.append("    1. do texto-referência do gabarito extraem-se as palavras-chave")
    L.append("       (minúsculas, sem stopwords, lematizadas);")
    L.append("    2. o mesmo é feito com as sentenças que o pipeline extraiu;")
    L.append("    3. recall_kw = |gabarito ∩ pipeline| / |gabarito|.")
    L.append(f"  'captura' = recall_kw >= {LIMIAR_OVERLAP}. Ex.: buczak/contribuição = 0.00")
    L.append("  porque o pipeline extraiu \"a higher percentage of papers present their")
    L.append("  approaches as offline methods\" — frase com a palavra-gatilho, mas que")
    L.append("  não contém nenhuma palavra-chave da contribuição real do artigo.")
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
    L.append("  [3] recall_kw = fração das palavras-chave do gabarito presentes")
    L.append("      nas sentenças extraídas pelo pipeline para o campo.")
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
