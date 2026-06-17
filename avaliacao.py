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
        "problema": "However, the development of cybercrimes in Bangladesh remains high, "
                    "which confirms the necessity to enhance cybersecurity awareness, "
                    "especially in youth, who are responsible to the largest extent of the "
                    "digitalization of this country.",
        "metodologia": [
            "Data was collected among 398 university students in Dhaka, "
            "considering a stratified random sampling of both the public and "
            "the privately-owned institutions of higher learning, and analyzed "
            "using Partial Least Squares Structural Equation Modeling (SmartPLS 3.3.9).",
            "In the present study, researchers adopted a positivist research strategy, a "
            "method that facilitates the acquisition of empirical knowledge by formulating "
            "hypotheses rooted in observations.",
            "Stratified random sampling was employed based on the type of university "
            "(public vs. private) to ensure representation across different educational "
            "institutions in Dhaka.",
            "To validate the proposed research model, we employed Partial Least Squares "
            "Structural Equation Modeling (PLS-SEM) using SmartPLS version 3.3.9, "
            "complemented by SPSS v25 for descriptive statistics and demographic profiling.",
            "To assess the significance of path coefficients and mediation effects, we "
            "applied a bootstrapping technique with 5000 resamples, a non-parametric "
            "method suitable for testing complex relationships.",
        ],
        "contribuicao": [
            "The findings suggest that cybersecurity knowledge is a strong "
            "factor that contributes to forming risk perception and awareness; "
            "social networking is also a contributing factor in the mentioned "
            "outcomes.",
            "This study gives practical advice for teachers and policymakers to "
            "improve youth cybersecurity.",
            "It adds new insights by focusing on Gen Z in a developing country "
            "during the post-pandemic period.",
            "In this way, the study contributes to the global discussion of cybersecurity "
            "by placing these dynamics in the context of a developing country, providing "
            "not only theoretical complement but also empirical advice on ways to encourage "
            "cyber resilience in the youth going forward.",
            "First, it updates the Protection Motivation Theory to a setting in a developing "
            "country and demonstrates that the knowledge-to-risk-perception channel will "
            "work robustly and perhaps even more effectively with low literacy levels.",
        ],
    },
    "alawida2026": {
        "ref_count": 41,
        "objetivo": "This paper presents an in-depth analysis of cybersecurity insurance "
                    "companies in the United Arab Emirates (UAE), examining four key "
                    "factors: question categorization, alignment with two international/"
                    "national cybersecurity standards (ISO 27001 and NIST CSF), and "
                    "adherence to the UAE cybersecurity standard.",
        "problema": [
            "However, a significant gap exists, as most of these questions do not "
            "always explicitly comprehensively cover all relevant cybersecurity "
            "domains, particularly when benchmarked against international and local "
            "cybersecurity standards.",
            "However, there are still many questions regarding what aspects are covered, "
            "the role of insurance providers, and how such insurance contributes to overall "
            "organizational security.",
        ],
        "metodologia": [
            "Four representative companies and their application forms were selected for "
            "this analysis.",
            "We collected and analyzed cybersecurity insurance forms from four major "
            "insurance providers.",
        ],
        "contribuicao": [
            "This paper provides a comprehensive analysis and discussion of various "
            "cyberattacks and corresponding defense mechanisms, alongside actionable "
            "recommendations for improving questionnaire coverage and alignment to "
            "enhance their documentation and assessment practices.",
            "Our contribution can be summarized in following twofold: (i) we systematically "
            "identify security domains that are missing or under-represented in UAE "
            "cybersecurity insurance questionnaires, and (ii) we propose a baseline set of "
            "'recommendation' questions that insurers can adopt to improve cyber-risk "
            "exposure assessment.",
            "To address these gaps, we propose a recommended baseline questionnaire of ten "
            "essential questions grounded in international and national standards, along "
            "with cross-validation items and a prioritised improvement roadmap.",
        ],
    },
    "alazzam2026": {
        "ref_count": 33,
        "objetivo": [
            "The objective of the study was to assess the relative impact of the "
            "cybersecurity challenges, legal infrastructures, and e-governance "
            "maturity on the cyber threats and trust of clientele.",
            "The study looks into the intersection of Artificial Intelligence (AI) "
            "with the Internet of Things (IoT), especially the legal, regulatory, "
            "and cybersecurity integration challenges within the context of UAE "
            "and Jordan's financial sectors.",
        ],
        "problema": [
            "Few studies investigated the extent to which the policies and laws in "
            "place address the key cybersecurity risks.",
            "However, the adoption of these technologies poses serious legal and "
            "cybersecurity issues.",
        ],
        "metodologia": [
            "The study utilized a quantitative research design, gathering data through a "
            "survey distributed to employees and managers within a number of financial "
            "institutions.",
            "With a data sample of 400 employees, the survey data were analyzed through a "
            "variety of methods, such as descriptive statistics, reliability, Pearson "
            "correlations, and Structural Equation Modelling (SEM).",
            "The study employs quantitative methods to gain insight into the intersection "
            "of legal issues and cybersecurity concerns regarding the adoption of "
            "Artificial Intelligence (AI) and the Internet of Things (IoT) in financial "
            "institutions in the UAE and Jordan.",
            "In relation to sampling, a stratified random sampling approach was used, which "
            "enhances the study with representation by institution type, employee role, and "
            "level of digital technology integration.",
            "The size of the sample was 400, with 200 participants from each of the UAE and "
            "Jordan, which is adequate for conducting SEM analysis.",
            "Data were analyzed using IBM AMOS and SmartPLS.",
        ],
        "contribuicao": "",
    },
    "aljaradat2026": {
        "ref_count": 92,
        "objetivo": [
            "This study examines key factors associated with digital payment "
            "systems use behaviour in India, focusing on effort expectancy, "
            "grievance redressal, trust, performance expectancy, perceived "
            "cybersecurity risks, and social influence.",
            "This research examines associations between perceived cybersecurity risk, "
            "effort expectancy, grievance redressal mechanisms, performance expectancy, "
            "trust, social influence, and digital payment use behaviour in North India.",
        ],
        "problema": [
            "The cybercrimes targeting digital payment systems continue to present "
            "significant challenges for clients and financial institutions.",
            "However, alongside this rapid growth, the need to fortify cybersecurity "
            "measures has become increasingly critical.",
        ],
        "metodologia": [
            "Survey data were collected from urban regions in North India, and "
            "structural equation modelling was employed to analyse the "
            "determinants of use behaviour among active users.",
            "The questionnaire used in this study was adapted from an instrument previously "
            "validated and published in our earlier work.",
            "Data were collected between March 2023 and March 2024 using a multi-stage "
            "purposive (non-probability) sampling strategy designed to ensure that "
            "respondents were informed and relevant participants, namely, individuals with "
            "direct experience in formal banking and potential exposure to digital payment "
            "systems.",
            "Respondents were invited to participate only if they met the inclusion "
            "criterion of being active bank account holders, ensuring substantive relevance "
            "to digital payment use behaviour.",
        ],
        "contribuicao": [
            "The findings underscore the importance of mitigating cybersecurity risks "
            "and strengthening trust-related mechanisms to support sustained use, "
            "offering practical recommendations for policymakers and digital payment "
            "providers.",
            "This research offers new perspectives on the relationships among trust, "
            "cybersecurity concerns, and digital payment use behaviour.",
            "Extending the UTAUT framework with PCR, GR, and trust, the findings indicate "
            "that PE, EE, and GR are significantly associated with use behaviour, "
            "highlighting the salience of perceived performance benefits, ease of use, and "
            "accessible complaint-resolution mechanisms in explaining recent use patterns.",
        ],
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
        "metodologia": [
            "A convergent mixed-methods approach was employed, integrating "
            "survey data from 138 cybersecurity practitioners with "
            "semistructured interviews involving 15 experts from government, "
            "banking, energy, healthcare, and education sectors.",
            "A convergent parallel mixed-methods design was applied to assess AI's role in "
            "enhancing cybersecurity in Oman, involving the simultaneous collection and "
            "integration of quantitative and qualitative data to answer the research "
            "questions comprehensively.",
        ],
        "contribuicao": [
            "This research makes several key contributions: Reviews and analyzes major AI "
            "techniques and their effectiveness in detecting and mitigating cyber threats "
            "globally.",
            "Assesses the current state of AI adoption in Oman's cybersecurity "
            "infrastructure.",
            "Identifies barriers and opportunities for AI integration in national "
            "cybersecurity.",
            "Proposes recommendations to enhance AI adoption in cybersecurity practices.",
            "Develops a localized AI-based cybersecurity framework (OCAIF) tailored to "
            "Oman's needs and Vision 2040 goals.",
            "This study establishes a context-aware foundation for advancing AI-enabled "
            "cybersecurity in Oman.",
        ],
    },
    "buczak2016": {
        "ref_count": 113,
        "objetivo": "This survey paper describes a focused literature survey of machine "
                    "learning (ML) and data mining (DM) methods for cyber analytics in "
                    "support of intrusion detection.",
        "problema": "",
        "metodologia": "Short tutorial descriptions of each ML/DM method are provided. "
                       "Based on the number of citations or the relevance of an emerging "
                       "method, papers representing each method were identified, read, and "
                       "summarized.",
        "contribuicao": [
            "The complexity of ML/DM algorithms is addressed, discussion of challenges for "
            "using ML/DM for cyber security is presented, and some recommendations on when "
            "to use a given method are provided.",
            "The paper provides a set of comparison criteria for ML/DM methods and a set of "
            "recommendations on the best methods to use depending on the characteristics of "
            "the cyber problem to solve.",
        ],
    },
    "joshi2026": {
        "ref_count": 31,
        "objetivo": [
            "This bibliometric analysis studies cybersecurity research in smart "
            "cities and ITS over the last ten years (2015-2025) through 170 "
            "relevant documents retrieved from Scopus database.",
            "It focusses on research themes that include cybersecurity risk management, "
            "AI-driven threat detection, IoT security in urban environments, policy and "
            "governance frameworks, and resilience strategies against cyber threats.",
        ],
        "problema": [
            "The increasing uptake of digital technologies in smart cities and "
            "intelligent transport systems (ITS), has increased associated cybersecurity "
            "concerns, with cyber threats on critical infrastructure, public safety, and "
            "data privacy.",
            "Research studies continue to highlight persistent challenges in IoT device "
            "vulnerabilities and the lack of a standardized security protocol for securing "
            "large-scale urban networks.",
        ],
        "metodologia": [
            "The bibliometric analysis was done using the Bibliometrix R package "
            "Biblioshiny (version 5.0), which allowed to the performance analysis, "
            "mapping of keyword co-occurrence and trends in publication over time.",
            "The PRISMA guideline was used for reporting the review studies (see Fig. 1).",
            "A structured and systematic bibliometric analysis is used in this study to "
            "analyze cybersecurity in smart cities and ITS.",
            "Bibliometric data were obtained from the Scopus database via the search "
            "string: ('cyber security' OR 'transportation' OR 'smart city'), which is "
            "composed of the publications with titles, abstracts or keywords relating to "
            "security of computer systems, transportation networks and urban research.",
            "The dataset was extracted on 10 February 2025, thus guaranteeing the use of a "
            "consistent temporal cut-off throughout the analysis.",
        ],
        "contribuicao": [
            "The research highlights the growing academic focus on security issues in "
            "urban innovations involving IoT, AI systems, and blockchain systems.",
            "Proposes a unified model linking AI, blockchain, and IoT security "
            "interactions in smart cities.",
            "Outlines key themes on AI security, blockchain authentication, and "
            "IoT-enabled intelligent mobility systems.",
            "Identifies security gaps and highlights future directions such as "
            "quantum-safe cryptography and AI-based cyber defence.",
            "Theoretically, this research identifies major research themes and their "
            "trajectory and emphasizes IoT security, AI-based cyber defence, blockchain for "
            "secure payment, and smart governance models.",
            "The bibliometric analysis establishes the key influencers, institutions, and "
            "citation patterns that have influenced this field.",
        ],
    },
    "khraisat2019": {
        "ref_count": 90,
        "objetivo": [
            "This survey paper presents a taxonomy of contemporary IDS, a "
            "comprehensive review of notable recent works, and an overview of the "
            "datasets commonly used for evaluation purposes.",
            "This paper presents a discussion on IDS dataset problems which are of main "
            "concern to the research community in the area of network intrusion detection "
            "systems (NIDS).",
        ],
        "problema": [
            "Cyber-attacks are becoming more sophisticated and thereby presenting "
            "increasing challenges in accurately detecting intrusions.",
            "Failure to prevent the intrusions could degrade the credibility of security "
            "services, e.g. data confidentiality, integrity, and availability.",
            "SIDS has difficulty in detecting zero-day attacks for the reason that no "
            "matching signature exists in the database until the signature of the new "
            "attack is extracted and stored.",
        ],
        "metodologia": "",
        "contribuicao": [
            "It also presents evasion techniques used by attackers to avoid detection and "
            "discusses future research challenges to counter such techniques so as to make "
            "computer systems more secure.",
            "In this paper, we provide a structured and contemporary, wide-ranging study "
            "on intrusion detection system in terms of techniques and datasets; and also "
            "highlight challenges of the techniques and then make recommendations.",
            "In this paper, we have presented, in detail, a survey of intrusion detection "
            "system methodologies, types, and technologies with their advantages and "
            "limitations.",
        ],
    },
    "kundu2026": {
        "ref_count": 66,
        "objetivo": "This study proposes a trustworthy intrusion detection framework that "
                    "integrates Bald Eagle Search (BES) optimization with the XGBoost "
                    "classifier and SHAP-based explainability to enhance both predictive "
                    "performance and interpretability.",
        "problema": [
            "The rapid escalation of sophisticated cyberattacks demands intrusion "
            "detection systems that not only achieve high accuracy but also provide "
            "transparency and fairness in decision-making, addressing critical gaps "
            "in current machine learning-based cybersecurity solutions.",
            "However, its performance is highly dependent on hyperparameter tuning, and "
            "suboptimal configurations frequently lead to overfitting or unstable behavior.",
        ],
        "metodologia": "The BES algorithm efficiently fine-tunes key XGBoost hyperparameters to "
                       "overcome limitations of traditional models that struggle with complex, "
                       "high-dimensional intrusion patterns.",
        "contribuicao": [
            "SHAP visualizations further reveal feature-level contributions, ensuring "
            "transparent and accountable detection decisions, while fairness analysis "
            "highlights disparities across browser types, promoting responsible and "
            "bias-aware cybersecurity deployment.",
            "This study presents the first integration of Bald Eagle Search (BES) "
            "optimization with XGBoost for intrusion detection, addressing the "
            "shortcomings of poorly tuned boosting models in cybersecurity.",
            "The proposed model incorporates a SHAP-based explainability layer, providing "
            "transparent, instance-level explanations and overcoming the black-box "
            "limitations of most IDS models.",
            "The work includes a fairness-aware analysis, evaluating detection consistency "
            "across browser-type contexts—an aspect largely missing in existing intrusion "
            "detection research.",
            "A multi-objective model design optimizes accuracy, interpretability, "
            "transparency, and fairness, significantly improving real-world deployment "
            "suitability.",
            "Comparative experiments demonstrate that BES-tuned XGBoost outperforms GA, "
            "ACO, and PSO variants and achieves state-of-the-art accuracy while preserving "
            "interpretability.",
        ],
    },
    "molina2017": {
        "ref_count": 30,
        "objetivo": [
            "This paper presents ongoing work of a decision aiding software "
            "intended to support cyber risk and cyber threats analysis of an "
            "information and communication technology infrastructure.",
            "The work is focused on the evaluation of the most popular and relevant "
            "tools available for risk assessment and decision making in the "
            "cybersecurity domain.",
            "This article analyses different tools for ICT infrastructure data collection, "
            "vulnerability scanning and the support they can provide for cybersecurity risk "
            "assessment and decision making in organizations.",
            "This paper presents a study for the development of a cybersecurity risk "
            "analysis and management system.",
        ],
        "problema": "",
        "metodologia": [
            "Their properties, metrics and strategies are analysed and their support for "
            "cybersecurity risk analysis, decision-making and prevention is assessed for "
            "the protection of an organization's information assets.",
            "The study is focused on the comparison of different risk assessment "
            "strategies applied within the same case study.",
        ],
        "contribuicao": "",
    },
    "oka2026": {
        "ref_count": 20,
        "ref_obs": "20 referências reais; o pipeline extrai 17 (perde 3 entradas de "
                   "autoria institucional/legal sem ano entre parênteses).",
        "objetivo": [
            "This study examines how critical rail infrastructure operators are "
            "adopting and operationalising NIS 2 directives and identifies the "
            "sector-specific challenges that affect their compliance efforts.",
            "This paper examines the sector-specific adoption of the NIS 2 directive within "
            "European critical rail infrastructure through document analysis.",
        ],
        "problema": "Existing literature on the NIS 2 directive has primarily focused on "
                    "conceptual interpretations, legal analysis, or its limitations with "
                    "few studies assessing how individual sectors operationalise NIS 2 "
                    "directives.",
        "metodologia": [
            "Using a qualitative document analysis of key European Union Agency "
            "for Cybersecurity (ENISA) reports, supported by inductive and "
            "deductive coding of NIS 2 directive and Critical Entities "
            "Resilience (CER) directive.",
            "This study adopts a qualitative research design to examine sector-specific "
            "adoption of the NIS 2 directives within the European critical rail "
            "infrastructure.",
            "The analysis followed a combined inductive-deductive manual coding strategy.",
            "As critical rail infrastructure operators do not publish internal "
            "cybersecurity documentation, publicly available ENISA reports represent the "
            "most reliable empirical evidence base for understanding current practices "
            "within European critical rail infrastructure.",
        ],
        "contribuicao": "When compared with ENISA's technical implementation guidance, these findings "
                        "reveal that NIS 2 provides a practical guide for strengthening critical rail "
                        "infrastructure resilience, but significant operational, organisational, and "
                        "supply-chain barriers must still be addressed.",
    },
    "pandey2026": {
        "ref_count": 50,
        "objetivo": [
            "This study examines how such considerations are represented in "
            "medical procurement within Indian public healthcare sector.",
            "This paper presents a novel, data-driven analysis of e-procurement documents "
            "from Indian medical institutions (2014-2024) to evaluate the integration of "
            "cybersecurity into the procurement process.",
        ],
        "problema": [
            "Medical devices are increasingly connected to public networks, offering "
            "benefits such as remote diagnosis but also introducing new cybersecurity "
            "risks.",
            "Only a few studies have begun to explore this intersection.",
        ],
        "metodologia": [
            "We compile and standardize 760 e-procurement documents (2014-2024) "
            "from major public hospitals across the country and analyze them "
            "using natural language processing techniques, including Term "
            "Frequency-Inverse Document Frequency (TF-IDF), Non-negative Matrix "
            "Factorization (NMF), and regular expressions, to identify "
            "security-related clauses.",
            "In parallel, 123 CISA advisories (2018-2024) highlighting vulnerabilities in "
            "medical systems are mapped to the MITRE Common Weakness Enumeration (CWE) "
            "framework.",
            "It employs a dual-stream analytical approach to investigate the cybersecurity "
            "gap between known vulnerabilities in medical devices and the requirements "
            "specified in public procurement documents.",
            "The methodology is structured into two parallel workflows: one for analyzing "
            "ICS security advisories and another for analyzing e-procurement tenders.",
            "The study sampling frame comprises two complementary data sources: Industrial "
            "Control Systems (ICS) medical advisories issued by the Cybersecurity and "
            "Infrastructure Security Agency and procurement documents collected from the "
            "official e-tender portals of 17 government-run All India Institutes of Medical "
            "Sciences (AIIMS) across India.",
        ],
        "contribuicao": "By comparing data from two corpora, Indian medical procurements and CISA "
                        "cybersecurity advisories, we were able to create an accurate contrast between "
                        "the security needs outlined by experts and those addressed in procurement "
                        "processes.",
    },
    "shama2026": {
        "ref_count": 70,
        "objetivo": "This study presents the scientometric mapping of NSAI research in "
                    "cybersecurity, analyzing publications from Scopus database spanning "
                    "2016-2025.",
        "problema": "This study addresses a significant gap in the literature; to the best "
                    "of our knowledge, no comprehensive scientometric mapping of "
                    "neuro-symbolic AI (NSAI) research in cybersecurity has been conducted "
                    "to date.",
        "metodologia": [
            "Accordingly, this study adopts a dual-methodology framework integrating "
            "scientometric analysis with a systematic literature review (SLR).",
            "Scientometric analysis employs quantitative bibliometric techniques to map "
            "the intellectual structure of the field through examination of publication "
            "metadata, citation relations, and collaboration networks.",
        ],
        "contribuicao": [
            "Our key contributions include: (1) rigorous bibliometric analysis revealing "
            "temporal trends and intellectual structures; (2) systematic taxonomies of "
            "integration architectures, neural components, and symbolic frameworks; (3) "
            "identification of critical gaps including computational efficiency "
            "benchmarking, standardized evaluation frameworks, and explainability "
            "integration.",
            "This study provides the first comprehensive scientometric mapping of "
            "neuro-symbolic AI research in cybersecurity, analyzing 262 publications "
            "(2016-2025) with systematic review of 63 high-quality studies.",
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
    soma_recall    = {c: 0.0 for c in CAMPOS}
    soma_precision = {c: 0.0 for c in CAMPOS}
    soma_f1        = {c: 0.0 for c in CAMPOS}
    capturados     = {c: 0   for c in CAMPOS}
    n = 0
    for name, data in results.items():
        if name == "__global__" or name not in GABARITO:
            continue
        n += 1
        info = data.get("structured_info", {})
        gab = GABARITO[name]
        campos_art: dict[str, dict] = {}
        for campo, chave in CAMPOS.items():
            aceitaveis = gab[campo]
            if isinstance(aceitaveis, str):
                aceitaveis = [aceitaveis] if aceitaveis else []
            pred_sents = info.get(chave, [])

            # pré-computa conjuntos de keywords (descarta vazios)
            gab_kws  = [kw for f in aceitaveis if (kw := _keywords(f))]
            pred_kws = [kw for p in pred_sents  if (kw := _keywords(p))]

            # recall: por frase do gabarito, cobertura da melhor sentença extraída
            # mean sobre todas as frases → pipeline precisa cobrir TODAS, não só uma
            if not gab_kws:
                recall = 1.0          # nada para recuperar → perfeito por convenção
            elif not pred_kws:
                recall = 0.0
            else:
                recall = sum(
                    max(len(gkw & pkw) / len(gkw) for pkw in pred_kws)
                    for gkw in gab_kws
                ) / len(gab_kws)

            # precision: por sentença extraída, relevância p/ a melhor frase do gabarito
            # mean sobre todas as sentenças → penaliza extrair mais do que o necessário
            if not pred_kws:
                precision = 1.0       # nada extraído → nada errado (vacuamente perfeito)
            elif not gab_kws:
                precision = 0.0       # extraiu algo mas gabarito vazio → tudo irrelevante
            else:
                precision = sum(
                    max(len(gkw & pkw) / len(pkw) for gkw in gab_kws)
                    for pkw in pred_kws
                ) / len(pred_kws)

            f1 = (2 * precision * recall / (precision + recall)
                  if (precision + recall) > 0 else 0.0)
            capturado = f1 >= LIMIAR_OVERLAP

            soma_recall[campo]    += recall
            soma_precision[campo] += precision
            soma_f1[campo]        += f1
            capturados[campo]     += int(capturado)
            campos_art[campo] = {
                "recall_kw":             round(recall,    3),
                "precision_kw":          round(precision, 3),
                "f1_kw":                 round(f1,        3),
                "capturado":             capturado,
                "n_frases_gabarito":     len(gab_kws),
                "n_sentencas_extraidas": len(pred_kws),
            }
        por_artigo[name] = campos_art
    resumo = {
        campo: {
            "recall_medio":    round(soma_recall[campo]    / n, 3) if n else None,
            "precision_media": round(soma_precision[campo] / n, 3) if n else None,
            "f1_medio":        round(soma_f1[campo]        / n, 3) if n else None,
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
    L.append("\n[3] CAMPOS ESTRUTURADOS  (Precision / Recall / F1 por palavra-chave; "
             f"limiar captura F1 = {LIMIAR_OVERLAP})")
    L.append("-" * 72)
    L.append("  Recall   : por frase do gabarito, cobertura da MELHOR sentença extraída")
    L.append("             (média sobre todas as frases → pipeline precisa cobrir TODAS).")
    L.append("  Precision: por sentença extraída, relevância p/ a MELHOR frase do gabarito")
    L.append("             (média sobre todas as sentenças → penaliza extração em excesso).")
    L.append("  F1       : média harmônica de P e R.  'captura' = F1 >= limiar.")
    L.append("  Palavras-chave: minúsculas, sem stopwords, lematizadas.")
    L.append("")
    L.append(f"  {'Artigo':<16}  {'Obj':>5}  {'Prob':>5}  {'Met':>5}  {'Contr':>5}   (F1)")
    for name, c in campos["por_artigo"].items():
        L.append(f"  {name:<16}  "
                 f"{c['objetivo']['f1_kw']:>5.2f}  "
                 f"{c['problema']['f1_kw']:>5.2f}  "
                 f"{c['metodologia']['f1_kw']:>5.2f}  "
                 f"{c['contribuicao']['f1_kw']:>5.2f}")
    L.append("  " + "-" * 52)
    r = campos["resumo_por_campo"]
    L.append(f"  {'F1 médio':<16}  "
             f"{r['objetivo']['f1_medio']:>5.2f}  "
             f"{r['problema']['f1_medio']:>5.2f}  "
             f"{r['metodologia']['f1_medio']:>5.2f}  "
             f"{r['contribuicao']['f1_medio']:>5.2f}")
    L.append(f"  {'recall médio':<16}  "
             f"{r['objetivo']['recall_medio']:>5.2f}  "
             f"{r['problema']['recall_medio']:>5.2f}  "
             f"{r['metodologia']['recall_medio']:>5.2f}  "
             f"{r['contribuicao']['recall_medio']:>5.2f}")
    L.append(f"  {'precision média':<16}  "
             f"{r['objetivo']['precision_media']:>5.2f}  "
             f"{r['problema']['precision_media']:>5.2f}  "
             f"{r['metodologia']['precision_media']:>5.2f}  "
             f"{r['contribuicao']['precision_media']:>5.2f}")
    L.append(f"  {'captura %':<16}  "
             f"{r['objetivo']['taxa_captura_pct']:>5.0f}  "
             f"{r['problema']['taxa_captura_pct']:>5.0f}  "
             f"{r['metodologia']['taxa_captura_pct']:>5.0f}  "
             f"{r['contribuicao']['taxa_captura_pct']:>5.0f}")

    L.append("\n" + "=" * 72)
    L.append("LEGENDA")
    L.append("  [1] Contador independente: tokeniza por sequências de letras e")
    L.append("      reconta do zero (sem usar preprocess/most_common do pipeline),")
    L.append("      compartilhando só as stopwords e o lematizador. 'Confirmado' =")
    L.append("      o termo é, segundo essa contagem, de frequência >= à 10ª maior.")
    L.append("  [2] Erro = extraídas - reais (+ excesso, - faltam).")
    L.append("  [3] Recall  = mean_i max_j |kw(g_i) ∩ kw(p_j)| / |kw(g_i)|")
    L.append("      Precision= mean_j max_i |kw(g_i) ∩ kw(p_j)| / |kw(p_j)|")
    L.append("      F1 = 2PR/(P+R).  g_i = frase do gabarito, p_j = sentença extraída.")
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
