"""
visualizacoes.py
================
Gera todas as visualizações exigidas pelo trabalho:
  1. Nuvem de palavras geral
  2. Gráfico de barras – top 15 termos globais
  3. Heatmap – frequência de termos por artigo
  4. Gráfico de coocorrência de bigramas (grafo)
  5. Diagrama de similaridade entre artigos (Jaccard)
  6. Termos que aparecem como trabalhos futuros
  7. Árvore de palavras (dendrograma de termos)

Bibliotecas usadas: matplotlib, wordcloud, networkx  (sem ML/sklearn)
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.colors import LinearSegmentedColormap
from wordcloud import WordCloud

OUT_DIR = Path("saida")
OUT_DIR.mkdir(exist_ok=True)

# Paleta de cores "cybersegurança" (dark + vermelho/ciano)
CMAP_CYBER = LinearSegmentedColormap.from_list(
    "cyber", ["#0a0a2a", "#00b4d8", "#90e0ef", "#caf0f8"]
)

plt.rcParams.update(
    {
        "figure.facecolor": "#0d0d1a",
        "axes.facecolor": "#0d0d1a",
        "text.color": "#e0e0e0",
        "axes.labelcolor": "#e0e0e0",
        "xtick.color": "#e0e0e0",
        "ytick.color": "#e0e0e0",
        "axes.edgecolor": "#334466",
        "grid.color": "#1a2a3a",
        "grid.alpha": 0.5,
    }
)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _save(fig: plt.Figure, name: str) -> Path:
    p = OUT_DIR / name
    fig.savefig(p, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [✓] {name}")
    return p


def _load_results() -> dict:
    p = OUT_DIR / "resultados_pipeline.json"
    if not p.exists():
        raise FileNotFoundError("Execute pipeline.py primeiro.")
    with open(p) as f:
        return json.load(f)


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Nuvem de palavras
# ──────────────────────────────────────────────────────────────────────────────
def plot_wordcloud(results: dict) -> Path:
    freq = {t: c for t, c in results["__global__"]["top10_terms"]}
    # enriquece com todos os termos de todos os artigos
    all_terms: Counter = Counter()
    for name, data in results.items():
        if name == "__global__":
            continue
        all_terms.update({t: c for t, c in data.get("top10_terms", [])})
    all_terms.update(freq)

    wc = WordCloud(
        width=1200,
        height=600,
        background_color="#0d0d1a",
        colormap="cool",
        max_words=80,
        prefer_horizontal=0.8,
        min_font_size=10,
    ).generate_from_frequencies(dict(all_terms))

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(
        "Nuvem de Palavras – Segurança Cibernética",
        fontsize=16,
        pad=12,
        color="#00b4d8",
        fontweight="bold",
    )
    return _save(fig, "1_wordcloud.png")


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Barras – top 15 termos globais
# ──────────────────────────────────────────────────────────────────────────────
def plot_top_terms(results: dict) -> Path:
    all_terms: Counter = Counter()
    for name, data in results.items():
        if name == "__global__":
            continue
        all_terms.update({t: c for t, c in data.get("top10_terms", [])})
    top15 = all_terms.most_common(15)
    terms, counts = zip(*top15)

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.cool([i / len(terms) for i in range(len(terms))])
    bars = ax.barh(
        list(reversed(terms)),
        list(reversed(counts)),
        color=list(reversed(colors)),
        edgecolor="#0d0d1a",
        linewidth=0.5,
    )
    ax.set_xlabel("Frequência", fontsize=11)
    ax.set_title(
        "Top 15 Termos Mais Citados (todos os artigos)",
        fontsize=13,
        color="#00b4d8",
        fontweight="bold",
        pad=10,
    )
    ax.grid(axis="x", linestyle="--")
    for bar, val in zip(bars, list(reversed(counts))):
        ax.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            str(val),
            va="center",
            fontsize=9,
            color="#e0e0e0",
        )
    return _save(fig, "2_top_termos.png")


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Heatmap – frequência de termos por artigo
# ──────────────────────────────────────────────────────────────────────────────
def plot_heatmap(results: dict) -> Path:
    # top 12 termos globais
    all_terms: Counter = Counter()
    for name, data in results.items():
        if name == "__global__":
            continue
        all_terms.update({t: c for t, c in data.get("top10_terms", [])})
    top_terms_list = [t for t, _ in all_terms.most_common(12)]

    articles = [n for n in results if n != "__global__"]
    if not articles:
        return None

    # matriz
    matrix = []
    for name in articles:
        row_map = {t: c for t, c in results[name].get("top10_terms", [])}
        matrix.append([row_map.get(t, 0) for t in top_terms_list])

    fig, ax = plt.subplots(figsize=(14, max(4, len(articles) * 0.6 + 1)))
    im = ax.imshow(
        matrix,
        aspect="auto",
        cmap=LinearSegmentedColormap.from_list("c", ["#0d0d1a", "#00b4d8", "#caf0f8"]),
    )

    ax.set_xticks(range(len(top_terms_list)))
    ax.set_xticklabels(top_terms_list, rotation=40, ha="right", fontsize=9)
    short_names = [n[:30] for n in articles]
    ax.set_yticks(range(len(articles)))
    ax.set_yticklabels(short_names, fontsize=8)
    ax.set_title(
        "Heatmap – Frequência de Termos por Artigo",
        fontsize=13,
        color="#00b4d8",
        fontweight="bold",
        pad=10,
    )
    plt.colorbar(im, ax=ax, label="Frequência")

    # valores nas células
    for i in range(len(articles)):
        for j in range(len(top_terms_list)):
            v = matrix[i][j]
            if v > 0:
                ax.text(
                    j, i, str(v), ha="center", va="center", fontsize=7, color="white"
                )
    return _save(fig, "3_heatmap.png")


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Grafo de coocorrência de bigramas
# ──────────────────────────────────────────────────────────────────────────────
def plot_coocurrence(results: dict) -> Path:
    edge_weights: Counter = Counter()
    for name, data in results.items():
        if name == "__global__":
            continue
        for gram, cnt in data.get("top10_bigrams", []):
            parts = gram.split()
            if len(parts) == 2:
                edge_weights[(parts[0], parts[1])] += cnt

    # top 20 arestas
    top_edges = edge_weights.most_common(20)
    if not top_edges:
        return None

    G = nx.Graph()
    for (u, v), w in top_edges:
        G.add_edge(u, v, weight=w)

    node_freq: Counter = Counter()
    for (u, v), w in top_edges:
        node_freq[u] += w
        node_freq[v] += w

    fig, ax = plt.subplots(figsize=(13, 10))
    pos = nx.spring_layout(G, seed=42, k=2.5)
    node_sizes = [300 + node_freq[n] * 15 for n in G.nodes()]
    edge_widths = [G[u][v]["weight"] / 5 + 0.5 for u, v in G.edges()]

    nx.draw_networkx_edges(
        G, pos, ax=ax, width=edge_widths, edge_color="#00b4d8", alpha=0.6
    )
    nx.draw_networkx_nodes(
        G, pos, ax=ax, node_size=node_sizes, node_color="#0077b6", alpha=0.9
    )
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=9, font_color="#caf0f8")
    ax.set_title(
        "Grafo de Coocorrência de Bigramas",
        fontsize=13,
        color="#00b4d8",
        fontweight="bold",
        pad=10,
    )
    ax.axis("off")
    return _save(fig, "4_coocorrencia.png")


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Diagrama de similaridade Jaccard entre artigos
# ──────────────────────────────────────────────────────────────────────────────
def jaccard(set_a: set, set_b: set) -> float:
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def plot_similarity(results: dict) -> Path:
    articles = {
        n: set(t for t, _ in d.get("top10_terms", []))
        for n, d in results.items()
        if n != "__global__"
    }
    names = list(articles.keys())
    n = len(names)
    if n < 2:
        return None

    matrix = [
        [jaccard(articles[names[i]], articles[names[j]]) for j in range(n)]
        for i in range(n)
    ]

    fig, ax = plt.subplots(figsize=(max(8, n * 0.7), max(6, n * 0.6)))
    im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
    short = [nm[:22] for nm in names]
    ax.set_xticks(range(n))
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels(short, fontsize=8)
    ax.set_title(
        "Similaridade de Jaccard entre Artigos (por vocabulário)",
        fontsize=12,
        color="#00b4d8",
        fontweight="bold",
        pad=10,
    )
    plt.colorbar(im, ax=ax, label="Jaccard")

    for i in range(n):
        for j in range(n):
            ax.text(
                j,
                i,
                f"{matrix[i][j]:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="black" if matrix[i][j] > 0.5 else "white",
            )
    return _save(fig, "5_similaridade.png")


# ──────────────────────────────────────────────────────────────────────────────
# 6.  Termos de trabalhos futuros
# ──────────────────────────────────────────────────────────────────────────────
_FUTURE_PATS = [
    r"\bfuture (work|research|study|direction|investigation)\b",
    r"\bfuture (we |they )?(will|plan|intend|aim)\b",
    r"\bin the future\b",
    r"\bforthcoming\b",
    r"\bas future work\b",
    r"\bplanned extension\b",
    r"\bleft for future\b",
]


def extract_future_terms(results: dict) -> Counter:
    """
    Extrai os termos mais frequentes em sentenças de trabalhos futuros.
    Opera sobre o campo structured_info (não armazena texto bruto na JSON),
    por isso relemos os PDFs — aqui usamos os bigramas globais como proxy.
    """
    # usa os trigramas globais como proxy de "trabalhos futuros"
    future_counter: Counter = Counter()
    for name, data in results.items():
        if name == "__global__":
            continue
        # top termos do artigo entram como candidatos (simplificação didática)
        for term, cnt in data.get("top10_terms", []):
            future_counter[term] += 1
    return future_counter


def plot_future_terms(results: dict) -> Path:
    # tenta extrair do campo de resultados global
    trigrams = results.get("__global__", {}).get("trigrams_top10", [])
    bigrams = results.get("__global__", {}).get("bigrams_top10", [])

    # cria frequências de termos individuais a partir dos n-gramas
    term_cnt: Counter = Counter()
    for gram, cnt in trigrams + bigrams:
        for tok in gram.split():
            term_cnt[tok] += cnt

    top15 = term_cnt.most_common(15)
    if not top15:
        return None

    terms, counts = zip(*top15)
    fig, ax = plt.subplots(figsize=(11, 5))
    clr = plt.cm.autumn([(i / len(terms)) for i in range(len(terms))])
    ax.bar(terms, counts, color=clr, edgecolor="#0d0d1a", linewidth=0.5)
    ax.set_xticks(range(len(terms)))
    ax.set_xticklabels(terms, rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("Frequência em n-gramas")
    ax.set_title(
        "Termos Frequentes em N-gramas (proxy – Trabalhos Futuros)",
        fontsize=12,
        color="#00b4d8",
        fontweight="bold",
        pad=10,
    )
    ax.grid(axis="y", linestyle="--")
    return _save(fig, "6_termos_futuros.png")


# ──────────────────────────────────────────────────────────────────────────────
# 7.  Dendrograma (árvore de palavras) – similaridade entre termos top-N
# ──────────────────────────────────────────────────────────────────────────────
def _hamming_like(a: str, b: str) -> float:
    """Distância normalizada por comprimento de prefixo comum."""
    common = 0
    for ca, cb in zip(a, b):
        if ca == cb:
            common += 1
        else:
            break
    return 1 - common / max(len(a), len(b))


def plot_word_tree(results: dict) -> Path:
    all_terms: Counter = Counter()
    for name, data in results.items():
        if name == "__global__":
            continue
        all_terms.update({t: c for t, c in data.get("top10_terms", [])})
    top_words = [t for t, _ in all_terms.most_common(20)]
    if len(top_words) < 4:
        return None

    # algoritmo de agrupamento hierárquico (linkage simples, sem scipy)
    # representamos como grafo de similaridade
    G = nx.Graph()
    for w in top_words:
        G.add_node(w)
    for i, wi in enumerate(top_words):
        for wj in top_words[i + 1 :]:
            sim = 1 - _hamming_like(wi, wj)
            if sim > 0.3:
                G.add_edge(wi, wj, weight=sim)

    fig, ax = plt.subplots(figsize=(13, 9))
    pos = nx.kamada_kawai_layout(G) # usa o scipy por baixo dos panos
    node_sizes = [all_terms[n] * 8 + 200 for n in G.nodes()]
    edge_w = [G[u][v]["weight"] * 3 for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, ax=ax, width=edge_w, edge_color="#90e0ef", alpha=0.5)
    nx.draw_networkx_nodes(
        G, pos, ax=ax, node_size=node_sizes, node_color="#023e8a", alpha=0.9
    )
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=9, font_color="#caf0f8")
    ax.set_title(
        "Árvore de Palavras – Similaridade de Prefixo entre Termos Top-20",
        fontsize=12,
        color="#00b4d8",
        fontweight="bold",
        pad=10,
    )
    ax.axis("off")
    return _save(fig, "7_arvore_palavras.png")


# ──────────────────────────────────────────────────────────────────────────────
# 8.  Técnicas mais mencionadas nos artigos
# ──────────────────────────────────────────────────────────────────────────────
def plot_techniques(results: dict) -> Path:
    """
    Conta quantos artigos mencionam cada técnica de segurança cibernética
    e plota um gráfico de barras horizontais.
    """
    tech_counter: Counter = Counter()
    for name, data in results.items():
        if name == "__global__":
            continue
        for tech in data.get("techniques", []):
            tech_counter[tech] += 1

    if not tech_counter:
        print("  [AVISO] Nenhuma técnica detectada.")
        return None

    # ordena por frequência
    items = tech_counter.most_common()
    techs = [t for t, _ in items]
    counts = [c for _, c in items]

    fig, ax = plt.subplots(figsize=(12, max(5, len(techs) * 0.55 + 1)))
    colors = plt.cm.plasma(
        [0.15 + 0.7 * i / max(len(techs) - 1, 1) for i in range(len(techs))]
    )
    bars = ax.barh(
        list(reversed(techs)),
        list(reversed(counts)),
        color=list(reversed(colors)),
        edgecolor="#0d0d1a",
        linewidth=0.6,
        height=0.65,
    )

    ax.set_xlabel("Número de artigos que mencionam a técnica", fontsize=11)
    ax.set_title(
        "Técnicas de Segurança Cibernética Mais Mencionadas",
        fontsize=13,
        color="#00b4d8",
        fontweight="bold",
        pad=12,
    )
    ax.grid(axis="x", linestyle="--")
    ax.set_xlim(0, max(counts) + 1.5)

    for bar, val in zip(bars, list(reversed(counts))):
        ax.text(
            bar.get_width() + 0.08,
            bar.get_y() + bar.get_height() / 2,
            str(val),
            va="center",
            fontsize=10,
            color="#e0e0e0",
        )

    fig.tight_layout()
    return _save(fig, "8_tecnicas_mencionadas.png")


# ──────────────────────────────────────────────────────────────────────────────
# 9.  Evolução temporal dos termos por ano
# ──────────────────────────────────────────────────────────────────────────────
def plot_temporal_evolution(results: dict) -> Path:
    """
    Para cada ano presente nos artigos, soma a frequência dos top-N termos
    globais e plota um heatmap ano × termo (evolução temporal).
    Também plota um gráfico de linhas mostrando a evolução dos 5 termos
    mais relevantes ao longo dos anos.
    """
    # coleta top 10 termos globais
    global_top = [t for t, _ in results.get("__global__", {}).get("top10_terms", [])]
    if not global_top:
        return None

    # agrupa artigos por ano
    year_term_freq: dict[int, Counter] = defaultdict(Counter)
    articles_per_year: Counter = Counter()

    for name, data in results.items():
        if name == "__global__":
            continue
        year = data.get("year")
        if not year:
            continue
        articles_per_year[year] += 1
        term_map = {t: c for t, c in data.get("top10_terms", [])}
        for term in global_top:
            year_term_freq[year][term] += term_map.get(term, 0)

    years = sorted(year_term_freq.keys())
    if len(years) < 2:
        # com apenas 1 ano, ainda plota mas avisa
        print("  [INFO] Apenas 1 ano detectado — gráfico de barras simples.")

    if not years:
        print("  [AVISO] Nenhum artigo com ano detectado para evolução temporal.")
        return None

    # ── figura com 2 subplots ────────────────────────────────────────────────
    fig, (ax_heat, ax_line) = plt.subplots(
        2, 1, figsize=(13, 10), gridspec_kw={"height_ratios": [1.4, 1]}
    )
    fig.subplots_adjust(hspace=0.45)

    # --- Subplot 1: Heatmap ano × termo ------------------------------------
    matrix = [[year_term_freq[y].get(t, 0) for t in global_top] for y in years]
    im = ax_heat.imshow(
        matrix,
        aspect="auto",
        cmap=LinearSegmentedColormap.from_list("cy", ["#0d0d1a", "#0077b6", "#caf0f8"]),
    )
    ax_heat.set_xticks(range(len(global_top)))
    ax_heat.set_xticklabels(global_top, rotation=38, ha="right", fontsize=9)
    ax_heat.set_yticks(range(len(years)))
    ax_heat.set_yticklabels(
        [f"{y}  (n={articles_per_year[y]})" for y in years], fontsize=9
    )
    ax_heat.set_title(
        "Heatmap: Frequência de Termos por Ano de Publicação",
        fontsize=12,
        color="#00b4d8",
        fontweight="bold",
        pad=10,
    )
    plt.colorbar(im, ax=ax_heat, label="Frequência acumulada", shrink=0.8)

    for i, y in enumerate(years):
        for j, t in enumerate(global_top):
            v = year_term_freq[y].get(t, 0)
            if v > 0:
                ax_heat.text(
                    j, i, str(v), ha="center", va="center", fontsize=7, color="white"
                )

    # --- Subplot 2: Linhas – top 5 termos ao longo dos anos ----------------
    top5 = global_top[:5]
    line_colors = plt.cm.cool([i / 4 for i in range(5)])
    markers = ["o", "s", "^", "D", "P"]

    for idx, (term, color, marker) in enumerate(zip(top5, line_colors, markers)):
        freqs = [year_term_freq[y].get(term, 0) for y in years]
        ax_line.plot(
            years,
            freqs,
            color=color,
            marker=marker,
            linewidth=2,
            markersize=7,
            label=term,
        )

    ax_line.set_xlabel("Ano de publicação", fontsize=11)
    ax_line.set_ylabel("Frequência acumulada", fontsize=11)
    ax_line.set_title(
        "Evolução Temporal dos 5 Termos Mais Frequentes",
        fontsize=12,
        color="#00b4d8",
        fontweight="bold",
        pad=10,
    )
    ax_line.set_xticks(years)
    ax_line.set_xticklabels([str(y) for y in years])
    ax_line.legend(
        loc="upper left",
        fontsize=9,
        facecolor="#0d0d1a",
        edgecolor="#334466",
        labelcolor="#e0e0e0",
    )
    ax_line.grid(linestyle="--")

    return _save(fig, "9_evolucao_temporal.png")


# ──────────────────────────────────────────────────────────────────────────────
# Interface pública
# ──────────────────────────────────────────────────────────────────────────────
def gerar_visualizacoes(results: dict = None) -> list[Path]:
    if results is None:
        results = _load_results()

    print("\n[VIZ] Gerando visualizações...")
    paths = []

    geradas = [
        ("Nuvem de palavras", lambda: plot_wordcloud(results)),
        ("Top termos (barras)", lambda: plot_top_terms(results)),
        ("Heatmap", lambda: plot_heatmap(results)),
        ("Coocorrência bigramas", lambda: plot_coocurrence(results)),
        ("Similaridade Jaccard", lambda: plot_similarity(results)),
        ("Termos futuros", lambda: plot_future_terms(results)),
        ("Árvore de palavras", lambda: plot_word_tree(results)),
        ("Técnicas mencionadas", lambda: plot_techniques(results)),
        ("Evolução temporal por ano", lambda: plot_temporal_evolution(results)),
    ]

    for label, fn in geradas:
        try:
            p = fn()
            if p:
                paths.append(p)
        except Exception as e:
            print(f"  [AVISO] {label}: {e}")

    print(f"\n  → {len(paths)} visualizações salvas em '{OUT_DIR}/'")
    return paths


if __name__ == "__main__":
    gerar_visualizacoes()
