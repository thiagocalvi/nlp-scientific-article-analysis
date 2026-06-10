"""
avaliacao.py
============
Etapa 4 – Avaliação de desempenho do sistema de extração de informações.

Métricas calculadas:
  • Precisão  (Precision)
  • Revocação (Recall)
  • F1-Score
  • Taxa de cobertura de referências
  • Cobertura de campos estruturados

Os gabaritos (ground truth) são definidos manualmente abaixo como exemplo
didático para os primeiros artigos. Para uma avaliação completa, preencha
todas as entradas de `GABARITO`.
"""

import json
from pathlib import Path

RESULTS_FILE = Path("saida/resultados_pipeline.json")
OUT_DIR      = Path("saida")


# ──────────────────────────────────────────────────────────────────────────────
# GABARITO (ground truth) – preencher manualmente por artigo
# Cada chave é o stem do nome do arquivo PDF (sem extensão).
# Cada campo indica SE o artigo tem (True) aquele elemento, para avaliação
# de cobertura dos campos estruturados.
# ──────────────────────────────────────────────────────────────────────────────
# Exemplo de gabarito para avaliação de presença/ausência de campos:
# (preencher True/False de acordo com leitura manual dos artigos)
GABARITO: dict[str, dict[str, bool]] = {
    # "nome_do_arquivo": {
    #     "tem_objetivo":     True,
    #     "tem_problema":     True,
    #     "tem_metodologia":  True,
    #     "tem_contribuicao": True,
    #     "tem_referencias":  True,
    # },
}

# Limiar mínimo de sentenças para considerar que o campo foi "encontrado"
MIN_SENTENCES = 1


# ──────────────────────────────────────────────────────────────────────────────
# Avaliação de cobertura de campos estruturados
# ──────────────────────────────────────────────────────────────────────────────
def avaliar_cobertura(results: dict) -> dict:
    """
    Para cada artigo, verifica se os campos obrigatórios foram extraídos.
    Retorna métricas de cobertura geral (sem gabarito) e métricas de
    precisão/recall quando o gabarito estiver disponível.
    """
    metricas_por_artigo = {}
    tp_total = fp_total = fn_total = 0

    for name, data in results.items():
        if name == "__global__":
            continue
        info = data.get("structured_info", {})

        extraido = {
            "tem_objetivo":     len(info.get("objectives", []))    >= MIN_SENTENCES,
            "tem_problema":     len(info.get("problems", []))       >= MIN_SENTENCES,
            "tem_metodologia":  len(info.get("methods", []))        >= MIN_SENTENCES,
            "tem_contribuicao": len(info.get("contributions", []))  >= MIN_SENTENCES,
            "tem_referencias":  len(data.get("references", []))     >= 1,
        }

        cobertura_pct = 100 * sum(extraido.values()) / len(extraido)

        m = {
            "extraido": extraido,
            "cobertura_pct": round(cobertura_pct, 1),
            "num_refs": len(data.get("references", [])),
            "top3_termos": [t for t, _ in data.get("top10_terms", [])[:3]],
        }

        # se temos gabarito para este artigo
        if name in GABARITO:
            gab = GABARITO[name]
            tp = fp = fn = 0
            for campo in extraido:
                pred = extraido[campo]
                gold = gab.get(campo, False)
                if pred and gold:
                    tp += 1
                elif pred and not gold:
                    fp += 1
                elif not pred and gold:
                    fn += 1

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1        = (2 * precision * recall / (precision + recall)
                         if (precision + recall) > 0 else 0.0)
            m["precision"] = round(precision, 3)
            m["recall"]    = round(recall, 3)
            m["f1"]        = round(f1, 3)
            tp_total += tp; fp_total += fp; fn_total += fn

        metricas_por_artigo[name] = m

    # métricas globais
    n = len(metricas_por_artigo)
    cobertura_media = (
        sum(v["cobertura_pct"] for v in metricas_por_artigo.values()) / n
        if n else 0
    )

    global_metrics: dict = {
        "num_artigos_avaliados": n,
        "cobertura_media_pct": round(cobertura_media, 1),
    }

    if GABARITO:
        prec_g = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0
        rec_g  = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0
        f1_g   = (2 * prec_g * rec_g / (prec_g + rec_g)
                  if (prec_g + rec_g) > 0 else 0)
        global_metrics["precision_global"] = round(prec_g, 3)
        global_metrics["recall_global"]    = round(rec_g, 3)
        global_metrics["f1_global"]        = round(f1_g, 3)

    return {"por_artigo": metricas_por_artigo, "global": global_metrics}


# ──────────────────────────────────────────────────────────────────────────────
# Relatório de avaliação
# ──────────────────────────────────────────────────────────────────────────────
def gerar_relatorio(metricas: dict) -> str:
    lines = []
    lines.append("=" * 60)
    lines.append("  RELATÓRIO DE AVALIAÇÃO DE DESEMPENHO")
    lines.append("  Tema: Segurança Cibernética")
    lines.append("=" * 60)

    g = metricas["global"]
    lines.append(f"\nArtigos avaliados : {g['num_artigos_avaliados']}")
    lines.append(f"Cobertura média   : {g['cobertura_media_pct']}%")
    if "f1_global" in g:
        lines.append(f"Precisão global   : {g['precision_global']}")
        lines.append(f"Recall global     : {g['recall_global']}")
        lines.append(f"F1-Score global   : {g['f1_global']}")

    lines.append("\n" + "-" * 60)
    lines.append(f"{'Artigo':<35} {'Cob%':>5} {'Obj':>4} {'Prob':>5} "
                 f"{'Met':>4} {'Cont':>5} {'Refs':>5}")
    lines.append("-" * 60)

    for name, m in metricas["por_artigo"].items():
        e = m["extraido"]
        short = name[:33]
        lines.append(
            f"{short:<35} {m['cobertura_pct']:>5.1f} "
            f"{'✓' if e['tem_objetivo'] else '✗':>4} "
            f"{'✓' if e['tem_problema'] else '✗':>5} "
            f"{'✓' if e['tem_metodologia'] else '✗':>4} "
            f"{'✓' if e['tem_contribuicao'] else '✗':>5} "
            f"{m['num_refs']:>5}"
        )
        if "f1" in m:
            lines.append(f"  → P={m['precision']} R={m['recall']} F1={m['f1']}")

    lines.append("=" * 60)
    lines.append("\nLEGENDA:")
    lines.append("  Obj  = Objetivo encontrado")
    lines.append("  Prob = Problema encontrado")
    lines.append("  Met  = Metodologia encontrada")
    lines.append("  Cont = Contribuição encontrada")
    lines.append("  Refs = Número de referências extraídas")

    lines.append("\nMETODOLOGIA DE AVALIAÇÃO:")
    lines.append("  • Cobertura de campos: verificação automática de presença")
    lines.append("    de pelo menos 1 sentença por campo estruturado.")
    lines.append("  • Precisão/Recall/F1: calculados quando gabarito manual")
    lines.append("    está disponível (variável GABARITO em avaliacao.py).")
    lines.append("  • Extração baseada em heurísticas de padrões linguísticos")
    lines.append("    sem uso de modelos de ML pré-treinados.")

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Interface pública
# ──────────────────────────────────────────────────────────────────────────────
def avaliar(results: dict) -> dict:
    print("\n[4] Avaliando desempenho...")
    metricas = avaliar_cobertura(results)

    relatorio = gerar_relatorio(metricas)
    print(relatorio)

    # salva JSON
    p_json = OUT_DIR / "avaliacao.json"
    with open(p_json, "w", encoding="utf-8") as f:
        json.dump(metricas, f, ensure_ascii=False, indent=2)

    # salva relatório texto
    p_txt = OUT_DIR / "avaliacao_relatorio.txt"
    p_txt.write_text(relatorio, encoding="utf-8")
    print(f"\n  [✓] Avaliação salva em: {p_json}")
    print(f"  [✓] Relatório salvo em: {p_txt}")

    return metricas


if __name__ == "__main__":
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            results = json.load(f)
        avaliar(results)
    else:
        print("Execute pipeline.py primeiro.")
