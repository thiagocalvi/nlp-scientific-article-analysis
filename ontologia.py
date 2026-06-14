"""
ontologia.py
============
Etapa 3 – Serialização dos dados extraídos em formatos de ontologia.
Formato suportado: JSON-LD (.jsonld),
"""

import json
import re
from datetime import date
from pathlib import Path

OUT_DIR = Path("saida")
ONTO_DIR = Path("ontologia")
ONTO_DIR.mkdir(exist_ok=True)


def _safe_id(text: str) -> str:
    """Gera um identificador URI-seguro a partir de uma string."""
    s = re.sub(r"[^a-zA-Z0-9_]", "_", text)
    s = re.sub(r"_+", "_", s)
    return s[:60].strip("_")


def _esc(text: str) -> str:
    """Escapa aspas duplas para uso em strings RDF."""
    return text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")


CONTEXT_JSONLD = {
    "@vocab": "http://uem.br/ontologia/artigo#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
    "titulo": "titulo",
    "objetivos": {"@id": "temObjetivo", "@type": "@id"},
    "problemas": {"@id": "temProblema", "@type": "@id"},
    "metodologias": {"@id": "temMetodologia", "@type": "@id"},
    "contribuicoes": {"@id": "temContribuicao", "@type": "@id"},
    "referencias": {"@id": "cita", "@type": "@id"},
    "termos": {"@id": "possuiTermo", "@type": "@id"},
    "texto": "texto",
    "frequencia": {"@id": "frequencia", "@type": "xsd:integer"},
    "label": "rdfs:label",
}


def export_jsonld(results: dict, path: Path) -> None:
    graph = []

    for name, data in results.items():
        if name == "__global__":
            continue
        aid = _safe_id(name)
        info = data.get("structured_info", {})

        node: dict = {
            "@id": f"http://uem.br/artigo/{aid}",
            "@type": "ArtigoCientifico",
            "titulo": name,
        }

        # sub-nós inline
        if info.get("objectives"):
            node["objetivos"] = [
                {
                    "@id": f"http://uem.br/artigo/{aid}_Obj{i}",
                    "@type": "Objetivo",
                    "texto": o[:300],
                }
                for i, o in enumerate(info["objectives"], 1)
            ]
        if info.get("problems"):
            node["problemas"] = [
                {
                    "@id": f"http://uem.br/artigo/{aid}_Prob{i}",
                    "@type": "Problema",
                    "texto": p[:300],
                }
                for i, p in enumerate(info["problems"], 1)
            ]
        if info.get("methods"):
            node["metodologias"] = [
                {
                    "@id": f"http://uem.br/artigo/{aid}_Met{i}",
                    "@type": "Metodologia",
                    "texto": m[:300],
                }
                for i, m in enumerate(info["methods"], 1)
            ]
        if info.get("contributions"):
            node["contribuicoes"] = [
                {
                    "@id": f"http://uem.br/artigo/{aid}_Cont{i}",
                    "@type": "Contribuicao",
                    "texto": c[:300],
                }
                for i, c in enumerate(info["contributions"], 1)
            ]
        if data.get("references"):
            node["referencias"] = [
                {
                    "@id": f"http://uem.br/artigo/{aid}_Ref{i}",
                    "@type": "Referencia",
                    "texto": r[:200],
                }
                for i, r in enumerate(data["references"][:5], 1)
            ]
        top3 = data.get("top10_terms", [])[:3]
        if top3:
            node["termos"] = [
                {
                    "@id": f"http://uem.br/termo/{_safe_id(t)}",
                    "@type": "Termo",
                    "label": t,
                    "frequencia": c,
                }
                for t, c in top3
            ]
        graph.append(node)

    # termos globais
    for term, freq in results.get("__global__", {}).get("top10_terms", []):
        graph.append(
            {
                "@id": f"http://uem.br/termo/{_safe_id(term)}",
                "@type": "Termo",
                "label": term,
                "frequencia": freq,
            }
        )

    doc = {
        "@context": CONTEXT_JSONLD,
        "@graph": graph,
        "gerado_em": date.today().isoformat(),
        "tema": "Segurança Cibernética",
    }
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  [✓] JSON-LD salvo em: {path}")


def exportar_ontologia(results: dict) -> None:
    print("\n[3] Exportando ontologia...")
    export_jsonld(results, ONTO_DIR / "artigos_cybersec.jsonld")


if __name__ == "__main__":
    import json

    p = Path("saida/resultados_pipeline.json")
    if p.exists():
        with open(p) as f:
            results = json.load(f)
        exportar_ontologia(results)
    else:
        print("Execute pipeline.py primeiro.")
