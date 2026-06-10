"""
ontologia.py
============
Etapa 3 – Serialização dos dados extraídos em formatos de ontologia.
Formatos suportados: Turtle (.ttl), JSON-LD (.jsonld), OWL/XML (.owl)
"""

import json
import re
from pathlib import Path
from datetime import date

OUT_DIR  = Path("saida")
ONTO_DIR = Path("ontologia")
ONTO_DIR.mkdir(exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _safe_id(text: str) -> str:
    """Gera um identificador URI-seguro a partir de uma string."""
    s = re.sub(r'[^a-zA-Z0-9_]', '_', text)
    s = re.sub(r'_+', '_', s)
    return s[:60].strip('_')


def _esc(text: str) -> str:
    """Escapa aspas duplas para uso em strings RDF."""
    return text.replace('\\', '\\\\').replace('"', '\\"').replace('\n', ' ')


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Turtle (.ttl)
# ──────────────────────────────────────────────────────────────────────────────
PREFIXES_TTL = """\
@prefix rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix owl:  <http://www.w3.org/2002/07/owl#> .
@prefix xsd:  <http://www.w3.org/2001/XMLSchema#> .
@prefix sa:   <http://uem.br/ontologia/artigo#> .
@prefix dc:   <http://purl.org/dc/elements/1.1/> .

"""

CLASS_DEFS_TTL = """\
# ── Classes ──────────────────────────────────────────────────────────────────
sa:ArtigoCientifico rdf:type owl:Class ;
    rdfs:label "Artigo Científico" ;
    rdfs:comment "Representa um artigo científico analisado." .

sa:Objetivo rdf:type owl:Class ;
    rdfs:label "Objetivo" ;
    rdfs:comment "O objetivo declarado no artigo." .

sa:Problema rdf:type owl:Class ;
    rdfs:label "Problema" ;
    rdfs:comment "O problema ou lacuna identificado no artigo." .

sa:Metodologia rdf:type owl:Class ;
    rdfs:label "Metodologia" ;
    rdfs:comment "Método ou metodologia descrita no artigo." .

sa:Contribuicao rdf:type owl:Class ;
    rdfs:label "Contribuição" ;
    rdfs:comment "Contribuição do artigo para a literatura." .

sa:Referencia rdf:type owl:Class ;
    rdfs:label "Referência Bibliográfica" .

sa:Termo rdf:type owl:Class ;
    rdfs:label "Termo Relevante" ;
    rdfs:comment "Termo frequente identificado por análise de PLN." .

# ── Propriedades de objeto ────────────────────────────────────────────────────
sa:temObjetivo rdf:type owl:ObjectProperty ;
    rdfs:domain sa:ArtigoCientifico ;
    rdfs:range  sa:Objetivo ;
    rdfs:label  "tem objetivo" .

sa:temProblema rdf:type owl:ObjectProperty ;
    rdfs:domain sa:ArtigoCientifico ;
    rdfs:range  sa:Problema ;
    rdfs:label  "tem problema" .

sa:temMetodologia rdf:type owl:ObjectProperty ;
    rdfs:domain sa:ArtigoCientifico ;
    rdfs:range  sa:Metodologia ;
    rdfs:label  "tem metodologia" .

sa:temContribuicao rdf:type owl:ObjectProperty ;
    rdfs:domain sa:ArtigoCientifico ;
    rdfs:range  sa:Contribuicao ;
    rdfs:label  "tem contribuição" .

sa:cita rdf:type owl:ObjectProperty ;
    rdfs:domain sa:ArtigoCientifico ;
    rdfs:range  sa:Referencia ;
    rdfs:label  "cita" .

sa:possuiTermo rdf:type owl:ObjectProperty ;
    rdfs:domain sa:ArtigoCientifico ;
    rdfs:range  sa:Termo ;
    rdfs:label  "possui termo" .

# ── Propriedades de dado ──────────────────────────────────────────────────────
sa:titulo      rdf:type owl:DatatypeProperty ; rdfs:domain sa:ArtigoCientifico ; rdfs:range xsd:string .
sa:texto       rdf:type owl:DatatypeProperty ; rdfs:range  xsd:string .
sa:frequencia  rdf:type owl:DatatypeProperty ; rdfs:domain sa:Termo          ; rdfs:range  xsd:integer .
sa:textoRef    rdf:type owl:DatatypeProperty ; rdfs:domain sa:Referencia     ; rdfs:range  xsd:string .

"""


def export_turtle(results: dict, path: Path) -> None:
    lines = [PREFIXES_TTL, CLASS_DEFS_TTL,
             f"# Gerado em: {date.today().isoformat()}\n\n"]

    global_top = {t: c for t, c in results.get("__global__", {}).get("top10_terms", [])}

    for name, data in results.items():
        if name == "__global__":
            continue
        aid = _safe_id(name)
        lines.append(f"sa:{aid} rdf:type sa:ArtigoCientifico ;\n")
        lines.append(f'    sa:titulo "{_esc(name)}"^^xsd:string ;\n')

        info = data.get("structured_info", {})

        # objetivos
        for i, obj in enumerate(info.get("objectives", []), 1):
            oid = f"{aid}_Obj{i}"
            lines.append(f"    sa:temObjetivo sa:{oid} ;\n")

        # problemas
        for i, prob in enumerate(info.get("problems", []), 1):
            pid = f"{aid}_Prob{i}"
            lines.append(f"    sa:temProblema sa:{pid} ;\n")

        # métodos
        for i, meth in enumerate(info.get("methods", []), 1):
            mid = f"{aid}_Met{i}"
            lines.append(f"    sa:temMetodologia sa:{mid} ;\n")

        # contribuições
        for i, cont in enumerate(info.get("contributions", []), 1):
            cid = f"{aid}_Cont{i}"
            lines.append(f"    sa:temContribuicao sa:{cid} ;\n")

        # top3 termos
        for term, freq in data.get("top10_terms", [])[:3]:
            tid = _safe_id(term)
            lines.append(f"    sa:possuiTermo sa:Termo_{tid} ;\n")

        # referências (primeiras 3)
        for i, ref in enumerate(data.get("references", [])[:3], 1):
            rid = f"{aid}_Ref{i}"
            lines.append(f"    sa:cita sa:{rid} ;\n")

        # fecha o bloco do artigo
        lines[-1] = lines[-1].rstrip(" ;\n") + " .\n\n"

        # instâncias de objetivo
        for i, obj in enumerate(info.get("objectives", []), 1):
            oid = f"{aid}_Obj{i}"
            lines.append(f'sa:{oid} rdf:type sa:Objetivo ;\n')
            lines.append(f'    sa:texto "{_esc(obj[:300])}"^^xsd:string .\n\n')

        # instâncias de problema
        for i, prob in enumerate(info.get("problems", []), 1):
            pid = f"{aid}_Prob{i}"
            lines.append(f'sa:{pid} rdf:type sa:Problema ;\n')
            lines.append(f'    sa:texto "{_esc(prob[:300])}"^^xsd:string .\n\n')

        # instâncias de método
        for i, meth in enumerate(info.get("methods", []), 1):
            mid = f"{aid}_Met{i}"
            lines.append(f'sa:{mid} rdf:type sa:Metodologia ;\n')
            lines.append(f'    sa:texto "{_esc(meth[:300])}"^^xsd:string .\n\n')

        # instâncias de contribuição
        for i, cont in enumerate(info.get("contributions", []), 1):
            cid = f"{aid}_Cont{i}"
            lines.append(f'sa:{cid} rdf:type sa:Contribuicao ;\n')
            lines.append(f'    sa:texto "{_esc(cont[:300])}"^^xsd:string .\n\n')

        # instâncias de referência
        for i, ref in enumerate(data.get("references", [])[:3], 1):
            rid = f"{aid}_Ref{i}"
            lines.append(f'sa:{rid} rdf:type sa:Referencia ;\n')
            lines.append(f'    sa:textoRef "{_esc(ref[:200])}"^^xsd:string .\n\n')

    # termos globais
    for term, freq in results.get("__global__", {}).get("top10_terms", []):
        tid = _safe_id(term)
        lines.append(f'sa:Termo_{tid} rdf:type sa:Termo ;\n')
        lines.append(f'    rdfs:label "{_esc(term)}"^^xsd:string ;\n')
        lines.append(f'    sa:frequencia {freq}^^xsd:integer .\n\n')

    path.write_text("".join(lines), encoding="utf-8")
    print(f"  [✓] Turtle salvo em: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# 2.  JSON-LD
# ──────────────────────────────────────────────────────────────────────────────
CONTEXT_JSONLD = {
    "@vocab":         "http://uem.br/ontologia/artigo#",
    "rdfs":           "http://www.w3.org/2000/01/rdf-schema#",
    "xsd":            "http://www.w3.org/2001/XMLSchema#",
    "titulo":         "titulo",
    "objetivos":      {"@id": "temObjetivo",   "@type": "@id"},
    "problemas":      {"@id": "temProblema",   "@type": "@id"},
    "metodologias":   {"@id": "temMetodologia","@type": "@id"},
    "contribuicoes":  {"@id": "temContribuicao","@type":"@id"},
    "referencias":    {"@id": "cita",          "@type": "@id"},
    "termos":         {"@id": "possuiTermo",   "@type": "@id"},
    "texto":          "texto",
    "frequencia":     {"@id": "frequencia", "@type": "xsd:integer"},
    "label":          "rdfs:label",
}


def export_jsonld(results: dict, path: Path) -> None:
    graph = []

    for name, data in results.items():
        if name == "__global__":
            continue
        aid = _safe_id(name)
        info = data.get("structured_info", {})

        node: dict = {
            "@id":   f"http://uem.br/artigo/{aid}",
            "@type": "ArtigoCientifico",
            "titulo": name,
        }

        # sub-nós inline
        if info.get("objectives"):
            node["objetivos"] = [
                {"@id": f"http://uem.br/artigo/{aid}_Obj{i}",
                 "@type": "Objetivo", "texto": o[:300]}
                for i, o in enumerate(info["objectives"], 1)
            ]
        if info.get("problems"):
            node["problemas"] = [
                {"@id": f"http://uem.br/artigo/{aid}_Prob{i}",
                 "@type": "Problema", "texto": p[:300]}
                for i, p in enumerate(info["problems"], 1)
            ]
        if info.get("methods"):
            node["metodologias"] = [
                {"@id": f"http://uem.br/artigo/{aid}_Met{i}",
                 "@type": "Metodologia", "texto": m[:300]}
                for i, m in enumerate(info["methods"], 1)
            ]
        if info.get("contributions"):
            node["contribuicoes"] = [
                {"@id": f"http://uem.br/artigo/{aid}_Cont{i}",
                 "@type": "Contribuicao", "texto": c[:300]}
                for i, c in enumerate(info["contributions"], 1)
            ]
        if data.get("references"):
            node["referencias"] = [
                {"@id": f"http://uem.br/artigo/{aid}_Ref{i}",
                 "@type": "Referencia", "texto": r[:200]}
                for i, r in enumerate(data["references"][:5], 1)
            ]
        top3 = data.get("top10_terms", [])[:3]
        if top3:
            node["termos"] = [
                {"@id": f"http://uem.br/termo/{_safe_id(t)}",
                 "@type": "Termo", "label": t, "frequencia": c}
                for t, c in top3
            ]
        graph.append(node)

    # termos globais
    for term, freq in results.get("__global__", {}).get("top10_terms", []):
        graph.append({
            "@id":       f"http://uem.br/termo/{_safe_id(term)}",
            "@type":     "Termo",
            "label":     term,
            "frequencia": freq,
        })

    doc = {
        "@context": CONTEXT_JSONLD,
        "@graph":   graph,
        "gerado_em": date.today().isoformat(),
        "tema": "Segurança Cibernética",
    }
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  [✓] JSON-LD salvo em: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# 3.  OWL/XML (subset)
# ──────────────────────────────────────────────────────────────────────────────
OWL_HEADER = """\
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE rdf:RDF [
    <!ENTITY owl  "http://www.w3.org/2002/07/owl#" >
    <!ENTITY rdf  "http://www.w3.org/1999/02/22-rdf-syntax-ns#" >
    <!ENTITY rdfs "http://www.w3.org/2000/01/rdf-schema#" >
    <!ENTITY xsd  "http://www.w3.org/2001/XMLSchema#" >
    <!ENTITY sa   "http://uem.br/ontologia/artigo#" >
]>
<rdf:RDF
    xmlns:owl ="http://www.w3.org/2002/07/owl#"
    xmlns:rdf ="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
    xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"
    xmlns:xsd ="http://www.w3.org/2001/XMLSchema#"
    xmlns:sa  ="http://uem.br/ontologia/artigo#">

  <owl:Ontology rdf:about="http://uem.br/ontologia/artigo">
    <rdfs:label>Ontologia de Artigo Científico – Segurança Cibernética</rdfs:label>
  </owl:Ontology>

  <!-- Classes -->
  <owl:Class rdf:about="&sa;ArtigoCientifico"><rdfs:label>Artigo Científico</rdfs:label></owl:Class>
  <owl:Class rdf:about="&sa;Objetivo">       <rdfs:label>Objetivo</rdfs:label></owl:Class>
  <owl:Class rdf:about="&sa;Problema">       <rdfs:label>Problema</rdfs:label></owl:Class>
  <owl:Class rdf:about="&sa;Metodologia">    <rdfs:label>Metodologia</rdfs:label></owl:Class>
  <owl:Class rdf:about="&sa;Contribuicao">   <rdfs:label>Contribuição</rdfs:label></owl:Class>
  <owl:Class rdf:about="&sa;Referencia">     <rdfs:label>Referência</rdfs:label></owl:Class>
  <owl:Class rdf:about="&sa;Termo">          <rdfs:label>Termo Relevante</rdfs:label></owl:Class>

"""

OWL_FOOTER = "</rdf:RDF>\n"


def _xml_esc(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;")
             .replace(">", "&gt;").replace('"', "&quot;")
             .replace("'", "&apos;"))


def export_owl(results: dict, path: Path) -> None:
    lines = [OWL_HEADER]

    for name, data in results.items():
        if name == "__global__":
            continue
        aid = _safe_id(name)
        info = data.get("structured_info", {})
        base = f"http://uem.br/artigo/{aid}"

        lines.append(f'  <!-- Artigo: {_xml_esc(name)} -->\n')
        lines.append(f'  <sa:ArtigoCientifico rdf:about="{base}">\n')
        lines.append(f'    <sa:titulo>{_xml_esc(name)}</sa:titulo>\n')

        for i, obj in enumerate(info.get("objectives", []), 1):
            oid = f"{base}_Obj{i}"
            lines.append(f'    <sa:temObjetivo rdf:resource="{oid}"/>\n')
        for i, prob in enumerate(info.get("problems", []), 1):
            pid = f"{base}_Prob{i}"
            lines.append(f'    <sa:temProblema rdf:resource="{pid}"/>\n')
        for i, meth in enumerate(info.get("methods", []), 1):
            mid = f"{base}_Met{i}"
            lines.append(f'    <sa:temMetodologia rdf:resource="{mid}"/>\n')
        for i, cont in enumerate(info.get("contributions", []), 1):
            cid = f"{base}_Cont{i}"
            lines.append(f'    <sa:temContribuicao rdf:resource="{cid}"/>\n')
        lines.append(f'  </sa:ArtigoCientifico>\n\n')

        for i, obj in enumerate(info.get("objectives", []), 1):
            lines.append(f'  <sa:Objetivo rdf:about="{base}_Obj{i}">\n')
            lines.append(f'    <sa:texto>{_xml_esc(obj[:300])}</sa:texto>\n')
            lines.append(f'  </sa:Objetivo>\n\n')

        for i, prob in enumerate(info.get("problems", []), 1):
            lines.append(f'  <sa:Problema rdf:about="{base}_Prob{i}">\n')
            lines.append(f'    <sa:texto>{_xml_esc(prob[:300])}</sa:texto>\n')
            lines.append(f'  </sa:Problema>\n\n')

        for i, meth in enumerate(info.get("methods", []), 1):
            lines.append(f'  <sa:Metodologia rdf:about="{base}_Met{i}">\n')
            lines.append(f'    <sa:texto>{_xml_esc(meth[:300])}</sa:texto>\n')
            lines.append(f'  </sa:Metodologia>\n\n')

        for i, cont in enumerate(info.get("contributions", []), 1):
            lines.append(f'  <sa:Contribuicao rdf:about="{base}_Cont{i}">\n')
            lines.append(f'    <sa:texto>{_xml_esc(cont[:300])}</sa:texto>\n')
            lines.append(f'  </sa:Contribuicao>\n\n')

    # termos globais
    for term, freq in results.get("__global__", {}).get("top10_terms", []):
        tid = _safe_id(term)
        lines.append(f'  <sa:Termo rdf:about="http://uem.br/termo/{tid}">\n')
        lines.append(f'    <rdfs:label>{_xml_esc(term)}</rdfs:label>\n')
        lines.append(f'    <sa:frequencia rdf:datatype="&xsd;integer">{freq}</sa:frequencia>\n')
        lines.append(f'  </sa:Termo>\n\n')

    lines.append(OWL_FOOTER)
    path.write_text("".join(lines), encoding="utf-8")
    print(f"  [✓] OWL/XML salvo em: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Interface pública
# ──────────────────────────────────────────────────────────────────────────────
def exportar_ontologia(results: dict) -> None:
    print("\n[3] Exportando ontologias...")
    export_turtle(results, ONTO_DIR / "artigos_cybersec.ttl")
    export_jsonld(results, ONTO_DIR / "artigos_cybersec.jsonld")
    export_owl   (results, ONTO_DIR / "artigos_cybersec.owl")


if __name__ == "__main__":
    import json
    p = Path("saida/resultados_pipeline.json")
    if p.exists():
        with open(p) as f:
            results = json.load(f)
        exportar_ontologia(results)
    else:
        print("Execute pipeline.py primeiro.")
