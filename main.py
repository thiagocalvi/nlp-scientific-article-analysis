"""
main.py
=======
Ponto de entrada principal.
Executa as 4 etapas em sequência:
  1. Pipeline (leitura, pré-proc, bag-of-words, extração)
  2. Ontologia (Turtle + JSON-LD + OWL)
  3. Avaliação de desempenho
  4. Visualizações

Uso:
    python main.py                   # usa pasta 'artigos/'
    python main.py --dir /caminho/   # pasta personalizada
"""

import sys
import argparse
from pathlib import Path

from pipeline       import run_pipeline
from ontologia      import exportar_ontologia
from avaliacao      import avaliar
from visualizacoes  import gerar_visualizacoes


def main(pdf_dir: Path) -> None:
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║  Sistema de Análise de Artigos Científicos – PLN/IIA     ║")
    print("║  Universidade Estadual de Maringá – Depto. Informática   ║")
    print("║  Tema: Segurança Cibernética                             ║")
    print("╚" + "═" * 58 + "╝\n")

    # ── Etapa 1 + 2 ──────────────────────────────────────────────
    results = run_pipeline(pdf_dir)
    if not results:
        print("\n[ERRO] Nenhum resultado. Verifique a pasta de artigos.\n")
        sys.exit(1)

    # ── Etapa 3 ──────────────────────────────────────────────────
    exportar_ontologia(results)

    # ── Etapa 4 ──────────────────────────────────────────────────
    avaliar(results)

    # ── Visualizações ─────────────────────────────────────────────
    gerar_visualizacoes(results)

    print("\n" + "═" * 60)
    print("  Execução concluída! Arquivos em 'saida/' e 'ontologia/'")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analisador de artigos científicos – Segurança Cibernética"
    )
    parser.add_argument(
        "--dir", type=Path, default=Path("artigos"),
        help="Diretório com os PDFs (padrão: artigos/)"
    )
    args = parser.parse_args()
    main(args.dir)
