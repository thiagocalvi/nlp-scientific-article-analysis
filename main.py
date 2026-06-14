"""
main.py
=======
Ponto de entrada principal.
Executa as 4 etapas em sequência ou inicia apenas o servidor web.

Uso:
    python main.py                   # Roda o pipeline inteiro usando a pasta 'artigos/'
    python main.py --dir /caminho/   # Roda o pipeline com pasta personalizada
    python main.py --server          # Inicia APENAS o servidor web usando dados já processados
"""

import argparse
import shutil
import socket
import subprocess
import sys
from pathlib import Path

from avaliacao import avaliar
from ontologia import exportar_ontologia
from pipeline import run_pipeline
from visualizacoes import gerar_visualizacoes


def get_porta_livre(porta_inicial: int = 8000) -> int:
    """Procura a próxima porta disponível caso a inicial esteja em uso."""
    porta = porta_inicial
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("localhost", porta)) != 0:
                return porta
        porta += 1


def iniciar_ambiente_web():
    """Copia os arquivos necessários e inicia o servidor HTTP local."""
    print("\n" + "=" * 72)
    print("  INICIANDO AMBIENTE WEB")
    print("=" * 72)

    web_dir = Path("web")
    web_dir.mkdir(exist_ok=True)

    print("[*] Sincronizando arquivos de dados para o diretório 'web/'...")
    arquivos_para_copiar = [
        "ontologia/artigos_cybersec.jsonld",
        "saida/resultados_pipeline.json",
        "saida/avaliacao.json",
    ]

    arquivos_faltando = False

    for arquivo in arquivos_para_copiar:
        caminho_origem = Path(arquivo)
        if caminho_origem.exists():
            shutil.copy2(caminho_origem, web_dir)
            print(f"    - {caminho_origem.name} (Sincronizado)")
        else:
            print(f"    - [ALERTA] {caminho_origem.name} não encontrado na origem.")
            arquivos_faltando = True

    if arquivos_faltando:
        print(
            "\n[!] Atenção: Alguns arquivos não foram encontrados. O dashboard pode não carregar corretamente."
        )
        print(
            "[!] Recomenda-se rodar o pipeline completo ('python main.py') se for a primeira execução.\n"
        )

    porta = get_porta_livre(8000)

    print("\n[*] Servidor HTTP nativo ativado.")
    print(f"[*] Acesse no seu navegador: http://localhost:{porta}")
    print("[*] Logs do servidor em tempo real (Pressione CTRL+C para encerrar):\n")
    print("-" * 72)

    try:
        subprocess.run([sys.executable, "-m", "http.server", str(porta)], cwd="web")
    except KeyboardInterrupt:
        print("\n\n" + "=" * 72)
        print("  Servidor encerrado pelo usuário. Sistema finalizado.")
        print("=" * 72 + "\n")


def main(pdf_dir: Path, apenas_servidor: bool) -> None:
    if apenas_servidor:
        iniciar_ambiente_web()
        return

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

    print("Deseja iniciar o servidor web para visualização? [S/N]")
    iniciar_servidor = input("> ").strip().lower()

    if iniciar_servidor in ["s", "sim", "y", "yes"]:
        iniciar_ambiente_web()
    else:
        print("Programa finalizado.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analisador de artigos científicos – Segurança Cibernética"
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=Path("artigos"),
        help="Diretório com os PDFs (padrão: artigos/)",
    )
    # Nova flag adicionada aqui:
    parser.add_argument(
        "--server",
        action="store_true",
        help="Pula o pipeline de NLP e inicia apenas o servidor web com os resultados existentes.",
    )

    args = parser.parse_args()
    main(args.dir, args.server)
