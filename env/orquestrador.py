#!/usr/bin/env python3
"""
================================================================================
ORQUESTRADOR - Executa todas as 4 abordagens e gera relatório consolidado
================================================================================
Executa cada abordagem em sequência sobre o mesmo conjunto de perguntas,
coleta os registros de auditoria e gera:
  - resultados/resultados_{abordagem}.csv       (uma por abordagem)
  - resultados/auditoria_{abordagem}.json       (auditoria detalhada)
  - resultados/comparativo_abordagens.csv       (consolidado)
  - resultados/resumo_estatistico.txt           (ranking e médias)

Uso:
  python orquestrador.py
  python orquestrador.py --perguntas perguntas_teste.txt
================================================================================
"""

import argparse

import config
import comum
import abordagem_tfidf
import abordagem_llm_puro
import abordagem_indice_toc
import abordagem_faiss
import abordagem_faiss_nvidia
import abordagem_toc_chunks


ABORDAGENS = {
    "tfidf": abordagem_tfidf.executar_abordagem,
    "llm_puro": abordagem_llm_puro.executar_abordagem,
    "indice_toc": abordagem_indice_toc.executar_abordagem,
    "faiss": abordagem_faiss.executar_abordagem,
    "faiss_nvidia": abordagem_faiss_nvidia.executar_abordagem,
    "toc_chunks": abordagem_toc_chunks.executar_abordagem,
}


def main():
    parser = argparse.ArgumentParser(description="Orquestrador de comparação RAG")
    parser.add_argument("--perguntas", help="Arquivo de perguntas (uma por linha)")
    parser.add_argument(
        "--abordagens",
        nargs="*",
        default=list(ABORDAGENS.keys()),
        help="Quais abordagens executar (padrão: todas)",
    )
    args = parser.parse_args()

    perguntas = comum.carregar_perguntas(args.perguntas if args.perguntas else None)
    if not perguntas:
        print("❌ Nenhuma pergunta encontrada.")
        print("   Crie o arquivo 'perguntas_teste.txt' com uma pergunta por linha,")
        print("   ou passe --perguntas <arquivo>.")
        return

    print("=" * 70)
    print(f"📋 {len(perguntas)} perguntas de teste carregadas")
    print(f"📦 Abordagens: {', '.join(args.abordagens)}")
    print("=" * 70)

    todos_registros = {}

    for nome in args.abordagens:
        if nome not in ABORDAGENS:
            print(f"⚠️  Abordagem desconhecida: {nome} (ignorada)")
            continue
        print(f"\n{'#' * 70}")
        print(f"# EXECUTANDO ABORDAGEM: {nome}")
        print(f"{'#' * 70}")
        try:
            registros = ABORDAGENS[nome](perguntas)
            todos_registros[nome] = registros
            comum.salvar_resultados_csv(nome, registros)
            comum.salvar_registros_json(nome, registros)
            print(f"  → {len(registros)} registros salvos.")
        except Exception as e:
            print(f"❌ Erro na abordagem {nome}: {e}")
            import traceback
            traceback.print_exc()

    if todos_registros:
        print(f"\n{'=' * 70}")
        print("📊 GERANDO RELATÓRIOS CONSOLIDADOS")
        print("=" * 70)
        csv_path = comum.gerar_relatorio_consolidado(todos_registros)
        resumo_path = comum.gerar_resumo_estatistico(todos_registros)
        print(f"  ✓ Comparativo: {csv_path}")
        print(f"  ✓ Resumo:      {resumo_path}")

        # Mostrar resumo no console
        print("\n" + resumo_path.read_text(encoding="utf-8"))

    print("\n✅ Concluído!")


if __name__ == "__main__":
    main()