#!/usr/bin/env python3
"""
================================================================================
PIPELINE COMPLETO - Executa abordagens RAG + Avaliação RAGAS
================================================================================
1. Carrega perguntas do ground_truth_amm_400.json
2. Executa as abordagens RAG sobre essas perguntas
3. Avalia os resultados com métricas RAGAS

Uso:
  python pipeline_ragas.py --abordagens faiss_nvidia --amostra 5
  python pipeline_ragas.py --abordagens toc_chunks faiss --amostra 10
================================================================================
"""

import argparse
import json
import time
from pathlib import Path

import config
import comum
import abordagem_tfidf
import abordagem_llm_puro
import abordagem_indice_toc
import abordagem_faiss
import abordagem_faiss_nvidia
import abordagem_toc_chunks
from ragas_evaluator import (
    RagasEvaluator,
    carregar_ground_truth_amm400,
)


ABORDAGENS = {
    "tfidf": abordagem_tfidf.executar_abordagem,
    "llm_puro": abordagem_llm_puro.executar_abordagem,
    "indice_toc": abordagem_indice_toc.executar_abordagem,
    "faiss": abordagem_faiss.executar_abordagem,
    "faiss_nvidia": abordagem_faiss_nvidia.executar_abordagem,
    "toc_chunks": abordagem_toc_chunks.executar_abordagem,
}


def carregar_perguntas_gt(amostra: int = None) -> list:
    """Carrega perguntas do ground_truth_amm_400.json."""
    gt_path = config.BASE_DIR / "ground_truth_amm_400.json"
    with open(gt_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    perguntas = [item["question"] for item in data]
    if amostra:
        perguntas = perguntas[:amostra]
    return perguntas


def main():
    parser = argparse.ArgumentParser(description="Pipeline RAG + RAGAS")
    parser.add_argument(
        "--abordagens",
        nargs="*",
        default=["toc_chunks"],
        help="Abordagens a executar (padrão: toc_chunks)",
    )
    parser.add_argument(
        "--amostra",
        type=int,
        default=5,
        help="Número de perguntas do GT (padrão: 5)",
    )
    parser.add_argument(
        "--skip-rag",
        action="store_true",
        help="Pular execução RAG (usar auditorias existentes)",
    )
    parser.add_argument(
        "--skip-ragas",
        action="store_true",
        help="Pular avaliação RAGAS",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help=f"Modelo LLM juiz (padrão: {config.LLM_MODEL})",
    )
    args = parser.parse_args()

    # ================================================================
    # FASE 1: Executar abordagens RAG
    # ================================================================
    if not args.skip_rag:
        perguntas = carregar_perguntas_gt(args.amostra)
        print(f"📋 {len(perguntas)} perguntas do ground truth carregadas")
        print(f"📦 Abordagens: {', '.join(args.abordagens)}")

        for nome in args.abordagens:
            if nome not in ABORDAGENS:
                print(f"⚠️  Abordagem desconhecida: {nome}")
                continue

            print(f"\n{'#' * 70}")
            print(f"# FASE 1 - RAG: {nome}")
            print(f"{'#' * 70}")
            try:
                registros = ABORDAGENS[nome](perguntas)
                comum.salvar_resultados_csv(nome, registros)
                comum.salvar_registros_json(nome, registros)
                print(f"  ✅ {len(registros)} registros salvos")
            except Exception as e:
                print(f"  ❌ Erro: {e}")
                import traceback
                traceback.print_exc()

    # ================================================================
    # FASE 2: Avaliação RAGAS
    # ================================================================
    if not args.skip_ragas:
        print(f"\n{'=' * 70}")
        print("FASE 2 - AVALIAÇÃO RAGAS")
        print(f"{'=' * 70}")

        gt_map = carregar_ground_truth_amm400()
        evaluator = RagasEvaluator(judge_model=args.judge_model)

        all_results = {}
        for nome in args.abordagens:
            print(f"\n{'#' * 70}")
            print(f"# RAGAS: {nome}")
            print(f"{'#' * 70}")

            # Carregar auditoria
            audit_path = config.RESULTADOS_DIR / f"auditoria_{nome}.json"
            if not audit_path.exists():
                print(f"  ⚠️  Auditoria não encontrada: {audit_path}")
                continue

            with open(audit_path, "r", encoding="utf-8") as f:
                records = json.load(f)

            # Filtrar perguntas com GT
            records_com_gt = [r for r in records if r["pergunta"] in gt_map]
            print(f"  📊 {len(records)} registros, {len(records_com_gt)} com GT")

            if not records_com_gt:
                continue

            results = evaluator.evaluate_batch(records_com_gt, gt_map)
            all_results[nome] = results

            # Salvar resultados RAGAS
            ragas_path = config.RESULTADOS_DIR / f"ragas_{nome}.json"
            with open(ragas_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"\n  ✅ RAGAS salvo em {ragas_path}")

            # Resumo
            summary = evaluator.summary(results)
            print(f"\n  📊 Resumo {nome}:")
            for k, v in summary.items():
                print(f"     {k}: {v}")

        # Consolidado
        if all_results:
            print(f"\n{'=' * 70}")
            print("📊 RELATÓRIO CONSOLIDADO RAGAS")
            print(f"{'=' * 70}")

            consolidado = {}
            for nome, results in all_results.items():
                consolidado[nome] = evaluator.summary(results)

            cons_path = config.RESULTADOS_DIR / "ragas_consolidado.json"
            with open(cons_path, "w", encoding="utf-8") as f:
                json.dump(consolidado, f, ensure_ascii=False, indent=2)

            # Tabela
            metric_names = [
                "avg_faithfulness", "avg_answer_relevancy", "avg_context_precision",
                "avg_context_recall", "avg_answer_correctness", "avg_answer_similarity",
                "avg_ragas_score",
            ]
            header = f"{'Abordagem':<20}" + "".join(
                f"{m.replace('avg_', ''):>20}" for m in metric_names
            )
            print(f"\n{header}")
            print("-" * len(header))
            for nome, s in consolidado.items():
                row = f"{nome:<20}"
                for m in metric_names:
                    val = s.get(m, "N/A")
                    row += f"{val:>20}" if isinstance(val, str) else f"{val:>20.4f}"
                print(row)

            # Ranking
            print(f"\n🏆 RANKING RAGAS:")
            ranking = sorted(
                consolidado.items(),
                key=lambda x: x[1].get("avg_ragas_score", 0),
                reverse=True,
            )
            for i, (nome, s) in enumerate(ranking, 1):
                print(f"  {i}. {nome:<25} {s.get('avg_ragas_score', 0):.4f}")

            print(f"\n✅ Consolidado salvo em {cons_path}")

    print("\n✅ Pipeline completo!")


if __name__ == "__main__":
    main()
