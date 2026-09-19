#!/usr/bin/env python3
"""
================================================================================
ORQUESTRADOR RAGAS - Avalia resultados das abordagens com métricas RAGAS
================================================================================
Uso:
  python orquestrador_ragas.py                          # todas as abordagens
  python orquestrador_ragas.py --abordagens faiss_nvidia toc_chunks
  python orquestrador_ragas.py --amostra 20             # primeiras 20 perguntas
================================================================================
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from ragas_evaluator import (
    RagasEvaluator,
    carregar_ground_truth_amm400,
    carregar_auditoria,
)


def main():
    parser = argparse.ArgumentParser(description="Avaliação RAGAS das abordagens")
    parser.add_argument(
        "--abordagens",
        nargs="*",
        default=None,
        help="Abordagens a avaliar (padrão: todas com auditoria existente)",
    )
    parser.add_argument(
        "--amostra",
        type=int,
        default=None,
        help="Limitar a N perguntas por abordagem",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help=f"Modelo LLM juiz (padrão: {config.LLM_MODEL})",
    )
    args = parser.parse_args()

    # Carregar ground truth
    gt_map = carregar_ground_truth_amm400()
    if not gt_map:
        print("❌ ground_truth_amm_400.json não encontrado!")
        return

    print(f"📋 Ground truth carregado: {len(gt_map)} perguntas")

    # Descobrir abordagens disponíveis (deduplicar: arquivos com _TIMESTAMP
    # pertencem à mesma abordagem canônica)
    ABORDAGENS_CONHECIDAS = [
        "tfidf", "llm_puro", "indice_toc", "faiss", "faiss_nvidia", "toc_chunks",
    ]
    if args.abordagens:
        abordagens = args.abordagens
    else:
        encontradas = set()
        for f in config.RESULTADOS_DIR.glob("auditoria_*.json"):
            nome = f.stem.replace("auditoria_", "")
            # Casa o prefixo canônico (ex: "faiss" em "faiss_20260919_003649")
            canonico = next(
                (a for a in ABORDAGENS_CONHECIDAS
                 if nome == a or nome.startswith(a + "_2")),
                None,
            )
            if canonico:
                encontradas.add(canonico)
        abordagens = [a for a in ABORDAGENS_CONHECIDAS if a in encontradas]

    if not abordagens:
        print("❌ Nenhuma auditoria encontrada em resultados/")
        return

    print(f"📦 Abordagens: {', '.join(abordagens)}")
    if args.amostra:
        print(f"🔢 Amostra: {args.amostra} perguntas por abordagem")

    # Inicializar avaliador
    evaluator = RagasEvaluator(judge_model=args.judge_model)

    # Avaliar cada abordagem
    all_results = {}
    for nome in abordagens:
        print(f"\n{'#' * 70}")
        print(f"# AVALIANDO: {nome}")
        print(f"{'#' * 70}")

        records = carregar_auditoria(nome)
        if not records:
            print(f"  ⚠️  Sem registros para {nome}")
            continue

        # Filtrar apenas perguntas que existem no ground truth
        records_com_gt = [r for r in records if r["pergunta"] in gt_map]
        print(f"  📊 {len(records)} registros, {len(records_com_gt)} com ground truth")

        if not records_com_gt:
            print(f"  ⚠️  Nenhuma pergunta de {nome} está no ground truth!")
            print(f"      Perguntas disponíveis no GT (primeiras 5):")
            for q in list(gt_map.keys())[:5]:
                print(f"        - {q[:80]}...")
            print(f"      Perguntas da abordagem (primeiras 5):")
            for r in records[:5]:
                print(f"        - {r['pergunta'][:80]}...")
            continue

        # Amostra
        if args.amostra:
            records_com_gt = records_com_gt[: args.amostra]

        results = evaluator.evaluate_batch(records_com_gt, gt_map)
        all_results[nome] = results

        # Salvar resultados individuais
        output_path = config.RESULTADOS_DIR / f"ragas_{nome}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n  ✅ Resultados salvos em {output_path}")

        # Resumo da abordagem
        summary = evaluator.summary(results)
        print(f"\n  📊 Resumo {nome}:")
        for k, v in summary.items():
            print(f"     {k}: {v}")

    # Relatório consolidado
    if all_results:
        print(f"\n{'=' * 70}")
        print("📊 RELATÓRIO CONSOLIDADO RAGAS")
        print(f"{'=' * 70}")

        consolidado = {}
        for nome, results in all_results.items():
            consolidado[nome] = evaluator.summary(results)

        # Salvar consolidado
        cons_path = config.RESULTADOS_DIR / "ragas_consolidado.json"
        with open(cons_path, "w", encoding="utf-8") as f:
            json.dump(consolidado, f, ensure_ascii=False, indent=2)

        # Imprimir tabela comparativa
        metric_names = [
            "avg_faithfulness", "avg_answer_relevancy", "avg_context_precision",
            "avg_context_recall", "avg_answer_correctness", "avg_answer_similarity",
            "avg_ragas_score",
        ]

        header = f"{'Abordagem':<20}" + "".join(f"{m.replace('avg_', ''):>20}" for m in metric_names)
        print(f"\n{header}")
        print("-" * len(header))

        for nome, s in consolidado.items():
            row = f"{nome:<20}"
            for m in metric_names:
                val = s.get(m, "N/A")
                row += f"{val:>20}" if isinstance(val, str) else f"{val:>20.4f}"
            print(row)

        # Ranking
        print(f"\n{'=' * 70}")
        print("🏆 RANKING RAGAS (por avg_ragas_score):")
        print("-" * 70)
        ranking = sorted(
            consolidado.items(),
            key=lambda x: x[1].get("avg_ragas_score", 0),
            reverse=True,
        )
        for i, (nome, s) in enumerate(ranking, 1):
            score = s.get("avg_ragas_score", 0)
            print(f"  {i}. {nome:<25} {score:.4f}")

        print(f"\n✅ Consolidado salvo em {cons_path}")


if __name__ == "__main__":
    main()
