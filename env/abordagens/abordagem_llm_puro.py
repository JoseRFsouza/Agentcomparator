#!/usr/bin/env python3
"""
================================================================================
ABORDAGEM 2: LLM PURO COM CONTEXTO (SEM BUSCA)
================================================================================
Não faz busca/recuperação. Recebe a pergunta e um contexto FIXO (um resumo
dos manuais, ou o contexto do capítulo mais genérico), e gera a resposta
apenas com o LLM.

Objetivo: servir de "controle" para medir se a recuperação (RAG) realmente
agrega valor, comparado a um LLM que recebe o mesmo tipo de contexto mas
sem estratégia de busca.

O contexto aqui é fixo e igual para todas as perguntas: um panorama dos
capítulos ATA disponíveis na base (obtido do índice TOC).
================================================================================
"""

import argparse
from typing import List, Dict

import config
import comum


def construir_contexto_fixo() -> str:
    """
    Constrói um contexto fixo (panorama dos capítulos disponíveis),
    sem fazer busca específica por pergunta.
    """
    try:
        toc = comum.carregar_indice_toc()
    except Exception:
        toc = []

    # Lista única de capítulos ATA disponíveis
    capitulos = {}
    for entrada in toc:
        ata = entrada.get("ata_principal", "")
        nome = entrada.get("subject_name", "")
        if ata and ata not in capitulos:
            capitulos[ata] = nome

    linhas = ["Os seguintes capítulos estão disponíveis nos manuais Boeing 737:"]
    for ata in sorted(capitulos.keys()):
        linhas.append(f"- ATA {ata}: {capitulos[ata]}")

    return "\n".join(linhas)


def executar_abordagem(perguntas: List[str]) -> List[Dict]:
    """Executa a abordagem LLM puro com contexto fixo."""
    contexto_fixo = construir_contexto_fixo()
    print(f"  Contexto fixo com {contexto_fixo.count('ATA')} capítulos ATA.")

    registros = []
    for pergunta in perguntas:
        # Contexto fixo (igual para todas as perguntas)
        resposta, tempo, meta = comum.gerar_resposta_llm(
            pergunta, contexto=contexto_fixo
        )

        registro = comum.criar_registro_auditoria(
            abordagem="llm_puro",
            pergunta=pergunta,
            documentos=["contexto_fixo_toc"],
            contexto=contexto_fixo,
            resposta=resposta,
            tempo=tempo,
            metadados=meta,
        )
        registros.append(registro)
        print(f"  ✓ [{registro['score_final']:>6.2f}] {pergunta[:60]}")

    return registros


def main():
    parser = argparse.ArgumentParser(description="Abordagem 2: LLM puro com contexto")
    parser.add_argument("--perguntas", help="Arquivo de perguntas (uma por linha)")
    parser.add_argument("--pergunta", help="Pergunta única")
    args = parser.parse_args()

    if args.pergunta:
        perguntas = [args.pergunta]
    else:
        perguntas = comum.carregar_perguntas()

    if not perguntas:
        print("❌ Nenhuma pergunta fornecida.")
        return

    print(f"🚀 Abordagem LLM Puro: {len(perguntas)} perguntas")
    registros = executar_abordagem(perguntas)

    comum.salvar_resultados_csv("llm_puro", registros)
    comum.salvar_registros_json("llm_puro", registros)
    print("\n✅ Resultados salvos em resultados/resultados_llm_puro.csv")


if __name__ == "__main__":
    main()