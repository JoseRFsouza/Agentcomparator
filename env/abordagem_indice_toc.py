#!/usr/bin/env python3
"""
================================================================================
ABORDAGEM 3: RAG COM ÍNDICE TOC (BUSCA ESTRUTURADA)
================================================================================
Usa o índice TOC (tabela de conteúdo, 4945 entradas) para identificar os
capítulos/assuntos relevantes a uma pergunta, como um humano consultaria o
sumário do manual. Depois extrai o conteúdo dos documentos correspondentes
e gera resposta com LLM.

Não depende de PyTorch.
================================================================================
"""

import argparse
import re
from typing import List, Dict, Tuple

import config
import comum
import traducao


def normalizar(texto: str) -> str:
    import unicodedata
    texto = texto.lower()
    texto = unicodedata.normalize("NFKD", texto)
    texto = texto.encode("ascii", "ignore").decode("ascii")
    return texto


def extrair_palavras(texto: str) -> set:
    return set(re.findall(r"[a-z0-9]{3,}", normalizar(texto)))


def buscar_no_toc(pergunta: str, toc: List[Dict], top_k: int = config.TOP_K) -> List[Dict]:
    """
    Busca no índice TOC as entradas (task_name, subject_name) mais relevantes
    à pergunta, pontuando por sobreposição de palavras.
    
    MELHORIA: Agora traduz a pergunta para inglês antes de buscar,
    pois o TOC está em inglês.
    """
    # Traduzir pergunta para inglês para casar com o TOC em inglês
    pergunta_traduzida = traducao.expandir_consulta(pergunta)
    
    # Usar ambas as versões (original + traduzida) para busca
    palavras_pergunta = extrair_palavras(pergunta) | extrair_palavras(pergunta_traduzida)
    
    if not palavras_pergunta:
        return []

    pontuadas = []
    for entrada in toc:
        texto_entrada = (
            entrada.get("subject_name", "")
            + " "
            + entrada.get("task_name", "")
            + " "
            + entrada.get("chapter_section_subject", "")
        )
        palavras_entrada = extrair_palavras(texto_entrada)
        if not palavras_entrada:
            continue
        intersecao = palavras_pergunta & palavras_entrada
        # Score = fração de sobreposição (Jaccard-like)
        uniao = palavras_pergunta | palavras_entrada
        score = len(intersecao) / len(uniao) if uniao else 0.0
        if score > 0:
            pontuadas.append((entrada, score))

    pontuadas.sort(key=lambda x: x[1], reverse=True)

    # Remover duplicatas por ata_principal (manter o primeiro de cada capítulo)
    vistos = set()
    unicos = []
    for entrada, score in pontuadas:
        ata = entrada.get("ata_principal", "")
        chave = (ata, entrada.get("chapter_section_subject", ""))
        if chave not in vistos:
            vistos.add(chave)
            unicos.append((entrada, score))
        if len(unicos) >= top_k:
            break

    return unicos


def buscar_documentos_por_ata(
    ata: str, documentos: List[Dict]
) -> Dict:
    """Encontra o documento da base de conhecimento correspondente a um ATA."""
    # O arquivo segue o padrão: "29___084.PDF" -> ata "29"
    for doc in documentos:
        nome = doc["arquivo"].replace("_", "").replace(".PDF", "").replace(".pdf", "")
        prefixo = nome[:2]  # ex: "29" de "29___084"
        if prefixo == ata:
            return doc
        # tentar também com 3 caracteres (ex: "49__B" -> ata 49)
        if ata and nome.startswith(ata):
            return doc
    return None


def executar_abordagem(perguntas: List[str]) -> List[Dict]:
    """Executa a abordagem RAG com índice TOC."""
    documentos = comum.carregar_base_conhecimento()
    toc = comum.carregar_indice_toc()

    registros = []
    for pergunta in perguntas:
        entradas_relevantes = buscar_no_toc(pergunta, toc)

        trechos = []
        fontes = []
        for entrada, score in entradas_relevantes:
            ata = entrada.get("ata_principal", "")
            doc = buscar_documentos_por_ata(ata, documentos)
            if doc:
                fonte = f"ATA {ata} - {entrada.get('subject_name', '')[:50]}"
                fontes.append(f"{doc['arquivo']} ({fonte})")
                # Usa conteúdo completo se disponível, senão usa preview
                conteudo = doc.get("conteudo_completo", "") or doc.get("conteudo_preview", "")
                # Limita ao tamanho configurado
                trecho = conteudo[: config.CHUNK_CHARS]
                trechos.append(
                    f"[{doc['arquivo']} | {entrada.get('task_name', '')}]\n{trecho}"
                )

        contexto = "\n\n---\n\n".join(trechos)

        resposta, tempo, meta = comum.gerar_resposta_llm(pergunta, contexto=contexto)

        registro = comum.criar_registro_auditoria(
            abordagem="indice_toc",
            pergunta=pergunta,
            documentos=fontes,
            contexto=contexto,
            resposta=resposta,
            tempo=tempo,
            metadados=meta,
        )
        registros.append(registro)
        print(f"  ✓ [{registro['score_final']:>6.2f}] {pergunta[:60]}")

    return registros


def main():
    parser = argparse.ArgumentParser(description="Abordagem 3: RAG com índice TOC")
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

    print(f"🚀 Abordagem Índice TOC: {len(perguntas)} perguntas")
    registros = executar_abordagem(perguntas)

    comum.salvar_resultados_csv("indice_toc", registros)
    comum.salvar_registros_json("indice_toc", registros)
    print("\n✅ Resultados salvos em resultados/resultados_indice_toc.csv")


if __name__ == "__main__":
    main()