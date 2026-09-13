#!/usr/bin/env python3
"""
================================================================================
ABORDAGEM 1: RAG com TF-IDF (BASELINE)
================================================================================
Busca documentos por similaridade de palavras (TF-IDF via scikit-learn),
extrai os trechos mais relevantes e gera resposta com LLM (NVIDIA API).

Não depende de PyTorch. Usa apenas scikit-learn + numpy + requests.
================================================================================
"""

import argparse
import re
from typing import List, Dict, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import config
import comum
import traducao


def extrair_ata_da_pergunta(pergunta: str) -> List[str]:
    """
    Extrai possíveis capítulos ATA baseado em palavras-chave da pergunta.
    Retorna lista de códigos ATA (ex: ['29', '32']).
    """
    pergunta_lower = pergunta.lower()
    
    # Mapeamento de palavras-chave para capítulos ATA
    mapeamento = {
        'hidraul': ['29'],  # Hydraulic Power
        'trem de pouso': ['32'],  # Landing Gear
        'pouso': ['32'],
        'landing gear': ['32'],
        'flight control': ['27'],  # Flight Controls
        'fcc': ['27'],
        'combustivel': ['28'],  # Fuel
        'motor': ['70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '80'],  # Engines
        'engine': ['70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '80'],
        'eletric': ['24'],  # Electrical Power
        'pneumatic': ['36'],  # Pneumatic
        'pressurizacao': ['21'],  # Air Conditioning
        'ar condicionado': ['21'],
        'navegacao': ['34'],  # Navigation
        'comunicacao': ['23'],  # Communications
    }
    
    atas_encontrados = []
    for palavra, atas in mapeamento.items():
        if palavra in pergunta_lower:
            atas_encontrados.extend(atas)
    
    return list(set(atas_encontrados))  # Remove duplicatas


def extrair_termos(texto: str) -> str:
    """Normaliza texto para vetorização TF-IDF."""
    texto = texto.lower()
    texto = re.sub(r"[^a-z0-9\s]", " ", texto)
    return texto


def construir_indice_tfidf(documentos: List[Dict]):
    """
    Constrói o vetorizador TF-IDF sobre os documentos.
    
    ESTRATÉGIA FINAL: Usa TODO o conteúdo disponível de cada documento.
    Para documentos muito grandes, limita o tamanho mas mantém diversidade.
    """
    textos = []
    for doc in documentos:
        # Usa conteúdo completo
        conteudo = doc.get("conteudo_completo", "") or doc.get("conteudo_preview", "")
        
        # Para documentos muito grandes (>500k), usa amostragem inteligente
        # que preserva seções importantes
        if len(conteudo) > 500000:
            # Divide em chunks de 100k e pega 1 a cada 5
            chunks = []
            for i in range(0, len(conteudo), 100000):
                if (i // 100000) % 5 == 0:  # Pega 1 a cada 5 chunks
                    chunks.append(conteudo[i:i+100000])
            conteudo = " ".join(chunks)
        elif len(conteudo) > 100000:
            # Para documentos médios, usa até 200k caracteres
            conteudo = conteudo[:200000]
        
        textos.append(extrair_termos(conteudo))
    
    # Vetorizador otimizado para termos técnicos
    vetorizador = TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=30000,  # Aumentado para capturar mais termos técnicos
        min_df=1,
        max_df=0.90,
        sublinear_tf=True  # Usa escala logarítmica para TF
    )
    matriz = vetorizador.fit_transform(textos)
    return vetorizador, matriz


def buscar_tfidf(
    query: str,
    vetorizador: TfidfVectorizer,
    matriz,
    documentos: List[Dict],
    top_k: int = config.TOP_K,
) -> List[Tuple[Dict, float]]:
    """
    Busca os documentos mais relevantes usando TF-IDF.
    
    MELHORIA: Prioriza documentos do ATA correto baseado em palavras-chave.
    """
    # Expandir consulta com tradução PT->EN
    query_expandida = traducao.expandir_consulta(query)
    query_vec = vetorizador.transform([extrair_termos(query_expandida)])
    similaridades = cosine_similarity(query_vec, matriz).flatten()
    
    # Extrair ATAs relevantes da pergunta
    atas_relevantes = extrair_ata_da_pergunta(query)
    
    # Boost para documentos do ATA correto
    if atas_relevantes:
        for i, doc in enumerate(documentos):
            # Extrai ATA do nome do arquivo (ex: "29___084.PDF" -> "29")
            nome_arquivo = doc["arquivo"]
            ata_doc = nome_arquivo[:2] if len(nome_arquivo) >= 2 else ""
            
            # Se o documento é do ATA relevante, aumenta o score
            if ata_doc in atas_relevantes:
                similaridades[i] *= 3.0  # Boost de 3x para ATA correto
    
    indices = np.argsort(similaridades)[::-1][:top_k]
    resultados = [(documentos[i], float(similaridades[i])) for i in indices]
    return resultados


def montar_contexto(resultados: List[Tuple[Dict, float]]) -> Tuple[str, List[str]]:
    """
    Monta o contexto a partir dos documentos recuperados.
    Usa o conteúdo mais relevante disponível.
    """
    trechos = []
    fontes = []
    for doc, score in resultados:
        if score > 0:
            fontes.append(doc["arquivo"])
            # Usa conteúdo completo se disponível, senão usa preview
            conteudo = doc.get("conteudo_completo", "") or doc.get("conteudo_preview", "")
            # Limita ao tamanho configurado
            trecho = conteudo[: config.CHUNK_CHARS]
            trechos.append(f"[Fonte: {doc['arquivo']} (score {score:.4f})]\n{trecho}")
    return "\n\n---\n\n".join(trechos), fontes


def executar_abordagem(perguntas: List[str]) -> List[Dict]:
    """Executa a abordagem TF-IDF para uma lista de perguntas."""
    documentos = comum.carregar_base_conhecimento()
    vetorizador, matriz = construir_indice_tfidf(documentos)

    registros = []
    for pergunta in perguntas:
        resultados = buscar_tfidf(pergunta, vetorizador, matriz, documentos)
        contexto, fontes = montar_contexto(resultados)

        resposta, tempo, meta = comum.gerar_resposta_llm(pergunta, contexto=contexto)

        registro = comum.criar_registro_auditoria(
            abordagem="tfidf",
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
    parser = argparse.ArgumentParser(description="Abordagem 1: RAG TF-IDF")
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

    print(f"🚀 Abordagem TF-IDF: {len(perguntas)} perguntas")
    registros = executar_abordagem(perguntas)

    comum.salvar_resultados_csv("tfidf", registros)
    comum.salvar_registros_json("tfidf", registros)
    print(f"\n✅ Resultados salvos em resultados/resultados_tfidf.csv")


if __name__ == "__main__":
    main()