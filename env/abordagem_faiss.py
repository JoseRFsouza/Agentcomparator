#!/usr/bin/env python3
"""
================================================================================
ABORDAGEM 4: RAG COM FAISS (BUSCA VETORIAL)
================================================================================
Usa FAISS (Facebook AI Similarity Search) para indexar e buscar os documentos
por similaridade vetorial. Como o PyTorch está indisponível neste ambiente,
os vetores são gerados via TF-IDF (scikit-learn) — vetores esparsos densificados
— e indexados com FAISS IndexFlatIP (produto interno = similaridade cosseno
para vetores normalizados).

Vantagens sobre busca linear: escalabilidade e velocidade, além de permitir
comparar uma infraestrutura de busca vetorial (FAISS) com as demais.
================================================================================
"""

import argparse
import re
from typing import List, Dict, Tuple

import numpy as np
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

import config
import comum


def extrair_termos(texto: str) -> str:
    texto = texto.lower()
    texto = re.sub(r"[^a-z0-9\s]", " ", texto)
    return texto


def construir_indice_faiss(documentos: List[Dict]):
    """
    Constrói vetores TF-IDF e indexa no FAISS.
    Retorna (vetorizador, indice_faiss, matriz_docs).
    
    ESTRATÉGIA MELHORADA: Busca em todo o conteúdo, ignorando metadados iniciais.
    Para manuais técnicos, o conteúdo relevante geralmente começa após os primeiros
    10000 caracteres (que contêm índices e metadados).
    """
    textos = []
    for doc in documentos:
        # Usa conteúdo completo
        conteudo = doc.get("conteudo_completo", "") or doc.get("conteudo_preview", "")
        
        # ESTRATÉGIA: Pular os primeiros 10000 caracteres (metadados/índice)
        # e pegar amostras do conteúdo técnico real
        if len(conteudo) > 10000:
            # Pega trechos do conteúdo técnico (após metadados)
            conteudo_tecnico = conteudo[10000:]
            
            # Para documentos muito grandes, usa amostragem estratégica
            if len(conteudo_tecnico) > 100000:
                # Pega 4 trechos de 25000 chars distribuídos pelo documento
                tamanho = len(conteudo_tecnico)
                trechos = []
                for i in range(4):
                    inicio = (tamanho // 4) * i
                    trechos.append(conteudo_tecnico[inicio:inicio + 25000])
                conteudo = " ".join(trechos)
            else:
                conteudo = conteudo_tecnico
        
        textos.append(extrair_termos(conteudo))
    
    # Aumentar max_features para capturar mais vocabulário técnico
    vetorizador = TfidfVectorizer(
        ngram_range=(1, 3),  # Usa unigramas, bigramas e trigramas
        max_features=20000,   # Mais features para capturar termos técnicos
        min_df=1,            # Inclui termos que aparecem pelo menos 1 vez
        max_df=0.95          # Ignora termos muito comuns
    )
    matriz = vetorizador.fit_transform(textos)

    # Densificar e normalizar (L2) para usar produto interno = cosseno
    matriz_densa = matriz.toarray().astype(np.float32)
    faiss.normalize_L2(matriz_densa)

    dim = matriz_densa.shape[1]
    indice = faiss.IndexFlatIP(dim)
    indice.add(matriz_densa)

    return vetorizador, indice, matriz_densa


def buscar_faiss(
    query: str,
    vetorizador: TfidfVectorizer,
    indice: faiss.IndexFlatIP,
    documentos: List[Dict],
    top_k: int = config.TOP_K,
) -> List[Tuple[Dict, float]]:
    """Busca os documentos mais relevantes usando FAISS."""
    query_vec = vetorizador.transform([extrair_termos(query)]).toarray().astype(np.float32)
    faiss.normalize_L2(query_vec)

    distancias, indices = indice.search(query_vec, top_k)

    resultados = []
    for rank, (dist, idx) in enumerate(zip(distancias[0], indices[0])):
        if idx < 0 or idx >= len(documentos):
            continue
        # dist é o produto interno (cosseno, pois vetores normalizados)
        resultados.append((documentos[idx], float(dist)))

    return resultados


def montar_contexto(resultados: List[Tuple[Dict, float]]) -> Tuple[str, List[str]]:
    """
    Monta o contexto a partir dos documentos recuperados.
    Usa o conteúdo mais relevante disponível.
    """
    trechos = []
    fontes = []
    for doc, score in resultados:
        fontes.append(doc["arquivo"])
        # Usa conteúdo completo se disponível, senão usa preview
        conteudo = doc.get("conteudo_completo", "") or doc.get("conteudo_preview", "")
        # Limita ao tamanho configurado
        trecho = conteudo[: config.CHUNK_CHARS]
        trechos.append(f"[Fonte: {doc['arquivo']} (cosseno {score:.4f})]\n{trecho}")
    return "\n\n---\n\n".join(trechos), fontes


def executar_abordagem(perguntas: List[str]) -> List[Dict]:
    """Executa a abordagem FAISS."""
    documentos = comum.carregar_base_conhecimento()
    vetorizador, indice, _ = construir_indice_faiss(documentos)
    print(f"  FAISS indexou {indice.ntotal} vetores (dimensão {indice.d}).")

    registros = []
    for pergunta in perguntas:
        resultados = buscar_faiss(pergunta, vetorizador, indice, documentos)
        contexto, fontes = montar_contexto(resultados)

        resposta, tempo, meta = comum.gerar_resposta_llm(pergunta, contexto=contexto)

        registro = comum.criar_registro_auditoria(
            abordagem="faiss",
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
    parser = argparse.ArgumentParser(description="Abordagem 4: RAG com FAISS")
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

    print(f"🚀 Abordagem FAISS: {len(perguntas)} perguntas")
    registros = executar_abordagem(perguntas)

    comum.salvar_resultados_csv("faiss", registros)
    comum.salvar_registros_json("faiss", registros)
    print("\n✅ Resultados salvos em resultados/resultados_faiss.csv")


if __name__ == "__main__":
    main()