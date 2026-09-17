#!/usr/bin/env python3
"""
================================================================================
CONFIGURAÇÃO COMUM DO PROJETO COMPARADOR RAG vs LLM
================================================================================
Centraliza configurações, caminhos e constantes usadas por todas as abordagens.
Este módulo NÃO depende de PyTorch (usa scikit-learn + faiss + requests).

Abordagens comparadas:
  1. RAG TF-IDF            (baseline - palavras-chave)
  2. LLM puro com contexto (sem busca, contexto fixo)
  3. RAG com índice TOC    (busca estruturada por capítulo ATA)
  4. RAG com FAISS         (busca vetorial/semântica)

Todas usam a MESMA base de conhecimento (manuais Boeing 737) e o MESMO
modelo LLM (NVIDIA API) para geração de resposta, garantindo uma
comparação justa entre as estratégias de RECUPERAÇÃO.
================================================================================
"""

import os
from pathlib import Path

# ----------------------------------------------------------------------------
# CAMINHOS
# ----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

# Base de conhecimento processada (gerada por processar_base_documental.py)
INDICE_CONHECIMENTO = BASE_DIR / "base_conhecimento_indice.json"

# Índice TOC (tabela de conteúdo dos manuais, 4945 entradas, 43 capítulos ATA)
INDICE_TOC = BASE_DIR.parent / "amm_toc_database.json"

# Ground truth (gabarito de perguntas/respostas) - preencher pelo usuário
GROUND_TRUTH = BASE_DIR / "ground_truth.json"

# Perguntas de teste (uma por linha)
PERGUNTAS_TESTE = BASE_DIR / "perguntas_teste.txt"

# Diretório de resultados
RESULTADOS_DIR = BASE_DIR / "resultados"
RESULTADOS_DIR.mkdir(exist_ok=True)

# ----------------------------------------------------------------------------
# CONFIGURAÇÃO NVIDIA API (LLM e embeddings)
# ----------------------------------------------------------------------------
NVIDIA_API_KEY = "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"
NVIDIA_CHAT_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
NVIDIA_EMBED_URL = "https://integrate.api.nvidia.com/v1/embeddings"

# Modelo LLM usado em TODAS as abordagens (para comparação justa)
LLM_MODEL = "meta/llama-3.2-11b-vision-instruct"
LLM_JUDGE = "google/diffusiongemma-26b-a4b-it"  # modelo usado para avaliação semântica (testado e funcionando)
# ----------------------------------------------------------------------------
# PARÂMETROS DE RECUPERAÇÃO
# ----------------------------------------------------------------------------
TOP_K = 5            # número de documentos recuperados
CHUNK_CHARS = 1200   # tamanho do trecho de contexto enviado ao LLM
TEMPERATURA = 0.3    # temperatura do LLM (menor = mais determinístico)
MAX_TOKENS = 400     # limite de tokens da resposta

# ----------------------------------------------------------------------------
# Constantes de dimensão
# ----------------------------------------------------------------------------
# Os embeddings pré-computados (base_conhecimento_indice.json) têm 512 dims.
# A busca FAISS neste projeto usa TF-IDF (scikit-learn), pois o PyTorch está
# indisponível neste ambiente. O TF-IDF produz vetores esparsos de alta
# dimensionalidade que são indexados no FAISS (IndexFlatIP).
EMBED_DIM_LEGACY = 512