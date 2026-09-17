#!/usr/bin/env python3
"""
================================================================================
ABORDAGEM: FAISS + EMBEDDINGS NVIDIA (RUNTIME)
================================================================================
Busca vetorial semântica usando FAISS pré-carregado do disco e vetorização 
da pergunta do usuário via API da NVIDIA.
================================================================================
"""

import argparse
import getpass
import json
import os
import pathlib
from typing import List, Dict, Tuple

import faiss
import numpy as np
from openai import OpenAI

import config
import comum

MODELO_EMBEDDING = "nvidia/llama-nemotron-embed-vl-1b-v2"

_store = {}

def carregar_store():
    if not _store:
        store_dir = pathlib.Path("faiss_nvidia_store")
        if not store_dir.exists():
            raise FileNotFoundError("❌ Pasta faiss_nvidia_store não encontrada! Execute primeiro o script 'build_nvidia_faiss_index.py'.")
        
        print("📂 Carregando índice FAISS (NVIDIA)...")
        _store["indice"] = faiss.read_index(str(store_dir / "faiss_nvidia.index"))
        with open(store_dir / "metadata.json", "r", encoding="utf-8") as f:
            _store["documentos"] = json.load(f)
            
        # Tenta pegar do config.py primeiro, depois do ambiente
        api_key = getattr(config, "NVIDIA_API_KEY", None) or os.getenv("NVIDIA_API_KEY")
        if not api_key:
            print("\n🔑 NVIDIA_API_KEY não encontrada no config.py nem nas variáveis de ambiente.")
            api_key = getpass.getpass("Cole sua NVIDIA API Key aqui (o texto ficará oculto) e pressione Enter: ").strip()
            if not api_key:
                raise ValueError("❌ NVIDIA_API_KEY não definida!")
            
        _store["client"] = OpenAI(
            api_key=api_key,
            base_url="https://integrate.api.nvidia.com/v1"
        )
        print(f"⚡ FAISS pronto com {_store['indice'].ntotal} vetores semânticos.")
    return _store


def gerar_embedding_pergunta(client: OpenAI, pergunta: str) -> np.ndarray:
    """Gera o embedding semântico para a pergunta usando o input_type 'query'."""
    response = client.embeddings.create(
        input=[pergunta],
        model=MODELO_EMBEDDING,
        encoding_format="float",
        extra_body={"modality": ["text"], "input_type": "query", "truncate": "NONE"}
    )
    
    emb = response.data[0].embedding
    vec = np.array([emb], dtype=np.float32)
    faiss.normalize_L2(vec)
    return vec


def buscar_faiss_nvidia(query: str, top_k: int = config.TOP_K) -> List[Tuple[Dict, float]]:
    store = carregar_store()
    indice = store["indice"]
    documentos = store["documentos"]
    client = store["client"]

    query_vec = gerar_embedding_pergunta(client, query)
    distancias, indices = indice.search(query_vec, top_k)

    resultados = []
    for dist, idx in zip(distancias[0], indices[0]):
        if 0 <= idx < len(documentos):
            resultados.append((documentos[idx], float(dist)))
    return resultados


def executar_abordagem(perguntas: List[str]) -> List[Dict]:
    carregar_store()
    registros = []

    for pergunta in perguntas:
        resultados = buscar_faiss_nvidia(pergunta, top_k=config.TOP_K)
        
        trechos = []
        fontes = []
        for doc, score in resultados:
            fontes.append(doc["arquivo"])
            conteudo = doc.get("conteudo_completo", "") or doc.get("conteudo_preview", "")
            trecho = conteudo[: config.CHUNK_CHARS]
            trechos.append(f"[Fonte: {doc['arquivo']} (similaridade {score:.4f})]\n{trecho}")
            
        contexto = "\n\n---\n\n".join(trechos)
        resposta, tempo, meta = comum.gerar_resposta_llm(pergunta, contexto=contexto)

        registro = comum.criar_registro_auditoria(
            abordagem="faiss_nvidia",
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
    parser = argparse.ArgumentParser(description="Abordagem FAISS + NVIDIA Nemotron Embeddings")
    parser.add_argument("--perguntas", help="Arquivo de perguntas")
    parser.add_argument("--pergunta", help="Pergunta única")
    args = parser.parse_args()

    perguntas = [args.pergunta] if args.pergunta else comum.carregar_perguntas()
    if not perguntas:
        print("❌ Nenhuma pergunta fornecida.")
        return

    print(f"🚀 Executando FAISS + Nemotron Embeddings para {len(perguntas)} perguntas")
    registros = executar_abordagem(perguntas)

    comum.salvar_resultados_csv("faiss_nvidia", registros)
    comum.salvar_registros_json("faiss_nvidia", registros)
    print("\n✅ Resultados salvos em resultados/resultados_faiss_nvidia.csv")


if __name__ == "__main__":
    main()