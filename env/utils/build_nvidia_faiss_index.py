"""
================================================================================
PRÉ-PROCESSAMENTO OFFLINE: FAISS + EMBEDDINGS NVIDIA (NEMOTRON)
================================================================================
Gera embeddings semânticos reais via API da NVIDIA para a base de conhecimento,
indexa com FAISS e salva no disco. Pede a API Key interativamente se necessário.
================================================================================
"""

import getpass
import json
import os
import pathlib
import faiss
import numpy as np
from openai import OpenAI

import config
import comum

MODELO_EMBEDDING = "nvidia/llama-nemotron-embed-vl-1b-v2"

def main():
    print("🚀 Iniciando a preparação offline do índice FAISS (NVIDIA)...")
    
    # Tenta pegar a chave do config, depois do ambiente, senão pede interativamente
    api_key = getattr(config, "NVIDIA_API_KEY", None) or os.getenv("NVIDIA_API_KEY")
    if not api_key:
        print("\n🔑 NVIDIA_API_KEY não encontrada no config.py nem nas variáveis de ambiente.")
        api_key = getpass.getpass("Cole sua NVIDIA API Key aqui (o texto ficará oculto) e pressione Enter: ").strip()
        if not api_key:
            raise ValueError("❌ Nenhuma chave foi fornecida. O processo foi cancelado.")

    # Inicializa o cliente OpenAI apontando para a NVIDIA
    client = OpenAI(
        api_key=api_key,
        base_url="https://integrate.api.nvidia.com/v1"
    )

    # Carrega a base NOVA (base_conhecimento.json) especificamente para esta abordagem
    print("📚 Carregando base de conhecimento NOVA (base_conhecimento.json)...")
    base_nova_path = pathlib.Path(__file__).parent / "base_conhecimento.json"
    with open(base_nova_path, 'r', encoding='utf-8') as f:
        documentos = json.load(f)
    
    print(f"✅ {len(documentos)} documentos carregados da base nova")
    textos = []
    metadata_docs = []

    for doc in documentos:
        conteudo = doc.get("conteudo_completo", "") or doc.get("conteudo_preview", "")
        if len(conteudo) > 10000:
            conteudo = conteudo[10000:]  # Pula metadados iniciais
        
        # ESTRATÉGIA DE SEGURANÇA: Fatiar o documento em chunks de ~6000 caracteres
        # para nunca estourar o limite de 8192 tokens da API de embeddings da NVIDIA.
        tamanho_chunk = 6000
        for i in range(0, len(conteudo), tamanho_chunk):
            pedaço = conteudo[i : i + tamanho_chunk]
            if len(pedaço.strip()) > 100:  # Ignora pedaços irrelevantes/vazios
                textos.append(pedaço)
                metadata_docs.append({
                    "arquivo": doc["arquivo"],
                    "conteudo_completo": pedaço,  # Associa o sub-chunk correspondente
                    "conteudo_preview": doc.get("conteudo_preview", "")
                })

    print(f"📡 Total de sub-chunks gerados: {len(textos)}. Enviando para a API da NVIDIA em lotes...")
    
    batch_size = 10
    todos_embeddings = []

    for i in range(0, len(textos), batch_size):
        batch = textos[i : i + batch_size]
        # A API NVIDIA exige que modality tenha o mesmo tamanho de input
        modalities = ["text"] * len(batch)
        
        response = client.embeddings.create(
            input=batch,
            model=MODELO_EMBEDDING,
            encoding_format="float",
            extra_body={"modality": modalities, "input_type": "passage", "truncate": "NONE"}
        )
        
        batch_embs = [item.embedding for item in response.data]
        todos_embeddings.extend(batch_embs)
        print(f"  Processados {min(i + batch_size, len(textos))}/{len(textos)} sub-chunks...")

    matriz_embeddings = np.array(todos_embeddings, dtype=np.float32)

    print("📐 Normalizando L2 e criando índice FAISS...")
    faiss.normalize_L2(matriz_embeddings)

    dim = matriz_embeddings.shape[1]
    indice = faiss.IndexFlatIP(dim)
    indice.add(matriz_embeddings)

    output_dir = pathlib.Path(__file__).parent / "faiss_nvidia_store"
    output_dir.mkdir(exist_ok=True)

    faiss.write_index(indice, str(output_dir / "faiss_nvidia.index"))
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata_docs, f, ensure_ascii=False, indent=2)

    print(f"✅ Índice FAISS criado e salvo em 'faiss_nvidia_store/'. Total de vetores: {indice.ntotal}")

if __name__ == "__main__":
    main()