#!/usr/bin/env python3
"""
================================================================================
PROCESSADOR DE BASE DOCUMENTAL COM CHUNKING POR FRASE
================================================================================
Diferente do processador original, este:
  1. Divide cada PDF em CHUNKS pequenos (por frase/bloco de ~500-1000 chars)
  2. Indexa cada chunk separadamente com TF-IDF
  3. Mantém metadados de origem (arquivo, posição) em cada chunk

Isso resolve o problema dos documentos gigantes onde o conteúdo técnico
fica "diluído" entre metadados e índices.

Uso:
    python processar_base_chunks.py --pasta "C:/caminho/PDF"
================================================================================
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import List, Dict

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    print("✓ scikit-learn importado")
except ImportError:
    print("❌ Instale: pip install scikit-learn")
    sys.exit(1)

try:
    import PyPDF2
    print("✓ PyPDF2 importado")
except ImportError:
    print("❌ Instale: pip install PyPDF2")
    sys.exit(1)


# ============================================================================
# LEITURA DE PDF
# ============================================================================

def ler_pdf(caminho: str) -> str:
    """Lê PDF e retorna texto completo."""
    try:
        texto = []
        with open(caminho, 'rb') as f:
            leitor = PyPDF2.PdfReader(f)
            for pagina in leitor.pages:
                t = pagina.extract_text()
                if t:
                    texto.append(t)
        return "\n".join(texto)
    except Exception as e:
        print(f"  ⚠️  Erro ao ler {caminho}: {e}")
        return ""


# ============================================================================
# CHUNKING POR FRASE/BLOCO
# ============================================================================

def dividir_em_chunks(texto: str, arquivo: str,
                      chunk_chars: int = 800,
                      overlap: int = 100) -> List[Dict]:
    """
    Divide o texto em chunks de ~chunk_chars caracteres,
    respeitando quebras de parágrafo/frase sempre que possível.
    Cada chunk mantém referência ao arquivo de origem.
    """
    chunks = []

    # Normalizar: juntar quebras de linha excessivas
    texto = re.sub(r'\n{3,}', '\n\n', texto)

    # Remover linhas que são apenas metadados de página
    # (padrões como "201 Mar25/2015", "05-00-00", números de página)
    linhas = texto.split('\n')
    linhas_limpas = []
    for linha in linhas:
        linha_s = linha.strip()
        # Ignorar linhas que são só datas/números de página/códigos de seção
        if re.match(r'^[A-Z]?\s*\d+\s+(Mar|Sep|Jan|Feb|Apr|May|Jun|Jul|Aug|Oct|Nov|Dec)', linha_s):
            continue
        if re.match(r'^\d{2}-\d{2}-\d{2}$', linha_s):
            continue
        if re.match(r'^(Page|PAGE)\s+\d+', linha_s):
            continue
        if len(linha_s) < 3:
            continue
        linhas_limpas.append(linha)

    texto_limpo = '\n'.join(linhas_limpas)

    # Dividir em chunks com overlap
    pos = 0
    chunk_id = 0
    while pos < len(texto_limpo):
        fim = pos + chunk_chars

        # Tentar quebrar em limite de frase/parágrafo
        if fim < len(texto_limpo):
            # Procurar quebra de parágrafo próxima
            quebra = texto_limpo.rfind('\n\n', pos + chunk_chars // 2, fim + 200)
            if quebra > pos:
                fim = quebra
            else:
                # Tentar quebra de frase
                quebra = texto_limpo.rfind('. ', pos + chunk_chars // 2, fim + 100)
                if quebra > pos:
                    fim = quebra + 1

        chunk_texto = texto_limpo[pos:fim].strip()

        # Só adicionar chunks com conteúdo significativo (>100 chars)
        if len(chunk_texto) > 100:
            chunks.append({
                "chunk_id": chunk_id,
                "arquivo": arquivo,
                "pos_inicio": pos,
                "texto": chunk_texto,
            })
            chunk_id += 1

        pos = fim - overlap if fim > overlap else fim

    return chunks


# ============================================================================
# INDEXAÇÃO TF-IDF DOS CHUNKS
# ============================================================================

def indexar_chunks(todos_chunks: List[Dict]):
    """Indexa todos os chunks com TF-IDF."""
    print(f"\n📊 Indexando {len(todos_chunks)} chunks com TF-IDF...")

    textos = [c["texto"].lower() for c in todos_chunks]

    vetorizador = TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=50000,
        min_df=1,
        max_df=0.90,
        sublinear_tf=True,
    )
    matriz = vetorizador.fit_transform(textos)

    # Salvar vocabulário para referência
    print(f"  Vocabulário: {len(vetorizador.vocabulary_)} termos")

    return vetorizador, matriz


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Processa PDFs em chunks pequenos para RAG"
    )
    parser.add_argument("--pasta", required=True, help="Pasta com PDFs")
    parser.add_argument("--saida", default=".", help="Pasta de saída")
    parser.add_argument("--chunk-chars", type=int, default=800,
                        help="Tamanho de cada chunk em caracteres (padrão: 800)")
    args = parser.parse_args()

    pasta = Path(args.pasta)
    if not pasta.exists():
        print(f"❌ Pasta não existe: {pasta}")
        return

    # ---- Passo 1: Ler e dividir PDFs em chunks --------------------------
    pdfs = sorted(pasta.glob("*.PDF")) + sorted(pasta.glob("*.pdf"))
    print(f"\n📁 Encontrados {len(pdfs)} PDFs")

    todos_chunks = []
    for pdf in pdfs:
        print(f"  📄 {pdf.name}...", end=" ")
        texto = ler_pdf(str(pdf))
        if not texto.strip():
            print("⚠️ vazio")
            continue
        chunks = dividir_em_chunks(texto, pdf.name, args.chunk_chars)
        print(f"✓ {len(chunks)} chunks ({len(texto)} chars)")
        todos_chunks.extend(chunks)

    print(f"\n📦 Total: {len(todos_chunks)} chunks de {len(pdfs)} PDFs")

    # ---- Passo 2: Indexar com TF-IDF ------------------------------------
    vetorizador, matriz = indexar_chunks(todos_chunks)

    # ---- Passo 3: Salvar índice de chunks --------------------------------
    # Salvar chunks (sem os embeddings TF-IDF — a matriz é esparsa e grande;
    # vamos reconstruir o vetorizador na hora da busca)
    indice = {
        "versao": "3.0-chunks",
        "total_chunks": len(todos_chunks),
        "chunk_chars": args.chunk_chars,
        "chunks": todos_chunks,
    }

    caminho_saida = Path(args.saida) / "base_chunks.json"
    with open(caminho_saida, 'w', encoding='utf-8') as f:
        json.dump(indice, f, ensure_ascii=False)

    tamanho_mb = caminho_saida.stat().st_size / 1024 / 1024
    print(f"\n💾 Índice de chunks salvo em: {caminho_saida} ({tamanho_mb:.1f} MB)")

    # Estatísticas
    chunks_por_arquivo = {}
    for c in todos_chunks:
        chunks_por_arquivo[c["arquivo"]] = chunks_por_arquivo.get(c["arquivo"], 0) + 1

    print("\n📊 Chunks por arquivo (top 10):")
    for arq, n in sorted(chunks_por_arquivo.items(), key=lambda x: -x[1])[:10]:
        print(f"  • {arq}: {n} chunks")

    print("\n✅ Processamento concluído!")


if __name__ == "__main__":
    main()
