# 📊 As 6 Abordagens RAG - Documentação Detalhada

## Índice
1. [TF-IDF (Baseline)](1-tfidf.md)
2. [LLM Puro (Sem RAG)](2-llm-puro.md)
3. [Índice TOC (Tabela de Conteúdo)](3-indice-toc.md)
4. [FAISS com TF-IDF](4-faiss-tfidf.md)
5. [FAISS com Embeddings NVIDIA](5-faiss-nvidia.md)
6. [TOC + Chunks (Híbrida)](6-toc-chunks.md)

---

## 📋 Visão Geral

| # | Abordagem | Tipo | Arquivo | Complexidade |
|---|-----------|------|---------|--------------|
| 1 | **TF-IDF** | Palavras-chave | `abordagem_tfidf.py` | Baixa |
| 2 | **LLM Puro** | Sem recuperação | `abordagem_llm_puro.py` | Baixa |
| 3 | **Índice TOC** | Estrutural | `abordagem_indice_toc.py` | Média |
| 4 | **FAISS TF-IDF** | Vetorial | `abordagem_faiss.py` | Média |
| 5 | **FAISS NVIDIA** | Vetorial + Embeddings | `abordagem_faiss_nvidia.py` | Alta |
| 6 | **TOC + Chunks** | Híbrida 2 estágios | `abordagem_toc_chunks.py` | Alta ⭐ |

---

## 🎯 Objetivo da Comparação

Avaliar quais estratégias de recuperação produzem melhores respostas para perguntas técnicas do Manual de Manutenção Boeing 737 AMM.

**Métricas utilizadas:**
- RAGAS (Faithfulness, Answer Relevancy, Context Precision, Context Recall, Answer Correctness, Answer Similarity)
- Métricas customizadas (Precisão, Completude, Especificidade, Rastreabilidade, Confiança)

**Ground Truth:** 400 perguntas técnicas com respostas verificadas

---

*Documento gerado em: 2024*
