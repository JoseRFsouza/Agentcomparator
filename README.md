# ✈️ Comparador de Abordagens RAG — Manual Boeing 737

> **Estudo comparativo de 6 técnicas de RAG aplicadas ao Manual de Manutenção do Boeing 737-300/400/500 (AMM)**

Este projeto avalia e ranqueia diferentes estratégias de **recuperação de informação (RAG)** para um chatbot de manutenção aeronáutica, gerando métricas objetivas com o framework **RAGAS**.

---

## 📋 Resumo

| Item | Descrição |
|------|-----------|
| **Objetivo** | Comparar 6 abordagens de RAG em manual técnico |
| **Base** | Manual AMM Boeing 737-300/400/500 |
| **Avaliação** | Métricas RAGAS + métricas customizadas |
| **Gabarito** | 400 perguntas com respostas verificadas (`ground_truth_amm_400.json`) |
| **Resultado** | Abordagem **TOC + Chunks** é a vencedora 🏆 |

---

## 🔬 As 6 Abordagens

1. **TF-IDF** — baseline por palavras-chave
2. **LLM Puro** — sem RAG, contexto fixo
3. **Índice TOC** — busca estruturada por capítulos ATA
4. **FAISS** — busca vetorial (TF-IDF + FAISS)
5. **FAISS NVIDIA** — embeddings NVIDIA + FAISS
6. **TOC + Chunks** ⭐ — híbrida (melhor resultado)

---

## 🚀 Início Rápido

```bash
# 1. Configurar chave API no env/config.py
# 2. Teste rápido (5 perguntas)
cd env
python pipeline_ragas.py --amostra 5

# 3. Avaliação completa (400 perguntas)
python pipeline_ragas.py --amostra 400
```

---

## 📊 Saídas (em `env/resultados/`)

- `comparativo_abordagens.csv` — ranking entre abordagens
- `ragas_consolidado.json` — métricas RAGAS
- `resumo_estatistico.txt` — estatísticas e ranking
- `auditoria_{abordagem}.json` — detalhe por pergunta

---

## 📖 Documentação Completa

👉 **Veja o [EXPLICACAO_PROJETO.md](EXPLICACAO_PROJETO.md)** para o manual completo:
- Configuração detalhada
- Todas as formas de execução
- Métricas RAGAS explicadas
- Troubleshooting e FAQ

---

## ⚙️ Requisitos

- Python 3.10+
- Chave da API NVIDIA
- `requests`, `faiss-cpu`, `scikit-learn`, `numpy`

---

## 🛠️ Solução Rápida de Problemas

| Problema | Solução |
|----------|---------|
| Erro 401 (API) | Verificar `NVIDIA_API_KEY` |
| Índice TOC não carrega | Ajustar caminho `INDICE_TOC` |
| Dependências faltando | `pip install requests faiss-cpu scikit-learn numpy` |

---

*Projeto acadêmico de pesquisa em IA aplicada à manutenção aeronáutica.*