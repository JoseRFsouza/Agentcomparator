# 4. FAISS com TF-IDF

## 📝 Descrição Geral

Abordagem vetorial que utiliza **FAISS** (Facebook AI Similarity Search) com vetores TF-IDF para busca semântica por similaridade.

## 🎯 Estratégia

```python
PERGUNTA (PT)
    ↓
Tradução PT→EN
    ↓
Vetorização TF-IDF
    ↓
Busca no índice FAISS
    ↓
Top-K documentos mais próximos
    ↓
Contexto → LLM → Resposta
```

## 🔧 Implementação

**Arquivo:** `env/abordagem_faiss.py`

### Processo Detalhado:

1. **Tradução**: PT → EN
2. **Vetorização**: Converte pergunta para vetor TF-IDF
3. **Índice FAISS**: Busca aproximada de vizinhos mais próximos
4. **Recuperação**: Retorna chunks mais similares
5. **Geração**: Contexto → LLM

## 📊 Características

| Característica | Valor |
|----------------|-------|
| Tipo | Vetorial |
| Velocidade | Rápida |
| Precisão | Média-Alta |
| Escalabilidade | Excelente |
| Requisitos | FAISS, TF-IDF |

## ✅ Pontos Fortes

- ✅ Busca semântica aproximada
- ✅ Muito rápida com FAISS
- ✅ Escalável para milhões de chunks
- ✅ Melhor que TF-IDF simples
- ✅ Busca aproximada eficiente

## ❌ Limitações

- ❌ Vetores TF-IDF limitados
- ❌ Não captura semântica profunda
- ❌ Requer construção de índice
- ❌ Memória para índice

## 📈 Resultados Esperados

**Ranking típico:** 3º lugar
**Score RAGAS médio:** 0.76
**Pontos fortes:** Velocidade + semântica
**Pontos fracos:** Limitação do TF-IDF

## 🔬 Quando Usar

- Base grande de documentos
- Necessidade de velocidade
- Recursos limitados para embeddings
- Busca aproximada suficiente

## 💡 Vantagem do FAISS

FAISS permite **busca em milissegundos** mesmo com 92.310 chunks:
- Busca exata: O(n)
- FAISS aproximada: O(log n)

---

*Índice FAISS construído com TF-IDF de 92.310 chunks*
