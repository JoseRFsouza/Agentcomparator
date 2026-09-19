# 1. TF-IDF (Baseline de Palavras-Chave)

## 📝 Descrição Geral

Abordagem **baseline** que utiliza TF-IDF (Term Frequency-Inverse Document Frequency) para recuperação de documentos por similaridade de palavras-chave.

## 🎯 Estratégia

```python
PERGUNTA (PT)
    ↓
Tradução PT→EN
    ↓
Vetorização TF-IDF
    ↓
Similaridade cosseno com base de chunks
    ↓
Top-K documentos mais similares
    ↓
Contexto → LLM → Resposta
```

## 🔧 Implementação

**Arquivo:** `env/abordagem_tfidf.py`

### Processo Detalhado:

1. **Tradução**: Pergunta em português é traduzida para inglês (manual em inglês)
2. **Vetorização**: Converte pergunta para vetor TF-IDF
3. **Similaridade**: Compara com todos os chunks usando similaridade cosseno
4. **Ranking**: Seleciona TOP_K documentos mais similares
5. **Geração**: Contexto + pergunta → LLM

## 📊 Características

| Característica | Valor |
|----------------|-------|
| Tipo | Palavras-chave |
| Velocidade | Muito rápida |
| Precisão | Média-Baixa |
| Escalabilidade | Boa |
| Requisitos | scikit-learn |

## ✅ Pontos Fortes

- ✅ Simples de implementar
- ✅ Muito rápida
- ✅ Não requer embeddings
- ✅ Baseline confiável
- ✅ Baixo consumo de memória

## ❌ Limitações

- ❌ Não captura semântica
- ❌ Depende de correspondência exata de termos
- ❌ Falha com sinônimos
- ❌ Sensível a variações de linguagem

## 📈 Resultados Esperados

**Ranking típico:** 5º lugar
**Score RAGAS médio:** 0.65
**Pontos fortes:** Velocidade, simplicidade
**Pontos fracos:** Semântica limitada

## 🔬 Quando Usar

- Baseline para comparação
- Recursos limitados
- Necessidade de velocidade extrema
- Domínio com vocabulário padronizado

## 💡 Melhorias Possíveis

- Stemming/Lematização
- N-grams
- Pesos customizados
- Filtro por seção ATA

---

*Base de conhecimento: 92.310 chunks de 800 caracteres*
