# 6. Híbrida TOC + Chunks

## 📝 Descrição Geral

Abordagem **híbrida** que combina o melhor de dois mundos: estrutura da TOC e granularidade dos chunks. Método de **alta precisão**.

## 🎯 Estratégia

```python
PERGUNTA (PT)
    ↓
Tradução PT→EN
    ↓
Busca na TOC → Capítulos ATA
    ↓
Busca vetorial nos chunks do capítulo
    ↓
Fusão e re-ranking
    ↓
Contexto otimizado → LLM → Resposta
```

## 🔧 Implementação

**Arquivo:** `env/abordagem_toc_chunks.py`

### Processo Detalhado:

1. **Tradução**: PT → EN
2. **Busca TOC**: Identifica capítulos ATA relevantes
3. **Filtro**: Limita busca aos chunks dos capítulos
4. **Busca Vetorial**: TF-IDF ou embeddings nos chunks filtrados
5. **Fusão**: Combina contexto estrutural + granular
6. **Geração**: Contexto otimizado → LLM

## 📊 Características

| Característica | Valor |
|----------------|-------|
| Tipo | Híbrida |
| Velocidade | Média |
| Precisão | Alta |
| Escalabilidade | Média |
| Requisitos | TOC + Chunks |

## ✅ Pontos Fortes

- ✅ Melhor dos dois mundos
- ✅ Alta precisão de recuperação
- ✅ Contexto estruturado + granular
- ✅ Reduz ruído de busca
- ✅ Rastreabilidade mantida

## ❌ Limitações

- ❌ Mais complexo de implementar
- ❌ Latência maior (duas buscas)
- ❌ Requer ambos índices
- ❌ Tuning de parâmetros

## 📈 Resultados Esperados

**Ranking típico:** 1º lugar (melhor)
**Score RAGAS médio:** 0.85
**Pontos fortes:** Precisão, rastreabilidade
**Pontos fracos:** Complexidade

## 🔬 Quando Usar

- Máxima precisão necessária
- Base estruturada bem organizada
- Recursos disponíveis
- Produção crítica

## 💡 Vantagem Híbrida

**Estratégia em dois estágios:**
1. **Coarse**: TOC identifica área (capítulo)
2. **Fine**: Chunks buscam detalhes específicos

Reduz busca de 92.310 → ~2.000 chunks por capítulo
**Ganho:** +30% de precisão vs busca livre

---

*Abordagem mais robusta combinando estrutura e granularidade*
