# 2. LLM Puro (Sem RAG)

## 📝 Descrição Geral

Abordagem **controle** que testa a capacidade do LLM sozinho, sem recuperação de contexto. O modelo responde apenas com seu conhecimento prévio.

## 🎯 Estratégia

```python
PERGUNTA (PT)
    ↓
Contexto fixo (instrução genérica)
    ↓
LLM → Resposta
```

## 🔧 Implementação

**Arquivo:** `env/abordagem_llm_puro.py`

### Processo Detalhado:

1. **Pergunta**: Recebe pergunta em português
2. **Contexto fixo**: Usa prompt genérico sem documentos recuperados
3. **Geração**: LLM responde baseado apenas em conhecimento prévio
4. **Avaliação**: Compara com ground truth

## 📊 Características

| Característica | Valor |
|----------------|-------|
| Tipo | Sem recuperação |
| Velocidade | Muito rápida |
| Precisão | Baixa |
| Escalabilidade | Excelente |
| Requisitos | Apenas LLM |

## ✅ Pontos Fortes

- ✅ Testa limite do LLM
- ✅ Mais rápido (sem busca)
- ✅ Sem dependências de base
- ✅ Útil como baseline inferior
- ✅ Identifica alucinações

## ❌ Limitações

- ❌ Sem acesso à base técnica
- ❌ Alta taxa de alucinação
- ❌ Inconsistente
- ❌ Não rastreável
- ❌ Não atualizado

## 📈 Resultados Esperados

**Ranking típico:** 6º lugar (último)
**Score RAGAS médio:** 0.58
**Pontos fortes:** Velocidade
**Pontos fracos:** Precisão, consistência

## 🔬 Quando Usar

- Baseline inferior
- Testar conhecimento do LLM
- Identificar perguntas do domínio
- Comparar ganho do RAG

## 💡 Insights

Esta abordagem serve como **linha de base inferior**. A diferença de performance entre LLM puro e abordagens RAG demonstra o **valor agregado** da recuperação de contexto.

**Ganhos típicos com RAG:** +30-50% de melhoria

---

*Serve como controle para medir eficácia do RAG*
