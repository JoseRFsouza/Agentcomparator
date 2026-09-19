# 3. Índice TOC (Tabela de Conteúdo)

## 📝 Descrição Geral

Abordagem que utiliza **busca estruturada** na Tabela de Conteúdo (TOC) do manual AMM para identificar capítulos ATA relevantes.

## 🎯 Estratégia

```python
PERGUNTA (PT)
    ↓
Tradução PT→EN
    ↓
Busca na TOC (4.945 entradas)
    ↓
Identifica capítulos ATA relevantes
    ↓
Recupera documentos dos capítulos
    ↓
Contexto → LLM → Resposta
```

## 🔧 Implementação

**Arquivo:** `env/abordagem_indice_toc.py`

### Processo Detalhado:

1. **Tradução**: PT → EN
2. **Busca TOC**: Compara pergunta com títulos de seções da TOC
3. **Matching**: Identifica capítulos ATA (e.g., 35-00-00, 06-00-00)
4. **Recuperação**: Busca documentos dos capítulos identificados
5. **Geração**: Contexto estruturado → LLM

## 📊 Características

| Característica | Valor |
|----------------|-------|
| Tipo | Estrutural |
| Velocidade | Rápida |
| Precisão | Média |
| Escalabilidade | Boa |
| Requisitos | Índice TOC |

## ✅ Pontos Fortes

- ✅ Rastreabilidade por capítulo
- ✅ Alta precisão estrutural
- ✅ Contexto organizado
- ✅ Boa para navegação hierárquica
- ✅ Menos ruído que busca livre

## ❌ Limitações

- ❌ Dependente da qualidade da TOC
- ❌ Pode perder detalhes granulares
- ❌ Requer matching de capítulos
- ❌ Não busca dentro de seções

## 📈 Resultados Esperados

**Ranking típico:** 4º lugar
**Score RAGAS médio:** 0.71
**Pontos fortes:** Rastreabilidade, estrutura
**Pontos fracos:** Granularidade

## 🔬 Quando Usar

- Manuais com estrutura clara
- Necessidade de rastreabilidade
- Perguntas sobre procedimentos gerais
- Quando capítulos são bem definidos

## 💡 Vantagem Única

**Rastreabilidade completa**: Sempre sabe exatamente de qual capítulo ATA a resposta veio.

Exemplo: "Capítulo 35-00-00 (Oxigênio) - Seção X"

---

*Índice TOC: 4.945 entradas organizadas por ATA*
