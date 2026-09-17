# 📘 Manual Completo do Projeto: Comparador de Abordagens RAG — Boeing 737

---

## 🎯 Objetivo do Projeto

Este projeto é um **estudo comparativo de 6 abordagens de RAG (Retrieval-Augmented Generation)** aplicadas ao **Manual de Manutenção do Boeing 737-300/400/500 (AMM)**.

A finalidade é **avaliar** e **ranquear** quais estratégias de recuperação de informação (busca) produzem as melhores respostas para perguntas técnicas de manutenção aeronáutica.

O projeto gera **métricas objetivas** usando:
1. **Métricas RAGAS** (framework padrão da indústria para avaliação de RAG)
2. **Métricas customizadas** (precisão, completude, confiança, etc.)

---

## 🏗️ Estrutura do Projeto

```
Agentcomparator/
├── EXPLICACAO_PROJETO.md          # Este documento
├── README.md                      # Resumo rápido (para GitHub)
├── dados_treino.json              # Dados para fine-tuning (extras)
├── tasks_extraidas.json           # Tasks extraídas dos manuais
├── .venv/                         # Ambiente virtual Python
│
└── env/                           # Código principal
    ├── config.py                  # ⚙️ CONFIGURAÇÃO (API, caminhos, parâmetros)
    ├── comum.py                   # Funções compartilhadas (métricas, CSV, auditoria)
    ├── traducao.py                # Tradução PT→EN para busca no manual
    │
    ├── # ─── AS 6 ABORDAGENS ─── #
    ├── abordagem_tfidf.py         # 1. TF-IDF (baseline palavras-chave)
    ├── abordagem_llm_puro.py      # 2. LLM puro (sem RAG, contexto fixo)
    ├── abordagem_indice_toc.py    # 3. RAG com índice TOC (capítulos ATA)
    ├── abordagem_faiss.py         # 4. RAG com FAISS (TF-IDF + indexação vetorial)
    ├── abordagem_faiss_nvidia.py  # 5. RAG com FAISS (embeddings NVIDIA)
    ├── abordagem_toc_chunks.py    # 6. Híbrida TOC + Chunks (melhor resultado)
    │
    ├── # ─── ORQUESTRADORES / PIPELINES ─── #
    ├── orquestrador.py            # Executa abordagens + métricas customizadas
    ├── orquestrador_ragas.py      # Avalia abordagens com métricas RAGAS
    ├── pipeline_ragas.py          # Pipeline completo (RAG + RAGAS em um)
    ├── ragas_evaluator.py         # Implementação das métricas RAGAS
    │
    ├── # ─── BASE DE CONHECIMENTO ─── #
    ├── base_conhecimento.json     # Base original processada
    ├── base_conhecimento_indice.json # Base indexada por documento
    ├── base_chunks.json           # 92.310 chunks de 800 caracteres
    ├── processar_base_chunks.py   # Script que gera os chunks
    │
    ├── # ─── GROUND TRUTH (GABARITO) ─── #
    ├── ground_truth.json          # Gabarito inicial (vazio/usuário)
    ├── ground_truth_amm_400.json  # ✅ 400 perguntas + respostas corretas
    ├── gerar_ground_truth.py      # Script para gerar gabarito
    │
    ├── # ─── DADOS AUXILIARES ─── #
    ├── perguntas_teste.txt        # Perguntas livres (uma por linha)
    ├── comparar_bases.py          # Compara bases de conhecimento
    ├── comparacao_bases.json      # Resultado da comparação
    ├── gerar_dataset_ragas.py     # Gera dataset para RAGAS
    ├── gerar_dados_treino.py      # Gera dados de treino
    ├── dados_treino.json          # Dados de treino gerados
    ├── build_nvidia_faiss_index.py # Constrói índice FAISS NVIDIA
    │
    ├── # ─── ÍNDICE FAISS NVIDIA ─── #
    ├── faiss_nvidia_store/        # Armazenamento do índice
    │   ├── faiss_nvidia.index     # Índice vetorial FAISS
    │   └── metadata.json          # Metadados dos embeddings
    │
    └── resultados/                # 📊 RESULTADOS GERADOS
        ├── auditoria_{abordagem}.json    # Auditoria detalhada por pergunta
        ├── resultados_{abordagem}.csv    # Resultados por abordagem
        ├── comparativo_abordagens.csv    # Comparativo consolidado
        ├── ragas_consolidado.json        # Métricas RAGAS consolidadas
        ├── ragas_toc_chunks.json         # Métricas RAGAS (por abordagem)
        └── resumo_estatistico.txt        # Ranking e estatísticas
```

---

## 🔬 As 6 Abordagens Comparadas

| # | Abordagem | Arquivo | Estratégia | Pontos Fortes |
|---|-----------|---------|------------|---------------|
| 1 | **TF-IDF** | `abordagem_tfidf.py` | Busca por palavras-chave | Simples, rápido, baseline |
| 2 | **LLM Puro** | `abordagem_llm_puro.py` | Sem busca; contexto fixo no prompt | Testa capacidade do LLM sozinho |
| 3 | **Índice TOC** | `abordagem_indice_toc.py` | Busca estruturada por capítulos ATA | Rastreabilidade por capítulo |
| 4 | **FAISS** | `abordagem_faiss.py` | Vetores TF-IDF indexados no FAISS | Busca semântica por similaridade |
| 5 | **FAISS NVIDIA** | `abordagem_faiss_nvidia.py` | Embeddings NVIDIA (512 dims) + FAISS | Embeddings de alta qualidade |
| 6 | **TOC + Chunks** ⭐ | `abordagem_toc_chunks.py` | 2 estágios: TOC + busca em chunks | **Melhor resultado do projeto** |

### Detalhes da Abordagem 6 (Vencedora) 🏆

A abordagem **TOC + Chunks** funciona em **2 estágios**:

```
PERGUNTA (PT)
    │
    ▼  Estágio 1: Busca no TOC
    (Traduz PT→EN, compara com tabela de conteúdo)
    → Identifica capítulos ATA relevantes
    │
    ▼  Estágio 2: Busca em chunks
    (Busca TF-IDF apenas nos chunks dos ATAs identificados)
    → Retorna trechos mais relevantes
    │
    ▼
RESPOSTA GERADA PELO LLM
```

Este método combina **precisão estrutural** (TOC) com **cobertura granular** (chunks).

---

## ⚙️ Configuração do Projeto

### Pré-requisitos

| Requisito | Versão | Notas |
|-----------|--------|-------|
| **Python** | 3.10+ | Projeto testado com 3.14 |
| **Chave NVIDIA API** | — | Necessária para LLM e embeddings |

### Instalação

```bash
# 1. Clonar repositório
git clone <seu-repositorio>
cd Agentcomparator

# 2. Criar ambiente virtual
python -m venv .venv

# 3. Ativar ambiente
#   Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
#   Linux/Mac:
source .venv/bin/activate

# 4. Instalar dependências
pip install requests faiss-cpu scikit-learn numpy python-dotenv
```

### Configuração da API (IMPORTANTE) ⚠️

Edite o arquivo `env/config.py` e configure:

```python
# ===== CONFIGURAÇÃO NVIDIA API =====
NVIDIA_API_KEY = "sua-chave-nvidia-aqui"     # ← Suba sua chave

# URLs da API
NVIDIA_CHAT_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
NVIDIA_EMBED_URL = "https://integrate.api.nvidia.com/v1/embeddings"

# Modelo LLM (mesmo para todas as abordagens - comparação justa)
LLM_MODEL = "meta/llama-3.2-11b-vision-instruct"

# ===== PARÂMETROS DE RECUPERAÇÃO =====
TOP_K = 5            # nº de documentos recuperados
CHUNK_CHARS = 1200   # tamanho do contexto enviado ao LLM
TEMPERATURA = 0.3    # determinismo da resposta
MAX_TOKENS = 400     # limite de tokens
```

### Caminhos de Arquivos

```python
# ===== CAMINHOS (env/config.py) =====
BASE_DIR = Path(__file__).resolve().parent

# Base de conhecimento processada
INDICE_CONHECIMENTO = BASE_DIR / "base_conhecimento_indice.json"

# Índice TOC (tabela de conteúdo dos manuais)
INDICE_TOC = Path(
    r"C:\caminho\para\chatdohangar_backend\manuais"
    r"\AMM B737 REV 84 DATE 25 SEPT 2016\737-345_RAM_AMM_D6-37569_TD"
    r"\amm_toc_database.json"
)

# Ground truth (gabarito de perguntas/respostas)
GROUND_TRUTH = BASE_DIR / "ground_truth.json"

# Perguntas de teste (uma por linha)
PERGUNTAS_TESTE = BASE_DIR / "perguntas_teste.txt"
```

> **⚠️ ATENÇÃO**: O caminho do `INDICE_TOC` aponta para uma pasta externa (`chatdohangar_backend`). Este caminho precisa existir e apontar para o `amm_toc_database.json`, caso contrário as abordagens que usam TOC falharão.

---

## 🚀 Como Executar o Projeto

### 🔹 PASSO 0: Validação Rápida (Teste de Sanidade)

Antes de rodar tudo, valide que o ambiente está funcionando:

```bash
cd env
python -c "import config; print('Config OK'); import comum; print('Comum OK')"
```

---

### 🔹 PASSO 1: Teste Rápido (5 perguntas)

Execute o pipeline completo com apenas 5 perguntas para validar o fluxo:

```bash
cd env
python pipeline_ragas.py --amostra 5
```

**O que faz:**
1. Carrega as 5 primeiras perguntas do `ground_truth_amm_400.json`
2. Executa a abordagem padrão (`toc_chunks`)
3. Avalia com métricas RAGAS
4. Gera relatórios

**Tempo estimado:** ~2-5 minutos

---

### 🔹 PASSO 2: Executar Todas as Abordagens (Avaliação Completa)

> **IMPORTANTE**: Para a avaliação RAGAS funcionar corretamente, você **DEVE** usar as perguntas do ground truth (`ground_truth_amm_400.json`). O RAGAS compara as respostas do sistema com as respostas corretas do gabarito.

```bash
cd env
python pipeline_ragas.py --amostra 400
```

**O que faz:**
1. Carrega as 400 perguntas do ground truth
2. Executa as 6 abordagens RAG
3. Avalia cada resposta com métricas RAGAS (LLM Judge)
4. Gera relatórios consolidados

**Tempo estimado:** **1-3 horas** (depende da velocidade da API NVIDIA)

---

### 🔹 Opções Avançadas do Pipeline

#### Executar apenas abordagens específicas:
```bash
# Só a melhor abordagem
python pipeline_ragas.py --abordagens toc_chunks --amostra 50

# Comparar duas abordagens
python pipeline_ragas.py --abordagens toc_chunks faiss_nvidia --amostra 100
```

#### Pular etapa RAG (reutilizar auditorias existentes):
```bash
# Se já executou o RAG, só avalia com RAGAS
python pipeline_ragas.py --skip-rag --amostra 400
```

#### Pular etapa RAGAS (só gerar respostas):
```bash
python pipeline_ragas.py --skip-ragas --amostra 400
```

---

### 🔹 Formas Alternativas de Execução

#### 1. Orquestrador Simples (métricas customizadas, sem RAGAS)
```bash
cd env

# Todas as abordagens com perguntas do arquivo
python orquestrador.py --perguntas perguntas_teste.txt

# Abordagens específicas
python orquestrador.py --abordagens tfidf faiss --perguntas perguntas_teste.txt
```

#### 2. Orquestrador RAGAS (avalia resultados existentes)
```bash
cd env

# Avalia todas as abordagens com auditoria existente
python orquestrador_ragas.py

# Abordagens específicas
python orquestrador_ragas.py --abordagens faiss_nvidia toc_chunks

# Limitar número de perguntas
python orquestrador_ragas.py --amostra 20
```

#### 3. Scripts de Processamento da Base
```bash
cd env

# Gerar chunks da base de conhecimento
python processar_base_chunks.py

# Construir índice FAISS NVIDIA
python build_nvidia_faiss_index.py

# Comparar bases de conhecimento
python comparar_bases.py
```

---

## 📊 Métricas de Avaliação

### Métricas RAGAS (framework padrão da indústria)

| Métrica | O que avalia | Faixa |
|---------|-------------|-------|
| **Faithfulness** | A resposta é fiel ao contexto recuperado? | 0-1 |
| **Answer Relevancy** | A resposta é relevante para a pergunta? | 0-1 |
| **Context Precision** | O contexto recuperado é relevante? | 0-1 |
| **Context Recall** | O contexto contém toda a informação do gabarito? | 0-1 |
| **Answer Correctness** | A resposta está correta vs gabarito? | 0-1 |
| **Answer Similarity** | Similaridade semântica com o gabarito? | 0-1 |

> **Como funciona**: Um **LLM Judge** (modelo Llama) analisa a pergunta, o contexto recuperado, a resposta gerada e o gabarito, atribuindo scores.

### Métricas Customizadas (em `comum.py`)

| Métrica | O que avalia |
|---------|-------------|
| **Precisão** | Proporção de palavras corretas na resposta |
| **Completude** | Densidade de informação relevante |
| **Especificidade** | Capacidade de citar valores/normas específicas |
| **Rastreabilidade** | Capacidade de indicar fonte/capítulo |
| **Confiança** | Grau de assertividade da resposta |

---

## 📁 Arquivos de Saída (Resultados)

Após a execução, os resultados são salvos em **`env/resultados/`**:

| Arquivo | Conteúdo |
|---------|----------|
| `resultados_{abordagem}.csv` | Resultados detalhados por pergunta |
| `auditoria_{abordagem}.json` | Auditoria completa (pergunta + contexto + resposta) |
| `comparativo_abordagens.csv` | Comparativo consolidado entre abordagens |
| `ragas_consolidado.json` | Métricas RAGAS consolidadas |
| `resumo_estatistico.txt` | Ranking e estatísticas resumidas |

### Exemplo do `resumo_estatistico.txt`:
```
🏆 RANKING DAS ABORDAGENS
1. toc_chunks      - Score: 0.88
2. faiss_nvidia    - Score: 0.82
3. faiss           - Score: 0.76
4. indice_toc      - Score: 0.71
5. tfidf           - Score: 0.65
6. llm_puro        - Score: 0.58
```

### Como interpretar a auditoria (`auditoria_toc_chunks.json`):
```json
{
  "pergunta": "Qual é o torque máximo para apertar a válvula de oxigênio?",
  "contexto_recuperado": "Seção 35-00-00: ...",
  "resposta_llm": "O torque máximo é de 25 lb-in...",
  "gabarito": "25 lb-in",
  "metricas": {
    "faithfulness": 0.95,
    "answer_correctness": 0.90
  }
}
```

---

## 🔄 Fluxo de Trabalho Completo

```
┌─────────────────────────────────────────────────────────────────┐
│  ETAPA 1: PREPARAÇÃO DA BASE (uma vez)                          │
│  - processar_base_chunks.py → base_chunks.json                  │
│  - build_nvidia_faiss_index.py → faiss_nvidia_store/            │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│  ETAPA 2: EXECUÇÃO RAG (6 abordagens processam as perguntas)    │
│  - Cada abordagem recupera contexto da base                     │
│  - LLM (Llama) gera resposta                                    │
│  - Guarda auditoria (JSON) e resultados (CSV)                   │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│  ETAPA 3: AVALIAÇÃO RAGAS                                       │
│  - LLM Judge compara resposta vs gabarito                       │
│  - Calcula 6 métricas RAGAS                                     │
│  - Gera ragas_consolidado.json                                  │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│  ETAPA 4: RELATÓRIOS                                            │
│  - comparativo_abordagens.csv                                   │
│  - resumo_estatistico.txt (ranking)                             │
└─────────────────────────────────────────────────────────────────┘
```

---

## ❓ Perguntas Frequentes (FAQ)

### 1. Preciso usar as perguntas do ground truth?
**SIM** — para a avaliação RAGAS completa, sim. O RAGAS precisa do gabarito para calcular `answer_correctness` e `context_recall`. Se você usar perguntas sem gabarito, só métricas de contexto (faithfulness, context_precision) serão calculadas.

### 2. Posso testar minhas próprias perguntas?
**SIM**, de duas formas:
- **Com gabarito**: Adicione as perguntas ao `ground_truth_amm_400.json` com a resposta correta
- **Sem gabarito**: Crie um arquivo `minhas_perguntas.txt` e use `python orquestrador.py --perguntas minhas_perguntas.txt`

### 3. Qual abordagem é a melhor?
A **TOC + Chunks** (`toc_chunks`) apresentou os melhores resultados nos testes realizados.

### 4. Quanto tempo demora?
- **5 perguntas** (teste): ~2-5 minutos
- **400 perguntas** (completo): 1-3 horas

### 5. Preciso pagar pela API?
A API NVIDIA tem **créditos gratuitos de teste** para novos usuários. Para volume grande, pode ser necessário crédito pago.

### 6. Posso mudar o modelo LLM?
**SIM** — edite `LLM_MODEL` no `config.py`. O importante é usar o **mesmo modelo** em todas as abordagens para comparação justa.

### 7. O projeto usa PyTorch?
**NÃO** — o projeto evita PyTorch (por compatibilidade com Python 3.14) e usa scikit-learn + FAISS-CPU.

### 8. O que é o índice TOC?
É a **tabela de conteúdo** dos manuais (4.945 entradas organizadas por capítulos ATA). Ajuda a navegar pela estrutura hierárquica do manual.

### 9. O que são os "chunks"?
São **trechos de texto** (800 caracteres) extraídos dos PDFs do AMM. Total: 92.310 chunks. Cada abordagem busca nos chunks para encontrar informação relevante.

### 10. Como sei se a API está funcionando?
```bash
python -c "import config; import comum; print('OK')"
```

---

## 🛠️ Solução de Problemas (Troubleshooting)

### ❌ Erro: "Nenhuma pergunta encontrada"
**Causa**: Arquivo de perguntas vazio ou ausente.
**Solução**: Crie `perguntas_teste.txt` ou use `--perguntas <arquivo>`.

### ❌ Erro: "ground_truth_amm_400.json não encontrado"
**Causa**: Arquivo do gabarito não está no diretório.
**Solução**: Verifique se o arquivo está em `env/ground_truth_amm_400.json`.

### ❌ Erro de autenticação NVIDIA (401)
**Causa**: Chave da API inválida ou expirada.
**Solução**: Verifique `NVIDIA_API_KEY` no `config.py`.

### ❌ Erro ao carregar o índice TOC
**Causa**: Caminho do `amm_toc_database.json` incorreto.
**Solução**: Verifique o `INDICE_TOC` no `config.py` e ajuste o caminho.

### ❌ Erro de timeout / lentidão
**Causa**: Muitas perguntas de uma vez.
**Solução**: Use `--amostra 5` para testar, depois aumente gradualmente.

### ❌ Erro de import (faiss, sklearn, etc.)
**Causa**: Dependências não instaladas.
**Solução**: `pip install requests faiss-cpu scikit-learn numpy python-dotenv`

---

## 🔧 Dependências

```txt
requests          # Chamadas à API NVIDIA
faiss-cpu         # Indexação vetorial FAISS
scikit-learn      # TF-IDF e métricas
numpy             # Operações numéricas
python-dotenv     # Variáveis de ambiente (opcional)
```

---

## 📖 Sobre a Base de Conhecimento

- **Fonte**: Manual de Manutenção Boeing 737-300/400/500 (AMM)
- **Documentos**: PDFs nomeados por capítulo (ex: `06___084.PDF`, `35___084.PDF`)
- **Chunks**: 92.310 trechos de 800 caracteres cada
- **Índice TOC**: 4.945 entradas organizadas por capítulos ATA
- **Ground Truth**: 400 perguntas técnicas com respostas verificadas

---

## 🤝 Contribuindo

Sinta-se à vontade para:
1. **Adicionar novas abordagens** de recuperação
2. **Melhorar o ground truth** com mais perguntas
3. **Otimizar parâmetros** (TOP_K, CHUNK_CHARS, TEMPERATURA)
4. **Testar outros modelos LLM**
5. **Aplicar a outros manuais** técnicos

---

## 📧 Contato

Para dúvidas ou sugestões, abra uma **issue** no repositório.

---

*Desenvolvido para fins acadêmicos e de pesquisa em IA aplicada à manutenção aeronáutica.*