import json
import os
import pathlib
import time
from openai import OpenAI

# 1. Importa a chave do seu arquivo config.py local
try:
    from config import NVIDIA_API_KEY  # Altere para o nome da variável no seu config.py (ex: OPENAI_API_KEY ou GEMINI_API_KEY)
except ImportError:
    print("❌ Erro: Arquivo config.py não encontrado na pasta atual.")
    exit(1)

# Inicializa o cliente NVIDIA usando a chave do config.py
client = OpenAI(
    api_key=NVIDIA_API_KEY,
    base_url="https://integrate.api.nvidia.com/v1"
)

# Modelo NVIDIA válido (mesma API key funciona para todos)
MODELO_GERADOR = "openai/gpt-oss-20b"
META_TOTAL = 400
PERGUNTAS_POR_DOC = 10

# Arquivos na mesma pasta do projeto
ARQUIVO_BASE_CONHECIMENTO = "base_conhecimento.json" # Nome do seu JSON local
ARQUIVO_SAIDA = pathlib.Path("ground_truth_amm_400.json")

# 2. Prompt Estruturado
PROMPT_SISTEMA = """Você é um engenheiro aeronáutico especialista em manutenção de aeronaves (Manual AMM).
Sua tarefa é criar perguntas técnicas extremamente detalhadas, específicas e realistas baseadas APENAS no trecho do manual fornecido.

Evite perguntas genéricas do tipo 'Qual o número da task que faz tal coisa?'. 
Foque obrigatoriamente em aspectos operacionais e físicos como:
- Localização e quantidades: 'Onde fica localizado...', 'Quantos ... estão instalados na aeronave', 'De que é feito...';
- Valores numéricos e limites: 'Qual é o torque dos parafusos de fixação...', 'Qual é a medida mínima...', 'Qual a altura máxima permitida...', 'Qual a pressão mínima/máxima...', 'Qual a velocidade mínima...';
- Fluidos e materiais: 'Qual a quantidade de óleo...', 'Qual é o selante utilizado no...';
- Funcionalidade e falhas: 'Para que serve o sistema...', 'Se houver uma falha o que acontece com o sistema...', 'Quando a luz de warning X é acionada, o que deve ter falhado?', 'Existe acionamento manual em falha?';
- Procedimentos práticos: 'Para instalar o componente X, qual a preparação necessária?', 'Após instalado, quais testes são necessários?', 'O que verificar na inspeção de...', 'O botão Y serve para quê?'.

Gere as perguntas e respostas estritamente baseadas nas informações presentes no texto."""

def carregar_base_json(caminho_json):
    try:
        with open(caminho_json, "r", encoding="utf-8") as f:
            dados = json.load(f)
        print(f"📚 Base de conhecimento carregada com sucesso! Total de itens: {len(dados)}")
        return dados
    except Exception as e:
        print(f"❌ Erro ao carregar a base JSON '{caminho_json}': {e}")
        return []

def carregar_dados_existentes():
    if ARQUIVO_SAIDA.exists():
        try:
            with open(ARQUIVO_SAIDA, 'r', encoding='utf-8') as f:
                dados = json.load(f)
            print(f"📂 Encontrado arquivo anterior com {len(dados)} perguntas salvas. Retomando...")
            return dados
        except Exception as e:
            print(f"⚠️ Erro ao ler arquivo existente: {e}")
    return []

def salvar_dados(dataset):
    with open(ARQUIVO_SAIDA, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

def processar_ground_truth(base_conhecimento):
    dataset_teste = carregar_dados_existentes()
    doc_idx = 0
    total_docs = len(base_conhecimento)

    print(f"\n🚀 Iniciando geração. Meta: {META_TOTAL} perguntas ({PERGUNTAS_POR_DOC} por item)...")

    while len(dataset_teste) < META_TOTAL and doc_idx < total_docs:
        doc = base_conhecimento[doc_idx]
        doc_idx += 1

        arquivo = doc.get("arquivo") or doc.get("source") or doc.get("id") or f"item_{doc_idx}"
        conteudo = doc.get("conteudo") or doc.get("texto") or doc.get("conteudo_completo") or ""

        if len(conteudo.strip()) < 300:
            continue

        trecho = conteudo[:25000] if len(conteudo) > 25000 else conteudo

        prompt_usuario = f"""
Com base no trecho do manual AMM abaixo, crie EXATAMENTE {PERGUNTAS_POR_DOC} perguntas técnicas variadas e detalhadas seguindo as diretrizes fornecidas.

TRECHO DO MANUAL:
{trecho}

Retorne ESTRITAMENTE um array JSON com os objetos contendo as chaves 'pergunta' e 'resposta_esperada':
[
  {{"pergunta": "...", "resposta_esperada": "..."}}
]
"""

        try:
            response = client.chat.completions.create(
                model=MODELO_GERADOR,
                messages=[
                    {"role": "system", "content": PROMPT_SISTEMA},
                    {"role": "user", "content": prompt_usuario}
                ],
                temperature=0.4,
                max_tokens=4096
            )

            conteudo_resposta = response.choices[0].message.content.strip()
            
            # Remover markdown code blocks se existirem
            conteudo_resposta = conteudo_resposta.replace("```json", "").replace("```", "").strip()
            
            # Encontrar o array JSON
            inicio_json = conteudo_resposta.find("[")
            fim_json = conteudo_resposta.rfind("]")
            if inicio_json != -1 and fim_json != -1:
                conteudo_resposta = conteudo_resposta[inicio_json:fim_json+1]
            
            # Tentar corrigir problemas comuns de JSON malformado
            # Remove quebras de linha dentro de strings
            import re
            conteudo_resposta = re.sub(r'\n\s*', ' ', conteudo_resposta)
            
            # Tenta fazer parse, se falhar tenta corrigir
            try:
                perguntas_geradas = json.loads(conteudo_resposta)
            except json.JSONDecodeError as je:
                print(f"  ⚠️ JSON malformado, tentando corrigir...")
                # Tenta extrair objetos individuais
                try:
                    # Busca por objetos {...} individuais
                    objetos = re.findall(r'\{[^{}]*"pergunta"[^{}]*\}', conteudo_resposta, re.DOTALL)
                    if objetos:
                        perguntas_geradas = [json.loads(obj) for obj in objetos]
                        print(f"  ✅ Recuperados {len(perguntas_geradas)} objetos parciais")
                    else:
                        raise je
                except:
                    print(f"  ❌ Não foi possível recuperar o JSON. Pulando documento.")
                    continue

            if isinstance(perguntas_geradas, list):
                novas_adicionadas = 0
                for item in perguntas_geradas:
                    if len(dataset_teste) >= META_TOTAL:
                        break
                    dataset_teste.append({
                        "question": item["pergunta"],
                        "ground_truth": item["resposta_esperada"],
                        "source_file": arquivo
                    })
                    novas_adicionadas += 1

                print(f"[{len(dataset_teste)}/{META_TOTAL}] ✓ {novas_adicionadas} perguntas geradas de: {arquivo}")
                salvar_dados(dataset_teste)

            time.sleep(0.5)

        except Exception as e:
            print(f"⚠️ Erro ao processar o item {arquivo}: {e}")
            time.sleep(1)

    print(f"\n🎉 Processo concluído! Total de {len(dataset_teste)} perguntas salvas em '{ARQUIVO_SAIDA}'.")

if __name__ == "__main__":
    base = carregar_base_json(ARQUIVO_BASE_CONHECIMENTO)
    if base:
        processar_ground_truth(base)
    else:
        print("❌ Não foi possível carregar os dados do arquivo JSON.")