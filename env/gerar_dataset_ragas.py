#!/usr/bin/env python3
"""
================================================================================
GERADOR EM LARGA ESCALA DE DATASET DE TESTE PARA O RAGAS (COM DEEPSEEK VIA NVIDIA)
================================================================================
Percorre a base de conhecimento de forma sistemática e usa o DeepSeek 
(via API da NVIDIA) para gerar um grande volume de pares (pergunta/resposta).
================================================================================
"""

import argparse
import getpass
import json
import os
import pathlib
import time
from openai import OpenAI

import config
import comum

# Modelo DeepSeek otimizado para raciocínio via NVIDIA NIM
MODELO_GERADOR = "deepseek-ai/deepseek-v4-flash-0731"

def main():
    parser = argparse.ArgumentParser(description="Gerador em Larga Escala de Dataset para o RAGAS")
    parser.add_argument("--num", type=int, default=400, help="Número total de perguntas a serem geradas (padrão: 400)")
    args = parser.parse_args()

    meta_perguntas = args.num
    print(f"🚀 Iniciando gerador em larga escala. Meta: {meta_perguntas} perguntas...")

    # Configuração da API Key
    api_key = getattr(config, "NVIDIA_API_KEY", None) or os.getenv("NVIDIA_API_KEY")
    if not api_key:
        print("\n🔑 NVIDIA_API_KEY não encontrada. Insira abaixo:")
        api_key = getpass.getpass("Cole sua NVIDIA API Key (o texto ficará oculto): ").strip()
        if not api_key:
            raise ValueError("❌ Nenhuma chave foi fornecida. O processo foi cancelado.")

    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=api_key
    )

    base_path = pathlib.Path(__file__).parent / "base_conhecimento.json"
    if not base_path.exists():
        raise FileNotFoundError("❌ base_conhecimento.json não encontrada na pasta atual!")

    with open(base_path, 'r', encoding='utf-8') as f:
        documentos = json.load(f)

    print(f"📚 Base carregada com {len(documentos)} documentos disponíveis.")
    
    dataset_teste = []
    output_file = pathlib.Path("dataset_teste_ragas_400.json")

    # Retoma de onde parou caso o arquivo já exista parcialmente
    if output_file.exists():
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                dataset_teste = json.load(f)
            print(f"📂 Arquivo anterior encontrado com {len(dataset_teste)} perguntas já geradas. Continuando...")
        except:
            pass

    doc_idx = 0
    total_docs = len(documentos)

    while len(dataset_teste) < meta_perguntas and doc_idx < total_docs:
        doc = documentos[doc_idx]
        doc_idx += 1

        arquivo = doc.get("arquivo", "desconhecido")
        conteudo = doc.get("conteudo_completo", "") or doc.get("conteudo_preview", "")
        
        if len(conteudo.strip()) < 300:
            continue

        trecho = conteudo[10000:18000] if len(conteudo) > 18000 else conteudo[:5000]

        print(f"🤖 [{len(dataset_teste) + 1}/{meta_perguntas}] Processando arquivo: {arquivo}...")

        prompt = f"""Você é um engenheiro aeronáutico especialista em manutenção de aeronaves. Com base estritamente no trecho do manual AMM abaixo, crie UMA pergunta técnica desafiadora, específica e realista que um técnico faria sobre este subsistema, e forneça a resposta exata baseada APENAS no texto.

TRECHO DO MANUAL:
{trecho}

Retorne estritamente no formato JSON puro, sem markdown, contendo exatamente estas duas chaves:
{{"pergunta": "...", "resposta_esperada": "..."}}"""

        try:
            completion = client.chat.completions.create(
                model=MODELO_GERADOR,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6,
                top_p=0.95,
                max_tokens=4096,
                extra_body={
                    "chat_template_kwargs": {
                        "thinking": True,
                        "reasoning_effort": "high"
                    }
                },
                stream=False
            )
            
            conteudo_resposta = completion.choices[0].message.content.strip()
            # Limpa eventuais blocos de código markdown gerados pelo modelo
            conteudo_resposta = conteudo_resposta.replace("```json", "").replace("```", "").strip()
            
            # Garante extração segura do JSON caso venha texto adicional
            inicio_json = conteudo_resposta.find("{")
            fim_json = conteudo_resposta.rfind("}")
            if inicio_json != -1 and fim_json != -1:
                conteudo_resposta = conteudo_resposta[inicio_json:fim_json+1]

            dados_qa = json.loads(conteudo_resposta)
            
            dataset_teste.append({
                "question": dados_qa["pergunta"],
                "ground_truth": dados_qa["resposta_esperada"],
                "source_file": arquivo
            })
            print(f"  ✓ Sucesso! Pergunta: {dados_qa['pergunta'][:50]}...")

            # Salvamento incremental a cada 10 perguntas
            if len(dataset_teste) % 10 == 0:
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(dataset_teste, f, ensure_ascii=False, indent=2)
                print(f"💾 Progresso salvo temporariamente ({len(dataset_teste)} perguntas)...")

            time.sleep(0.5)
            
        except Exception as e:
            print(f"  ⚠️ Erro ao processar o documento {arquivo}: {e}")
            time.sleep(1)

    # Salvamento final completo
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dataset_teste, f, ensure_ascii=False, indent=2)

    print(f"\n🎉 Geração concluída!")
    print(f"✅ Total de perguntas geradas e salvas em '{output_file}': {len(dataset_teste)}")

if __name__ == "__main__":
    main()