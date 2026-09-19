#!/usr/bin/env python3
"""
================================================================================
GERADOR DE DADOS DE TREINO PARA FINE-TUNING
================================================================================
Gera pares (pergunta, resposta) a partir dos chunks dos manuais Boeing 737.
Usa o LLM para criar perguntas e respostas baseadas no conteúdo real.

Uso:
    python gerar_dados_treino.py
================================================================================
"""

import json
import random
import re
from pathlib import Path
from typing import List, Dict

import comum
import config


def extrair_informacoes_tecnicas(texto: str) -> List[Dict]:
    """
    Extrai informações técnicas do texto que podem virar perguntas.
    """
    info = []
    
    # Padrões de informação técnica
    padroes = [
        (r"(\d+)\s*(?:sistemas?|systems?)", "quantidade_sistemas"),
        (r"(\d+)\s*psi", "pressao_psi"),
        (r"ATA\s*(\d+)", "ata"),
        (r"(\d+)[-_](\d+)[-_](\d+)", "task_number"),
        (r"([Ff]unctionally\s+independent)", "caracteristica"),
        (r"([Tt]wo|[Tt]hree|[Ff]our|\d+)\s+([Ff]light\s+[Cc]ontrol\s+[Cc]omputers?)", "quantidade_fcc"),
    ]
    
    for padrao, tipo in padroes:
        matches = re.findall(padrao, texto, re.IGNORECASE)
        if matches:
            info.append({
                "tipo": tipo,
                "valor": matches[0] if len(matches) == 1 else matches,
                "contexto": texto[:500]
            })
    
    return info


def gerar_pergunta_resposta(chunk: Dict) -> List[Dict]:
    """
    Gera pares pergunta/resposta a partir de um chunk.
    """
    texto = chunk["texto"]
    arquivo = chunk["arquivo"]
    
    # Extrair ATA do nome do arquivo
    ata_match = re.match(r"(\d+)", arquivo)
    ata = ata_match.group(1) if ata_match else "??"
    
    pares = []
    
    # Perguntas baseadas em padrões comuns
    templates = [
        {
            "pergunta": f"O que diz o ATA {ata} sobre {extrair_topico(texto)}?",
            "resposta": texto[:600],
            "tipo": "factual"
        },
        {
            "pergunta": f"Qual é a informação técnica principal do documento {arquivo}?",
            "resposta": extrair_resumo(texto),
            "tipo": "resumo"
        },
        {
            "pergunta": f"Como é descrito o sistema no ATA {ata}?",
            "resposta": texto[:400],
            "tipo": "descritivo"
        }
    ]
    
    # Adicionar perguntas específicas baseadas no conteúdo
    info_tecnica = extrair_informacoes_tecnicas(texto)
    for info in info_tecnica:
        if info["tipo"] == "quantidade_sistemas":
            num = info["valor"]
            pares.append({
                "pergunta": f"Quantos sistemas são mencionados no ATA {ata}?",
                "resposta": f"De acordo com o ATA {ata}, são mencionados {num} sistemas.",
                "tipo": "factual",
                "fonte": arquivo
            })
        elif info["tipo"] == "pressao_psi":
            pares.append({
                "pergunta": f"Qual é a pressão mencionada no ATA {ata}?",
                "resposta": f"A pressão mencionada é de {info['valor']} psi.",
                "tipo": "factual",
                "fonte": arquivo
            })
    
    # Adicionar templates genéricos
    for t in templates:
        if len(t["resposta"].strip()) > 50:  # Só se tiver conteúdo significativo
            pares.append({
                "pergunta": t["pergunta"],
                "resposta": t["resposta"],
                "tipo": t["tipo"],
                "fonte": arquivo
            })
    
    return pares


def extrair_topico(texto: str) -> str:
    """Extrai o tópico principal do texto."""
    # Procurar por palavras-chave técnicas
    topicos = [
        "hydraulic", "landing gear", "flight control", "fuel", "engine",
        "electrical", "pneumatic", "navigation", "communication"
    ]
    texto_lower = texto.lower()
    for t in topicos:
        if t in texto_lower:
            return t
    return "o sistema"


def extrair_resumo(texto: str) -> str:
    """Extrai as primeiras frases significativas."""
    frases = re.split(r'[.!?]+', texto)
    significativas = [f.strip() for f in frases if len(f.strip()) > 30]
    return ". ".join(significativas[:3]) + "."


def gerar_dados_treino():
    """Gera dados de treino a partir dos chunks."""
    print("📂 Carregando chunks...")
    
    caminho = config.BASE_DIR / "base_chunks.json"
    with open(caminho, "r", encoding="utf-8") as f:
        dados = json.load(f)
    
    chunks = dados["chunks"]
    print(f"✓ {len(chunks)} chunks carregados")
    
    # Selecionar chunks de alta qualidade (maior densidade de informação)
    chunks_selecionados = []
    for chunk in chunks:
        texto = chunk["texto"]
        # Filtrar chunks com muito metadado
        if len(texto) < 200:
            continue
        # Calcular densidade: palavras únicas / total
        palavras = texto.split()
        if len(palavras) < 20:
            continue
        densidade = len(set(palavras)) / len(palavras)
        if densidade > 0.5:  # Alta densidade de informação
            chunks_selecionados.append(chunk)
    
    print(f"✓ {len(chunks_selecionados)} chunks selecionados (alta qualidade)")
    
    # Gerar pares
    todos_pares = []
    for chunk in chunks_selecionados[:5000]:  # Limitar para não demorar muito
        pares = gerar_pergunta_resposta(chunk)
        todos_pares.extend(pares)
    
    print(f"✓ {len(todos_pares)} pares gerados")
    
    # Adicionar exemplos de "não sei" (importante para o modelo aprender a recusar)
    exemplos_recusa = [
        {
            "pergunta": "Qual é a cor da pintura do Boeing 737?",
            "resposta": "Esta informação não está disponível nos manuais fornecidos.",
            "tipo": "recusa",
            "fonte": "n/a"
        },
        {
            "pergunta": "Quem fabricou os assentos do Boeing 737?",
            "resposta": "Esta informação não está disponível nos manuais fornecidos.",
            "tipo": "recusa",
            "fonte": "n/a"
        },
        {
            "pergunta": "Qual é o preço de um Boeing 737?",
            "resposta": "Esta informação não está disponível nos manuais fornecidos.",
            "tipo": "recusa",
            "fonte": "n/a"
        }
    ]
    todos_pares.extend(exemplos_recusa * 10)  # Repetir para balancear
    
    # Salvar
    saida = config.BASE_DIR / "dados_treino.json"
    with open(saida, "w", encoding="utf-8") as f:
        json.dump(todos_pares, f, ensure_ascii=False, indent=2)
    
    print(f"💾 Dados salvos em: {saida}")
    print(f"📊 Total: {len(todos_pares)} exemplos")
    
    # Estatísticas
    tipos = {}
    for p in todos_pares:
        tipos[p["tipo"]] = tipos.get(p["tipo"], 0) + 1
    
    print("\n📊 Distribuição por tipo:")
    for tipo, count in sorted(tipos.items()):
        print(f"  • {tipo}: {count}")
    
    return todos_pares


if __name__ == "__main__":
    gerar_dados_treino()
