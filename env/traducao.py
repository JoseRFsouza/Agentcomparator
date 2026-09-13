#!/usr/bin/env python3
"""
================================================================================
TRADUÇÃO TÉCNICA (PT-BR -> EN)
================================================================================
Dicionário de mapeamento de termos técnicos de aviação do português para o
inglês, para permitir a busca em manuais escritos em inglês a partir de
perguntas em português.

Usado por todas as abordagens para normalizar a consulta.
================================================================================
"""

# Mapeamento de termos técnicos (pt-br -> en)
DICIONARIO = {
    # Sistemas da aeronave
    "hidraulico": "hydraulic",
    "hidraulica": "hydraulic",
    "hidraulicos": "hydraulic",
    "hidraulicas": "hydraulic",
    "trem de pouso": "landing gear",
    "pouso": "landing",
    "controle de voo": "flight control",
    "comando de voo": "flight control",
    "pressurizacao": "pressurization",
    "ar condicionado": "air conditioning",
    "combustivel": "fuel",
    "motor": "engine",
    "eletrico": "electrical",
    "eletrica": "electrical",
    "pneumatico": "pneumatic",
    "pneumatica": "pneumatic",
    "freio": "brake",
    "freios": "brakes",
    "navegacao": "navigation",
    "comunicacao": "communication",
    "comunicacoes": "communications",
    "iluminacao": "lighting",
    "luz": "light",
    "porta": "door",
    "portas": "doors",
    "janela": "window",
    "janelas": "windows",
    "fuselagem": "fuselage",
    "asa": "wing",
    "asas": "wings",
    "estabilizador": "stabilizer",
    "leme": "rudder",
    "aileron": "aileron",
    "flap": "flap",
    "flaps": "flaps",
    "spoiler": "spoiler",
    "spoilers": "spoilers",
    "computador": "computer",
    "computadores": "computers",
    "bomba": "pump",
    "bombas": "pumps",
    "valvula": "valve",
    "valvulas": "valves",
    "sensor": "sensor",
    "sensores": "sensors",
    "atuador": "actuator",
    "atuadores": "actuators",
    "reservatorio": "reservoir",
    "tanque": "tank",
    "tanques": "tanks",
    "tubulacao": "tubing",
    "cabo": "cable",
    "cabos": "cables",
    "fiacao": "wiring",
    # Ações de manutenção
    "inspecao": "inspection",
    "inspecionar": "inspect",
    "verificacao": "check",
    "verificar": "check",
    "manutencao": "maintenance",
    "manutencao": "maintenance",
    "reparo": "repair",
    "reparar": "repair",
    "substituicao": "replacement",
    "substituir": "replace",
    "ajuste": "adjustment",
    "ajustar": "adjust",
    "teste": "test",
    "testar": "test",
    "lubrificacao": "lubrication",
    "lubrificar": "lubricate",
    "limpeza": "cleaning",
    "limpar": "clean",
    "remocao": "removal",
    "remover": "remove",
    "instalacao": "installation",
    "instalar": "install",
    # Termos gerais
    "procedimento": "procedure",
    "procedimentos": "procedures",
    "sistema": "system",
    "sistemas": "systems",
    "falha": "failure",
    "falhas": "failures",
    "emergencia": "emergency",
    "pressao": "pressure",
    "temperatura": "temperature",
    "vazamento": "leak",
    "vazamentos": "leaks",
    "torque": "torque",
    "parafuso": "bolt",
    "parafusos": "bolts",
    "porca": "nut",
    "porcas": "nuts",
    "arruela": "washer",
    "arruelas": "washers",
    "quantos": "how many",
    "quantas": "how many",
    "qual": "what",
    "quais": "which",
    "como": "how",
    "onde": "where",
    "quando": "when",
    "tem": "has",
    "existe": "exists",
    "existem": "exist",
    "numero": "number",
    "aeronave": "aircraft",
    "aviao": "aircraft",
}


def traduzir(texto: str) -> str:
    """
    Traduz termos técnicos conhecidos do português para o inglês.
    Substitui apenas os termos do dicionário (não faz tradução completa).
    """
    import re

    resultado = texto.lower()
    # Ordenar por tamanho decrescente para substituir termos compostos primeiro
    termos = sorted(DICIONARIO.keys(), key=len, reverse=True)
    for termo in termos:
        # Substituir palavra inteira (com limite de palavra)
        resultado = re.sub(
            r"\b" + re.escape(termo) + r"\b",
            DICIONARIO[termo],
            resultado,
        )
    return resultado


def expandir_consulta(texto: str) -> str:
    """
    Retorna o texto original + a versão traduzida, para maximizar a
    correspondência com o conteúdo (que pode estar em PT ou EN).
    """
    traduzido = traduzir(texto)
    if traduzido == texto.lower():
        return texto
    return f"{texto} {traduzido}"