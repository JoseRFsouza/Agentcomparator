#!/usr/bin/env python3
"""
================================================================================
MÓDULO COMUM - Funções compartilhadas por todas as abordagens
================================================================================
- Carregamento da base de conhecimento e índice TOC
- Geração de resposta via NVIDIA API (LLM)
- Métricas de avaliação (precisão, completude, especificidade, rastreabilidade, confiança)
- Auditoria (registro estruturado de cada pergunta)
- Geração de relatórios CSV
================================================================================
"""

import json
import time
import csv
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import requests

import config


# ============================================================================
# 1. CARREGAMENTO DE DADOS
# ============================================================================

def carregar_base_conhecimento() -> List[Dict]:
    """Carrega a base de conhecimento processada."""
    with open(config.INDICE_CONHECIMENTO, "r", encoding="utf-8") as f:
        dados = json.load(f)
    return dados.get("documentos", [])


def carregar_indice_toc() -> List[Dict]:
    """Carrega o índice TOC dos manuais (tabela de conteúdo)."""
    with open(config.INDICE_TOC, "r", encoding="utf-8") as f:
        return json.load(f)


def carregar_ground_truth() -> List[Dict]:
    """Carrega o ground truth (gabarito), se existir."""
    if not config.GROUND_TRUTH.exists():
        return []
    with open(config.GROUND_TRUTH, "r", encoding="utf-8") as f:
        return json.load(f)


def carregar_perguntas(arquivo: Optional[Path] = None) -> List[str]:
    """Carrega a lista de perguntas de teste (uma por linha)."""
    caminho = arquivo or config.PERGUNTAS_TESTE
    if isinstance(caminho, str):
        caminho = Path(caminho)
    if not caminho.exists():
        return []
    with open(caminho, "r", encoding="utf-8") as f:
        perguntas = [linha.strip() for linha in f if linha.strip()]
    return perguntas


# ============================================================================
# 2. GERAÇÃO DE RESPOSTA (LLM - NVIDIA API)
# ============================================================================

def gerar_resposta_llm(
    prompt_usuario: str,
    contexto: str = "",
    system: str = "",
    modelo: str = config.LLM_MODEL,
) -> Tuple[str, float, Dict]:
    """
    Gera resposta usando a NVIDIA API.

    Retorna (texto_resposta, tempo_segundos, metadados).
    """
    if not system:
        system = (
            "Você é um assistente técnico especializado em manuais de "
            "manutenção de aeronaves Boeing 737. "
            "REGRAS OBRIGATÓRIAS:\n"
            "1. Responda APENAS com base no contexto fornecido.\n"
            "2. Sempre cite a fonte (ex: 'De acordo com o manual ATA 29...' "
            "ou 'Conforme o capítulo 32 do AMM...').\n"
            "3. Inclua TODOS os valores numéricos e detalhes técnicos "
            "disponíveis no contexto (pressão em psi, quantidades, "
            "códigos de task, nomes de componentes).\n"
            "4. Estruture a resposta em 2-3 parágrafos: resposta direta, "
            "detalhes técnicos, e referência à fonte.\n"
            "5. Use o máximo de informação relevante do contexto — "
            "respostas detalhadas são melhores que respostas curtas.\n"
            "6. Se o contexto não contém a informação, diga claramente."
        )

    mensagens = [{"role": "system", "content": system}]

    if contexto:
        mensagens.append(
            {
                "role": "user",
                "content": f"CONTEXTO DOS MANUAIS:\n{contexto}\n\nPERGUNTA: {prompt_usuario}",
            }
        )
    else:
        mensagens.append({"role": "user", "content": prompt_usuario})

    payload = {
        "model": modelo,
        "messages": mensagens,
        "temperature": config.TEMPERATURA,
        "max_tokens": config.MAX_TOKENS,
    }

    headers = {
        "Authorization": f"Bearer {config.NVIDIA_API_KEY}",
        "Content-Type": "application/json",
    }

    inicio = time.time()
    try:
        resp = requests.post(
            config.NVIDIA_CHAT_URL, headers=headers, json=payload, timeout=60
        )
        tempo = time.time() - inicio

        if resp.status_code == 200:
            data = resp.json()
            texto = data["choices"][0]["message"]["content"].strip()
            meta = {
                "status": "ok",
                "tokens": data.get("usage", {}),
                "finish_reason": data["choices"][0].get("finish_reason"),
            }
            return texto, tempo, meta
        else:
            texto = f"[ERRO {resp.status_code}] {resp.text[:300]}"
            meta = {"status": "erro", "http_status": resp.status_code}
            return texto, tempo, meta

    except requests.exceptions.Timeout:
        return "[TIMEOUT] A API demorou mais de 60s.", time.time() - inicio, {"status": "timeout"}
    except Exception as e:
        return f"[ERRO] {str(e)}", time.time() - inicio, {"status": "erro", "erro": str(e)}


# ============================================================================
# 3. MÉTRICAS DE AVALIAÇÃO
# ============================================================================

def _normalizar(texto: str) -> str:
    """Normaliza texto para comparação (minúsculas, sem acentos)."""
    texto = texto.lower()
    import unicodedata
    texto = unicodedata.normalize("NFKD", texto)
    texto = texto.encode("ascii", "ignore").decode("ascii")
    return texto


def metrica_precisao(resposta: str, contexto: str) -> float:
    """
    Precisão: fração das palavras-chave da RESPOSTA que aparecem no contexto.
    Isso mede se a resposta é fiel ao contexto recuperado (não inventa).
    """
    if not contexto or not resposta:
        return 0.0
    ctx_norm = _normalizar(contexto)
    resp_norm = _normalizar(resposta)

    # Palavras significativas da RESPOSTA (>= 4 chars)
    palavras_resp = set(re.findall(r"\w{4,}", resp_norm))
    if not palavras_resp:
        return 0.0

    palavras_ctx = set(re.findall(r"\w{4,}", ctx_norm))
    intersecao = palavras_resp & palavras_ctx
    return len(intersecao) / len(palavras_resp)


def metrica_completude(resposta: str, pergunta: str, contexto: str) -> float:
    """
    Completude: quão completa é a resposta.
    Combina: cobertura de palavras da pergunta + densidade de informação.
    """
    if not contexto or not resposta:
        return 0.0
    resp_norm = _normalizar(resposta)
    perg_norm = _normalizar(pergunta)

    # Palavras da pergunta presentes na resposta
    palavras_perg = set(re.findall(r"\w{4,}", perg_norm))
    palavras_resp = set(re.findall(r"\w{4,}", resp_norm))
    if not palavras_perg:
        cobertura_perg = 0.0
    else:
        cobertura_perg = len(palavras_perg & palavras_resp) / len(palavras_perg)

    # Densidade de informação: palavras únicas / total de palavras
    # Respostas densas em informação (não repetitivas) pontuam mais
    total_palavras = len(re.findall(r"\w+", resp_norm))
    palavras_unicas = len(set(re.findall(r"\w+", resp_norm)))
    if total_palavras == 0:
        densidade = 0.0
    else:
        densidade = min(1.0, palavras_unicas / max(total_palavras, 1) * 2.0)

    # Tamanho mínimo razoável (respostas muito curtas penalizadas)
    tamanho_score = min(1.0, total_palavras / 50.0)

    return 0.4 * cobertura_perg + 0.3 * densidade + 0.3 * tamanho_score


def metrica_especificidade(resposta: str) -> float:
    """
    Especificidade: presença de números, códigos, unidades técnicas.
    """
    padroes = [
        r"\d+\s*(psi|bar|pa|lb|kg|mm|cm|m\b|ft|nm|kts|kt|deg|percent)",  # unidades
        r"\d+[-/]\d+[-/]\d+",          # task number 32-11-00
        r"737[- ]\d+",                  # 737-300, 737-800
        r"ata\s*\d+",                   # ATA 29
        r"\d{3,4}\s*psi",               # pressão típica
        r"\b\d{2,4}\b",                 # números significativos
        r"chapter\s*\d+",
        r"amm\s*\d+",                   # referência AMM
        r"[a-f]\s+e\s+[a-f]\b",        # "A e B" (designação de sistemas)
    ]
    resp_norm = _normalizar(resposta)
    total = len(padroes)
    achados = sum(1 for p in padroes if re.search(p, resp_norm))
    return min(1.0, achados / 3.0)  # 3 padrões = máximo


def metrica_rastreabilidade(resposta: str) -> float:
    """
    Rastreabilidade: a resposta indica a fonte/documento?
    """
    marcadores = [
        "fonte", "documento", "manual", "capitulo", "chapter",
        "ata", ".pdf", "pagina", "task", "section",
    ]
    resp_norm = _normalizar(resposta)
    achados = sum(1 for m in marcadores if m in resp_norm)
    return min(1.0, achados / 3.0)  # 3 marcadores = rastreabilidade máxima


def metrica_confianca(resposta: str) -> float:
    """
    Confiança: 1.0 se a resposta é assertiva (baseada no contexto),
    0.5 se admite incerteza parcial, 0.0 se não soube responder.
    """
    incerteza_total = [
        "nao encontrei", "nao encontrado", "nao sei",
        "nao ha informacao", "sem informacao", "desconhecido",
        "nao foi possivel", "nao e possivel determinar",
    ]
    incerteza_parcial = [
        "nao tenho", "nao esta", "nao menciona", "nao especifica",
        "no entanto", "porem", "embora",
    ]
    resp_norm = _normalizar(resposta)
    
    # Verificar incerteza total (resposta inútil)
    for m in incerteza_total:
        if m in resp_norm:
            return 0.0
    
    # Incerteza parcial (respondeu mas com ressalvas)
    for m in incerteza_parcial:
        if m in resp_norm:
            return 0.5
    
    # Resposta assertiva
    return 1.0


def calcular_metricas(
    pergunta: str, resposta: str, contexto: str
) -> Dict[str, float]:
    """Calcula todas as métricas para uma resposta."""
    return {
        "precisao": round(metrica_precisao(resposta, contexto), 4),
        "completude": round(metrica_completude(resposta, pergunta, contexto), 4),
        "especificidade": round(metrica_especificidade(resposta), 4),
        "rastreabilidade": round(metrica_rastreabilidade(resposta), 4),
        "confianca": round(metrica_confianca(resposta), 4),
    }


def score_final(metricas: Dict[str, float]) -> float:
    """Score final ponderado (0-100)."""
    pesos = {
        "precisao": 0.30,
        "completude": 0.25,
        "especificidade": 0.20,
        "rastreabilidade": 0.15,
        "confianca": 0.10,
    }
    score = sum(metricas.get(k, 0.0) * w for k, w in pesos.items())
    return round(score * 100, 2)


# ============================================================================
# 4. AUDITORIA E RELATÓRIOS
# ============================================================================

def criar_registro_auditoria(
    abordagem: str,
    pergunta: str,
    documentos: List[str],
    contexto: str,
    resposta: str,
    tempo: float,
    metadados: Dict,
) -> Dict:
    """Cria um registro de auditoria estruturado para uma pergunta."""
    metricas = calcular_metricas(pergunta, resposta, contexto)
    return {
        "timestamp": datetime.now().isoformat(),
        "abordagem": abordagem,
        "pergunta": pergunta,
        "documentos_recuperados": documentos,
        "contexto_usado": contexto,
        "contexto_preview": contexto[:500],
        "resposta": resposta,
        "resposta_detalhada": resposta,
        "tempo_segundos": round(tempo, 3),
        "metadados_llm": metadados,
        "metricas": metricas,
        "score_final": score_final(metricas),
        "tokens_estimados": len(contexto.split()) + len(resposta.split()),
    }


def salvar_resultados_csv(
    abordagem: str, registros: List[Dict]
) -> Path:
    """Salva os registros de uma abordagem em CSV."""
    caminho = config.RESULTADOS_DIR / f"resultados_{abordagem}_{config.TIMESTAMP}.csv"
    colunas = [
        "timestamp", "abordagem", "pergunta", "documentos_recuperados",
        "resposta", "tempo_segundos",
        "precisao", "completude", "especificidade",
        "rastreabilidade", "confianca", "score_final",
    ]
    with open(caminho, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=colunas)
        writer.writeheader()
        for r in registros:
            m = r["metricas"]
            writer.writerow({
                "timestamp": r["timestamp"],
                "abordagem": r["abordagem"],
                "pergunta": r["pergunta"],
                "documentos_recuperados": " | ".join(r["documentos_recuperados"]),
                "resposta": r["resposta"],
                "tempo_segundos": r["tempo_segundos"],
                "precisao": m["precisao"],
                "completude": m["completude"],
                "especificidade": m["especificidade"],
                "rastreabilidade": m["rastreabilidade"],
                "confianca": m["confianca"],
                "score_final": r["score_final"],
            })
    return caminho


def salvar_registros_json(abordagem: str, registros: List[Dict]) -> Path:
    """Salva os registros completos (JSON) para auditoria detalhada."""
    caminho = config.RESULTADOS_DIR / f"auditoria_{abordagem}_{config.TIMESTAMP}.json"
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(registros, f, ensure_ascii=False, indent=2)
    return caminho


# ============================================================================
# 5. RELATÓRIO CONSOLIDADO
# ============================================================================

def gerar_relatorio_consolidado(todos_registros: Dict[str, List[Dict]]) -> Path:
    """
    Gera um relatório CSV consolidado comparando todas as abordagens.

    todos_registros: {"tfidf": [registros], "llm_puro": [...], ...}
    """
    caminho = config.RESULTADOS_DIR / f"comparativo_abordagens_{config.TIMESTAMP}.csv"
    colunas = [
        "pergunta", "abordagem", "tempo_segundos",
        "precisao", "completude", "especificidade",
        "rastreabilidade", "confianca", "score_final",
    ]
    with open(caminho, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=colunas)
        writer.writeheader()
        for abordagem, registros in todos_registros.items():
            for r in registros:
                m = r["metricas"]
                writer.writerow({
                    "pergunta": r["pergunta"],
                    "abordagem": abordagem,
                    "tempo_segundos": r["tempo_segundos"],
                    "precisao": m["precisao"],
                    "completude": m["completude"],
                    "especificidade": m["especificidade"],
                    "rastreabilidade": m["rastreabilidade"],
                    "confianca": m["confianca"],
                    "score_final": r["score_final"],
                })
    return caminho


def gerar_resumo_estatistico(todos_registros: Dict[str, List[Dict]]) -> Path:
    """Gera um resumo estatístico em texto comparando as abordagens."""
    caminho = config.RESULTADOS_DIR / f"resumo_estatistico_{config.TIMESTAMP}.txt"
    linhas = []
    linhas.append("=" * 70)
    linhas.append("RESUMO ESTATÍSTICO - COMPARAÇÃO DE ABORDAGENS")
    linhas.append("=" * 70)
    linhas.append(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    linhas.append("")

    metricas_nomes = ["precisao", "completude", "especificidade",
                      "rastreabilidade", "confianca", "score_final"]

    # Médias por abordagem
    linhas.append("MÉDIAS POR ABORDAGEM:")
    linhas.append("-" * 70)
    cabecalho = f"{'Abordagem':<20}" + "".join(f"{m:>16}" for m in metricas_nomes)
    linhas.append(cabecalho)
    for abordagem, registros in todos_registros.items():
        if not registros:
            continue
        n = len(registros)
        medias = []
        for m in metricas_nomes:
            if m == "score_final":
                vals = [r["score_final"] for r in registros]
            elif m == "tempo_segundos":
                vals = [r["tempo_segundos"] for r in registros]
            else:
                vals = [r["metricas"][m] for r in registros]
            medias.append(sum(vals) / n)
        linha = f"{abordagem:<20}" + "".join(f"{v:>16.3f}" for v in medias)
        linhas.append(linha)

    linhas.append("")
    linhas.append(f"Total de perguntas: {len(next(iter(todos_registros.values()), []))}")
    linhas.append("")

    # Ranking por score final
    linhas.append("RANKING (por score final médio):")
    linhas.append("-" * 70)
    ranking = []
    for abordagem, registros in todos_registros.items():
        if not registros:
            continue
        media = sum(r["score_final"] for r in registros) / len(registros)
        ranking.append((abordagem, media))
    ranking.sort(key=lambda x: x[1], reverse=True)
    for i, (ab, media) in enumerate(ranking, 1):
        linhas.append(f"  {i}. {ab:<20} {media:.2f}")
    if ranking:
        linhas.append("")
        linhas.append(f"🏆 VENCEDOR: {ranking[0][0]} ({ranking[0][1]:.2f} pontos)")

    with open(caminho, "w", encoding="utf-8") as f:
        f.write("\n".join(linhas))
    return caminho