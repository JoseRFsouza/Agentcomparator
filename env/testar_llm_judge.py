#!/usr/bin/env python3
"""
================================================================================
TESTE DE LLMs JUDGE - Verifica e compara LLMs de avaliação (RAGAS)
================================================================================
Testa quais LLMs funcionam como juiz (judge) via API NVIDIA e compara
seu desempenho em avaliação semântica.

Uso:
  python testar_llm_judge.py                       # testa todos os modelos
  python testar_llm_judge.py --modelo openai/gpt-oss-20b
================================================================================
"""

import argparse
import json
import time

import requests

import config


def testar_modelo(modelo: str) -> dict:
    """Testa se um modelo responde e avalia um caso simples."""
    print(f"\n{'='*70}")
    print(f"🧪 TESTANDO MODELO: {modelo}")
    print(f"{'='*70}")

    headers = {
        "Authorization": f"Bearer {config.NVIDIA_API_KEY}",
        "Content-Type": "application/json",
    }

    resultado = {"modelo": modelo, "funciona": False, "testes": {}}

    # ------------------------------------------------------------------
    # TESTE 1: Resposta básica (o modelo responde?)
    # ------------------------------------------------------------------
    print("\n[1/3] Testando resposta básica...")
    payload = {
        "model": modelo,
        "messages": [{"role": "user", "content": "Responda apenas: OK"}],
        "temperature": 0.0,
        "max_tokens": 50,
    }
    try:
        resp = requests.post(config.NVIDIA_CHAT_URL, headers=headers, json=payload, timeout=60)
        if resp.status_code == 200:
            content = resp.json()["choices"][0]["message"]["content"].strip()
            resultado["testes"]["resposta_basica"] = {"ok": True, "resposta": content}
            print(f"   ✅ Respondeu: {content[:50]}")
        else:
            resultado["testes"]["resposta_basica"] = {
                "ok": False,
                "erro": f"HTTP {resp.status_code}: {resp.text[:200]}",
            }
            print(f"   ❌ Erro HTTP {resp.status_code}: {resp.text[:100]}")
            return resultado
    except Exception as e:
        resultado["testes"]["resposta_basica"] = {"ok": False, "erro": str(e)}
        print(f"   ❌ Erro: {e}")
        return resultado

    # ------------------------------------------------------------------
    # TESTE 2: Avaliação de corretude (funciona como judge?)
    # ------------------------------------------------------------------
    print("[2/3] Testando avaliação de corretude...")
    payload = {
        "model": modelo,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Avalie se a RESPOSTA está correta comparada ao GROUND TRUTH. "
                    "Score 0.0 a 1.0. Responda APENAS com um número de 0.0 a 1.0.\n\n"
                    "GROUND TRUTH: 25 lb-in\n\n"
                    "RESPOSTA: O torque máximo é de 25 libras-polegada\n\n"
                    "Score (0.0 = incorreta, 1.0 = correta):"
                ),
            }
        ],
        "temperature": 0.0,
        "max_tokens": 50,
    }
    try:
        resp = requests.post(config.NVIDIA_CHAT_URL, headers=headers, json=payload, timeout=60)
        if resp.status_code == 200:
            content = resp.json()["choices"][0]["message"]["content"].strip()
            resultado["testes"]["avaliacao_corretude"] = {"ok": True, "resposta": content}
            print(f"   ✅ Score retornado: {content[:30]}")
        else:
            resultado["testes"]["avaliacao_corretude"] = {
                "ok": False,
                "erro": f"HTTP {resp.status_code}: {resp.text[:200]}",
            }
            print(f"   ❌ Erro HTTP {resp.status_code}")
    except Exception as e:
        resultado["testes"]["avaliacao_corretude"] = {"ok": False, "erro": str(e)}
        print(f"   ❌ Erro: {e}")

    # ------------------------------------------------------------------
    # TESTE 3: Resposta numérica consistente (retorna só número?)
    # ------------------------------------------------------------------
    print("[3/3] Testando consistência numérica...")
    payload = {
        "model": modelo,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Avalie se a RESPOSTA é fiel ao CONTEXTO. Score 0.0 a 1.0.\n"
                    "Responda APENAS com um número de 0.0 a 1.0.\n\n"
                    "CONTEXTO: O torque máximo é de 25 lb-in.\n\n"
                    "RESPOSTA: O torque máximo é de 30 lb-in.\n\n"
                    "Score (0.0 = totalmente inventada, 1.0 = totalmente fiel):"
                ),
            }
        ],
        "temperature": 0.0,
        "max_tokens": 50,
    }
    try:
        resp = requests.post(config.NVIDIA_CHAT_URL, headers=headers, json=payload, timeout=60)
        if resp.status_code == 200:
            content = resp.json()["choices"][0]["message"]["content"].strip()
            resultado["testes"]["consistencia_numerica"] = {"ok": True, "resposta": content}
            print(f"   ✅ Score retornado: {content[:30]}")
            # Verifica se retorna apenas número
            import re
            if re.match(r"^\d+\.?\d*$", content):
                resultado["so_numero"] = True
                print("   ✅ Retorna apenas número")
            else:
                resultado["so_numero"] = False
                print("   ⚠️  Retornou texto além do número")
        else:
            resultado["testes"]["consistencia_numerica"] = {
                "ok": False,
                "erro": f"HTTP {resp.status_code}",
            }
            print(f"   ❌ Erro HTTP {resp.status_code}")
    except Exception as e:
        resultado["testes"]["consistencia_numerica"] = {"ok": False, "erro": str(e)}
        print(f"   ❌ Erro: {e}")

    resultado["funciona"] = all(
        t.get("ok", False) for t in resultado["testes"].values()
    )
    return resultado


def main():
    parser = argparse.ArgumentParser(description="Testa LLMs judge")
    parser.add_argument("--modelo", help="Modelo específico a testar")
    args = parser.parse_args()

    modelos = [
        "openai/gpt-oss-20b",
        "google/diffusiongemma-26b-a4b-it",
    ]

    if args.modelo:
        modelos = [args.modelo]

    resultados = []
    for modelo in modelos:
        try:
            r = testar_modelo(modelo)
            resultados.append(r)
        except Exception as e:
            print(f"\n❌ Falha ao testar {modelo}: {e}")
            resultados.append({"modelo": modelo, "funciona": False, "erro": str(e)})

    # ------------------------------------------------------------------
    # RESUMO FINAL
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("📊 RESUMO FINAL")
    print(f"{'='*70}")
    for r in resultados:
        status = "✅ FUNCIONA" if r["funciona"] else "❌ FALHOU"
        print(f"\n{status}: {r['modelo']}")
        for nome, teste in r.get("testes", {}).items():
            if teste.get("ok"):
                print(f"   ✅ {nome}: {teste.get('resposta', '')[:40]}")
            else:
                print(f"   ❌ {nome}: {teste.get('erro', 'erro')}")
        if r.get("so_numero"):
            print("   → Retorna score numérico limpo (ideal para judge)")

    # Salvar resultados
    with open(config.RESULTADOS_DIR / "teste_llm_judge.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Resultados salvos em: resultados/teste_llm_judge.json")


if __name__ == "__main__":
    main()