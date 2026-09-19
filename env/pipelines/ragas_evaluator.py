#!/usr/bin/env python3
"""
================================================================================
AVALIADOR RAGAS - Métricas de avaliação para RAG (implementação própria)
================================================================================
Implementa as métricas do RAGAS sem depender de scikit-network (incompatível
com Python 3.14). Métricas implementadas:

  - faithfulness       : resposta é fiel ao contexto recuperado?
  - answer_relevancy   : resposta é relevante para a pergunta?
  - context_precision  : contexto recuperado é relevante?
  - context_recall     : contexto contém a info do ground truth?
  - answer_correctness : resposta está correta vs ground truth?
  - answer_similarity  : similaridade semântica com ground truth

Todas as métricas usam a NVIDIA API (Llama) como LLM Judge.
Uma verificação numérica programática trata valores/ranges/unidades.
================================================================================
"""

import json
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Any

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

import config


# ============================================================================
# NORMALIZADOR NUMÉRICO (PT/EN, ranges, unidades)
# ============================================================================

class NumericNormalizer:
    """Normaliza e compara valores numéricos com unidades PT/EN."""

    UNIT_MAP = {
        "polegadas": "in", "polegada": "in", "pol": "in", "in": "in",
        "inches": "in", "inch": "in",
        "pés": "ft", "pé": "ft", "ft": "ft", "feet": "ft",
        "metros": "m", "metro": "m",
        "milímetros": "mm", "milímetro": "mm",
        "lb": "lb", "lbs": "lb", "libras": "lb", "libra": "lb",
        "pounds": "lb", "pound": "lb",
        "kg": "kg", "quilos": "kg", "quilogramas": "kg",
        "psi": "psi", "kpa": "kPa",
        "nós": "knots", "nó": "knots", "knots": "knots", "kt": "knots",
        "mph": "mph",
        "pés quadrados": "sq ft", "sq ft": "sq ft",
        "graus": "deg", "grau": "deg", "degrees": "deg", "°": "deg",
        "galões": "gal", "gal": "gal", "gpm": "gpm",
        "volts": "V", "volt": "V", "v": "V",
        "hertz": "Hz", "hz": "Hz",
    }

    COMPARISON_MAP = {
        "menor que": "<", "less than": "<", "below": "<",
        "até": "<=", "up to": "<=", "máximo": "<=", "maximum": "<=",
        "maior que": ">", "greater than": ">", "above": ">",
        "mínimo": ">=", "minimum": ">=", "pelo menos": ">=",
        "aproximadamente": "~", "approximately": "~", "about": "~",
        "cerca de": "~", "around": "~",
    }

    @classmethod
    def extract_number(cls, text: str) -> Optional[float]:
        """Extrai número de texto PT/EN (vírgula ou ponto decimal)."""
        # Padrões: 1.234,56 (BR) | 1,234.56 (US) | 1234.56 | 123,45
        patterns = [
            r"(\d{1,3}(?:\.\d{3})+,\d+)",     # 1.234.567,89 (BR)
            r"(\d{1,3}(?:,\d{3})+\.\d+)",      # 1,234,567.89 (US)
            r"(\d+[.,]\d+)",                    # 1234,56 ou 1234.56
            r"(\d+)",                           # 1234
        ]
        for pat in patterns:
            m = re.search(pat, text)
            if m:
                s = m.group(1)
                if "," in s and "." in s:
                    if s.rindex(",") > s.rindex("."):
                        s = s.replace(".", "").replace(",", ".")  # BR
                    else:
                        s = s.replace(",", "")  # US
                elif "," in s:
                    s = s.replace(",", ".")
                try:
                    return float(s)
                except ValueError:
                    continue
        return None

    @classmethod
    def extract_unit(cls, text: str) -> Optional[str]:
        text_lower = text.lower()
        for name, norm in cls.UNIT_MAP.items():
            if re.search(r"\b" + re.escape(name) + r"\b", text_lower):
                return norm
        return None

    @classmethod
    def extract_comparison(cls, text: str) -> str:
        text_lower = text.lower()
        for phrase, op in cls.COMPARISON_MAP.items():
            if phrase in text_lower:
                return op
        return "="

    @classmethod
    def compare(cls, gt_text: str, answer_text: str) -> Dict[str, Any]:
        """Compara valor numérico do GT com a resposta."""
        gt_val = cls.extract_number(gt_text)
        ans_val = cls.extract_number(answer_text)
        gt_unit = cls.extract_unit(gt_text)
        ans_unit = cls.extract_unit(answer_text)
        gt_cmp = cls.extract_comparison(gt_text)

        result = {
            "gt_value": gt_val,
            "answer_value": ans_val,
            "gt_unit": gt_unit,
            "answer_unit": ans_unit,
            "gt_comparison": gt_cmp,
            "numeric_match": None,  # None = indeterminado
            "unit_match": None,
        }

        if gt_val is None or ans_val is None:
            return result

        # Unidade compatível?
        if gt_unit and ans_unit:
            result["unit_match"] = gt_unit == ans_unit
        else:
            result["unit_match"] = True  # sem unidade = assume compatível

        # Comparar
        if gt_cmp == "<":
            result["numeric_match"] = ans_val < gt_val
        elif gt_cmp == "<=":
            result["numeric_match"] = ans_val <= gt_val
        elif gt_cmp == ">":
            result["numeric_match"] = ans_val > gt_val
        elif gt_cmp == ">=":
            result["numeric_match"] = ans_val >= gt_val
        elif gt_cmp == "~":
            result["numeric_match"] = abs(ans_val - gt_val) / max(gt_val, 1) < 0.05
        else:
            result["numeric_match"] = abs(ans_val - gt_val) / max(gt_val, 1) < 0.02

        return result


# ============================================================================
# LLM JUDGE (via NVIDIA API)
# ============================================================================

class LLMJudge:
    """Usa LLM como juiz para métricas semânticas."""

    def __init__(self, model: str = None):
        self.model = model or config.LLM_JUDGE
        self.url = config.NVIDIA_CHAT_URL
        self.headers = {
            "Authorization": f"Bearer {config.NVIDIA_API_KEY}",
            "Content-Type": "application/json",
        }

    def _call(self, prompt: str, max_tokens: int = 300) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": max_tokens,
        }
        try:
            resp = requests.post(self.url, headers=self.headers, json=payload, timeout=60)
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"].strip()
            return f"ERROR: {resp.status_code}"
        except Exception as e:
            return f"ERROR: {e}"

    def _extract_score(self, text: str) -> float:
        """Extrai score 0.0-1.0 da resposta do LLM."""
        m = re.search(r"(\d+\.?\d*)", text)
        if m:
            val = float(m.group(1))
            if val > 1.0:
                val = val / 10.0 if val <= 10.0 else val / 100.0
            return max(0.0, min(1.0, val))
        return 0.0

    def faithfulness(self, answer: str, context: str) -> float:
        """A resposta é fiel ao contexto (não inventa informação)?"""
        prompt = f"""Avalie se a RESPOSTA é fiel ao CONTEXTO. Score 0.0 a 1.0.
Responda APENAS com um número de 0.0 a 1.0.

CONTEXTO:
{context[:2000]}

RESPOSTA:
{answer[:1000]}

Score (0.0 = totalmente inventada, 1.0 = totalmente fiel):"""
        return self._extract_score(self._call(prompt))

    def answer_relevancy(self, question: str, answer: str) -> float:
        """A resposta responde à pergunta?"""
        prompt = f"""Avalie se a RESPOSTA responde à PERGUNTA. Score 0.0 a 1.0.
Responda APENAS com um número de 0.0 a 1.0.

PERGUNTA: {question}

RESPOSTA: {answer[:800]}

Score (0.0 = não responde, 1.0 = responde completamente):"""
        return self._extract_score(self._call(prompt))

    def context_precision(self, question: str, context: str) -> float:
        """O contexto recuperado é relevante para a pergunta?"""
        prompt = f"""Avalie se o CONTEXTO é relevante para responder a PERGUNTA. Score 0.0 a 1.0.
Responda APENAS com um número de 0.0 a 1.0.

PERGUNTA: {question}

CONTEXTO:
{context[:2000]}

Score (0.0 = irrelevante, 1.0 = altamente relevante):"""
        return self._extract_score(self._call(prompt))

    def context_recall(self, ground_truth: str, context: str) -> float:
        """O contexto contém a informação do ground truth?"""
        prompt = f"""Avalie se o CONTEXTO contém a informação necessária para chegar à RESPOSTA CORRETA. Score 0.0 a 1.0.
Responda APENAS com um número de 0.0 a 1.0.

RESPOSTA CORRETA (ground truth): {ground_truth}

CONTEXTO:
{context[:2000]}

Score (0.0 = info ausente, 1.0 = info presente):"""
        return self._extract_score(self._call(prompt))

    def answer_correctness(self, question: str, ground_truth: str, answer: str) -> float:
        """A resposta está correta em relação ao ground truth?"""
        prompt = f"""Avalie se a RESPOSTA está correta comparada ao GROUND TRUTH. Score 0.0 a 1.0.
Considere equivalência de unidades (polegadas=in, pés=ft), idiomas (PT/EN) e valores aproximados.
Responda APENAS com um número de 0.0 a 1.0.

PERGUNTA: {question}

GROUND TRUTH: {ground_truth}

RESPOSTA: {answer[:800]}

Score (0.0 = incorreta, 1.0 = correta):"""
        return self._extract_score(self._call(prompt))

    def answer_similarity(self, ground_truth: str, answer: str) -> float:
        """Similaridade semântica entre resposta e ground truth."""
        prompt = f"""Avalie a similaridade semântica entre RESPOSTA e GROUND TRUTH. Score 0.0 a 1.0.
Considere sinônimos, traduções PT/EN e equivalência de unidades.
Responda APENAS com um número de 0.0 a 1.0.

GROUND TRUTH: {ground_truth}

RESPOSTA: {answer[:800]}

Score (0.0 = completamente diferente, 1.0 = semanticamente igual):"""
        return self._extract_score(self._call(prompt))


# ============================================================================
# AVALIADOR PRINCIPAL
# ============================================================================

class RagasEvaluator:
    """Avalia respostas RAG usando métricas RAGAS + verificação numérica."""

    def __init__(self, judge_model: str = None):
        self.judge = LLMJudge(model=judge_model)
        self.normalizer = NumericNormalizer()

    def evaluate_single(
        self,
        question: str,
        answer: str,
        context: str,
        ground_truth: str,
    ) -> Dict[str, Any]:
        """Avalia uma única pergunta."""

        # 1. Verificação numérica programática (rápida, sem custo de API)
        numeric_check = self.normalizer.compare(ground_truth, answer)

        # 2. Métricas RAGAS via LLM Judge
        metrics = {
            "faithfulness": self.judge.faithfulness(answer, context),
            "answer_relevancy": self.judge.answer_relevancy(question, answer),
            "context_precision": self.judge.context_precision(question, context),
            "context_recall": self.judge.context_recall(ground_truth, context),
            "answer_correctness": self.judge.answer_correctness(
                question, ground_truth, answer
            ),
            "answer_similarity": self.judge.answer_similarity(ground_truth, answer),
        }

        # 3. Se verificação numérica é definitiva, sobrescreve correctness
        if numeric_check["numeric_match"] is not None:
            if numeric_check["numeric_match"] and numeric_check.get("unit_match", True):
                metrics["answer_correctness"] = 1.0
                metrics["numeric_verified"] = True
            elif not numeric_check["numeric_match"]:
                metrics["answer_correctness"] = 0.0
                metrics["numeric_verified"] = False

        # 4. Score RAGAS composto
        ragas_score = sum(metrics.values()) / len(metrics)

        return {
            "ragas_metrics": metrics,
            "ragas_score": round(ragas_score, 4),
            "numeric_check": numeric_check,
        }

    def evaluate_batch(
        self,
        records: List[Dict],
        ground_truth_map: Dict[str, str],
    ) -> List[Dict]:
        """
        Avalia um lote de registros.

        Args:
            records: registros de auditoria (com pergunta, resposta, contexto)
            ground_truth_map: {pergunta: ground_truth}

        Returns:
            Lista de resultados com métricas RAGAS
        """
        results = []
        total = len(records)

        for i, rec in enumerate(records):
            pergunta = rec["pergunta"]
            resposta = rec["resposta"]
            contexto = rec.get("contexto_usado", "")
            gt = ground_truth_map.get(pergunta, "")

            print(f"  [{i+1}/{total}] {pergunta[:60]}...")

            if not gt:
                print(f"    ⚠️  Sem ground truth para esta pergunta, pulando.")
                continue

            eval_result = self.evaluate_single(pergunta, resposta, contexto, gt)

            results.append({
                **rec,
                "ground_truth": gt,
                **eval_result,
            })

            # Rate limiting
            time.sleep(0.3)

        return results

    def summary(self, results: List[Dict]) -> Dict:
        """Gera resumo estatístico dos resultados RAGAS."""
        if not results:
            return {}

        metric_names = [
            "faithfulness", "answer_relevancy", "context_precision",
            "context_recall", "answer_correctness", "answer_similarity",
        ]

        summary = {"total_questions": len(results)}

        for m in metric_names:
            vals = [r["ragas_metrics"][m] for r in results if "ragas_metrics" in r]
            if vals:
                summary[f"avg_{m}"] = round(sum(vals) / len(vals), 4)

        scores = [r["ragas_score"] for r in results if "ragas_score" in r]
        if scores:
            summary["avg_ragas_score"] = round(sum(scores) / len(scores), 4)

        # Contagem de verificação numérica
        numeric_ok = sum(
            1 for r in results
            if r.get("numeric_check", {}).get("numeric_match") is True
        )
        numeric_total = sum(
            1 for r in results
            if r.get("numeric_check", {}).get("numeric_match") is not None
        )
        if numeric_total > 0:
            summary["numeric_accuracy"] = round(numeric_ok / numeric_total, 4)
            summary["numeric_questions"] = numeric_total

        return summary


# ============================================================================
# UTILITÁRIOS
# ============================================================================

def carregar_ground_truth_amm400() -> Dict[str, str]:
    """Carrega ground_truth_amm_400.json e retorna {pergunta: resposta}.
    Normaliza GTs que venham como lista (junta em string)."""
    gt_path = config.BASE_DIR / "base" / "ground_truth_amm_400.json"
    if not gt_path.exists():
        return {}
    with open(gt_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    resultado = {}
    for item in data:
        gt = item["ground_truth"]
        if isinstance(gt, list):
            gt = ", ".join(str(x) for x in gt)
        resultado[item["question"]] = str(gt)
    return resultado


def carregar_auditoria(abordagem: str) -> List[Dict]:
    """Carrega a auditoria MAIS RECENTE de uma abordagem.
    Prefere arquivos com timestamp (auditoria_{abordagem}_{TIMESTAMP}.json);
    se não houver, cai no legado auditoria_{abordagem}.json."""
    candidatos = sorted(config.RESULTADOS_DIR.glob(f"auditoria_{abordagem}_*.json"))
    if candidatos:
        with open(candidatos[-1], "r", encoding="utf-8") as f:
            return json.load(f)
    path = config.RESULTADOS_DIR / f"auditoria_{abordagem}.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []
