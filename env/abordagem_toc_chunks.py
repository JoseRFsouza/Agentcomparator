#!/usr/bin/env python3
"""
================================================================================
ABORDAGEM 5: TOC-GUIADO + BUSCA EM CHUNKS (HÍBRIDA)
================================================================================
Estratégia em 2 estágios:

  Estágio 1 — Busca no TOC (tabela de conteúdo):
    Compara a pergunta (traduzida PT→EN) com as descrições de tasks do TOC
    usando TF-IDF sobre FRASES curtas e descritivas (task_name, subject_name).
    Isso identifica os capítulos ATA mais relevantes com alta precisão.

  Estágio 2 — Busca em chunks dentro dos documentos selecionados:
    Busca TF-IDF apenas nos chunks dos documentos dos ATAs identificados,
    retornando os trechos mais relevantes para o LLM.

Isso combina a precisão da busca estruturada (TOC) com a cobertura da
busca em texto completo (chunks).
================================================================================
"""

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import config
import comum
import traducao


# ============================================================================
# NORMALIZAÇÃO
# ============================================================================

def normalizar(texto: str) -> str:
    texto = texto.lower()
    texto = unicodedata.normalize("NFKD", texto)
    texto = texto.encode("ascii", "ignore").decode("ascii")
    return texto


def limpar(texto: str) -> str:
    texto = normalizar(texto)
    return re.sub(r"[^a-z0-9\s\-]", " ", texto)


# ============================================================================
# ESTÁGIO 1: BUSCA NO TOC
# ============================================================================

class BuscadorTOC:
    """Indexa o TOC e busca os capítulos ATA relevantes para uma pergunta."""

    def __init__(self, toc: List[Dict]):
        self.toc = toc
        # Montar textos: dar MUITO mais peso ao subject_name (nome do capítulo)
        # repetindo-o para aumentar seu peso no TF-IDF
        self.textos = []
        for entrada in toc:
            subject = entrada.get("subject_name", "")
            task = entrada.get("task_name", "")
            section = entrada.get("chapter_section_subject", "")
            # subject_name repetido 3x para dar mais peso
            texto = f"{subject} {subject} {subject} {task} {section}"
            self.textos.append(limpar(texto))

        self.vetorizador = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=20000,
            sublinear_tf=True,
        )
        self.matriz = self.vetorizador.fit_transform(self.textos)

    def buscar(self, pergunta: str, top_atas: int = 3) -> List[Tuple[str, float, List[Dict]]]:
        """
        Retorna os ATAs mais relevantes: [(ata, score, entradas)].
        
        Estratégia em 2 camadas:
        1. Busca direta por palavras-chave no subject_name (boost alto)
        2. TF-IDF tradicional como fallback
        """
        pergunta_expandida = traducao.expandir_consulta(pergunta)
        pergunta_limpa = limpar(pergunta)
        texto_busca = limpar(pergunta + " " + pergunta_expandida)
        
        # Palavras significativas da pergunta (>= 4 chars, sem stopwords comuns)
        stopwords = {"quantos", "quantas", "qual", "quais", "como", "onde",
                     "quando", "tem", "boeing", "aircraft", "have", "what",
                     "how", "many", "much", "fazer", "para", "sobre",
                     "normal", "operacao", "operation", "acontece", "houver",
                     "falha", "failure", "sistema", "system"}
        palavras_chave = set(re.findall(r"[a-z]{4,}", texto_busca)) - stopwords
        
        # Mapeamento de contexto para ATA (palavra-chave -> ATA prioritário)
        contexto_ata = {
            "hidraulic": ["29"], "hydraulic": ["29"],
            "pouso": ["32"], "landing": ["32"], "gear": ["32"],
            "fcc": ["27", "22"], "flight": ["27", "22"],
            "combustivel": ["28"], "fuel": ["28"],
            "motor": ["71", "72"], "engine": ["71", "72"],
            "eletric": ["24"], "eletrica": ["24"],
            "pneumatic": ["36"],
            "pressurizacao": ["21"], "conditioning": ["21"],
            "navegacao": ["34"], "navigation": ["34"],
        }
        
        # Forçar ATAs baseado em contexto
        atas_forcados = []
        for palavra in palavras_chave:
            for chave, atas in contexto_ata.items():
                if chave in palavra:
                    atas_forcados.extend(atas)

        # ---- Camada 1: match direto no subject_name ---------------------
        boost_ata: Dict[str, float] = {}
        for idx, entrada in enumerate(self.toc):
            subject = limpar(entrada.get("subject_name", ""))
            task = limpar(entrada.get("task_name", ""))
            ata = entrada.get("ata_principal", "")
            if not ata:
                continue
            
            score_direto = 0.0
            for palavra in palavras_chave:
                if palavra in subject:
                    score_direto += 2.0  # Match no subject = forte
                elif palavra in task:
                    score_direto += 1.0  # Match no task = médio
            
            if score_direto > 0:
                if ata not in boost_ata or score_direto > boost_ata[ata]:
                    boost_ata[ata] = score_direto

        # ---- Camada 2: TF-IDF tradicional -------------------------------
        vec = self.vetorizador.transform([texto_busca])
        scores_tfidf = cosine_similarity(vec, self.matriz).flatten()

        melhor_por_ata: Dict[str, Tuple[float, List[int]]] = {}
        for idx, score in enumerate(scores_tfidf):
            if score <= 0:
                continue
            ata = self.toc[idx].get("ata_principal", "")
            if not ata:
                continue
            if ata not in melhor_por_ata:
                melhor_por_ata[ata] = (score, [idx])
            elif score > melhor_por_ata[ata][0]:
                melhor_por_ata[ata] = (score, melhor_por_ata[ata][1] + [idx])
            else:
                melhor_por_ata[ata][1].append(idx)

        # ---- Combinar: boost direto + TF-IDF + ATAs forçados ------------
        score_final_ata: Dict[str, Tuple[float, List[int]]] = {}
        
        # Adicionar scores do TF-IDF
        for ata, (score, indices) in melhor_por_ata.items():
            score_final_ata[ata] = (score, indices)
        
        # Aplicar boost da busca direta
        for ata, boost in boost_ata.items():
            if ata in score_final_ata:
                old_score, indices = score_final_ata[ata]
                score_final_ata[ata] = (old_score + boost, indices)
            else:
                indices_ata = [i for i, e in enumerate(self.toc) 
                              if e.get("ata_principal") == ata][:3]
                score_final_ata[ata] = (boost, indices_ata)
        
        # Forçar ATAs de contexto (boost alto para garantir presença)
        for ata in atas_forcados:
            indices_ata = [i for i, e in enumerate(self.toc)
                          if e.get("ata_principal") == ata][:3]
            if ata in score_final_ata:
                old_score, old_indices = score_final_ata[ata]
                score_final_ata[ata] = (old_score + 5.0, old_indices)
            else:
                score_final_ata[ata] = (5.0, indices_ata)

        # Ordenar e retornar top N
        atas_ordenados = sorted(
            score_final_ata.items(), key=lambda x: x[1][0], reverse=True
        )[:top_atas]

        resultado = []
        for ata, (score, indices) in atas_ordenados:
            entradas = [self.toc[i] for i in indices[:3]]
            resultado.append((ata, score, entradas))

        return resultado


# ============================================================================
# ESTÁGIO 2: BUSCA EM CHUNKS
# ============================================================================

class BuscadorChunks:
    """Indexa chunks e busca dentro de documentos específicos."""

    def __init__(self, chunks: List[Dict]):
        self.chunks = chunks
        # Agrupar chunks por arquivo para busca filtrada
        self.chunks_por_arquivo: Dict[str, List[int]] = {}
        for i, c in enumerate(chunks):
            self.chunks_por_arquivo.setdefault(c["arquivo"], []).append(i)

        textos = [limpar(c["texto"]) for c in chunks]
        self.vetorizador = TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=50000,
            min_df=1,
            max_df=0.90,
            sublinear_tf=True,
        )
        self.matriz = self.vetorizador.fit_transform(textos)

    def buscar_nos_arquivos(
        self, pergunta: str, arquivos: List[str], top_k: int = 5
    ) -> List[Tuple[Dict, float]]:
        """Busca chunks apenas nos arquivos especificados."""
        # Coletar índices dos chunks dos arquivos relevantes
        indices_validos = []
        for arq in arquivos:
            indices_validos.extend(self.chunks_por_arquivo.get(arq, []))

        if not indices_validos:
            return []

        pergunta_expandida = traducao.expandir_consulta(pergunta)
        texto_busca = limpar(pergunta + " " + pergunta_expandida)
        vec = self.vetorizador.transform([texto_busca])

        # Calcular similaridade apenas para os chunks válidos
        submatriz = self.matriz[indices_validos]
        scores = cosine_similarity(vec, submatriz).flatten()

        # Top-K
        top_local = np.argsort(scores)[::-1][:top_k]
        resultados = []
        for i in top_local:
            if scores[i] > 0:
                idx_global = indices_validos[i]
                resultados.append((self.chunks[idx_global], float(scores[i])))

        return resultados


# ============================================================================
# EXECUÇÃO DA ABORDAGEM
# ============================================================================

# Cache global para não reindexar a cada pergunta
_cache = {}


def _inicializar():
    if "toc" not in _cache:
        toc = comum.carregar_indice_toc()
        _cache["toc"] = BuscadorTOC(toc)
        print(f"  TOC indexado: {len(toc)} entradas")

    if "chunks" not in _cache:
        caminho = config.BASE_DIR / "base_chunks.json"
        with open(caminho, "r", encoding="utf-8") as f:
            dados = json.load(f)
        _cache["chunks"] = BuscadorChunks(dados["chunks"])
        print(f"  Chunks indexados: {dados['total_chunks']} chunks")


def executar_abordagem(perguntas: List[str]) -> List[Dict]:
    """Executa a abordagem híbrida TOC + chunks."""
    _inicializar()
    buscador_toc: BuscadorTOC = _cache["toc"]
    buscador_chunks: BuscadorChunks = _cache["chunks"]

    registros = []
    for pergunta in perguntas:
        # ---- Estágio 1: Buscar ATAs relevantes no TOC ----
        atas = buscador_toc.buscar(pergunta, top_atas=2)
        arquivos_alvo = []
        info_toc = []
        for ata, score_ata, entradas in atas:
            # Mapear ATA para nome do arquivo (ex: "29" -> "29___084.PDF")
            for chunk in buscador_chunks.chunks:
                nome = chunk["arquivo"]
                if nome.replace("_", "")[:2].lstrip("0") == ata.lstrip("0") or \
                   nome[:2] == ata:
                    if nome not in arquivos_alvo:
                        arquivos_alvo.append(nome)
                    break
            for e in entradas[:2]:
                info_toc.append(
                    f"ATA {ata}: {e.get('task_name', '')} ({e.get('task_number', '')})"
                )

        # ---- Estágio 2: Buscar chunks nos documentos selecionados ----
        if arquivos_alvo:
            resultados = buscador_chunks.buscar_nos_arquivos(
                pergunta, arquivos_alvo, top_k=config.TOP_K
            )
        else:
            # Fallback: buscar em todos os chunks
            resultados = buscador_chunks.buscar_nos_arquivos(
                pergunta,
                list(buscador_chunks.chunks_por_arquivo.keys()),
                top_k=config.TOP_K,
            )

        # ---- Montar contexto ----
        trechos = []
        fontes = []
        for chunk, score in resultados:
            fonte = chunk["arquivo"]
            if fonte not in fontes:
                fontes.append(fonte)
            trechos.append(
                f"[Fonte: {fonte} | score {score:.4f}]\n{chunk['texto']}"
            )

        contexto = "\n\n---\n\n".join(trechos)

        resposta, tempo, meta = comum.gerar_resposta_llm(pergunta, contexto=contexto)

        registro = comum.criar_registro_auditoria(
            abordagem="toc_chunks",
            pergunta=pergunta,
            documentos=fontes,
            contexto=contexto,
            resposta=resposta,
            tempo=tempo,
            metadados={
                **meta,
                "atas_identificados": info_toc,
                "arquivos_alvo": arquivos_alvo,
            },
        )
        registros.append(registro)
        print(f"  ✓ [{registro['score_final']:>6.2f}] {pergunta[:60]}")
        print(f"    ATAs: {[a for a, _, _ in atas]} | Arquivos: {arquivos_alvo}")

    return registros


def main():
    parser = argparse.ArgumentParser(description="Abordagem 5: TOC + Chunks")
    parser.add_argument("--perguntas", help="Arquivo de perguntas")
    parser.add_argument("--pergunta", help="Pergunta única")
    args = parser.parse_args()

    if args.pergunta:
        perguntas = [args.pergunta]
    else:
        perguntas = comum.carregar_perguntas()

    if not perguntas:
        print("❌ Nenhuma pergunta fornecida.")
        return

    print(f"🚀 Abordagem TOC+Chunks: {len(perguntas)} perguntas")
    registros = executar_abordagem(perguntas)

    comum.salvar_resultados_csv("toc_chunks", registros)
    comum.salvar_registros_json("toc_chunks", registros)
    print("\n✅ Resultados salvos em resultados/resultados_toc_chunks.csv")


if __name__ == "__main__":
    main()
