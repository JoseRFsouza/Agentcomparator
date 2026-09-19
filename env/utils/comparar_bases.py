#!/usr/bin/env python3
"""
Script para comparar bases de conhecimento
"""

import json
from pathlib import Path


def analisar_base(caminho: Path):
    """Analisa uma base de conhecimento e retorna estatísticas."""
    print(f"\n{'=' * 70}")
    print(f"📄 Analisando: {caminho.name}")
    print(f"{'=' * 70}")
    print(f"Tamanho do arquivo: {caminho.stat().st_size / 1024 / 1024:.2f} MB")
    
    with open(caminho, 'r', encoding='utf-8') as f:
        dados = json.load(f)
    
    print(f"Tipo de dados: {type(dados).__name__}")
    
    stats = {
        'arquivo': caminho.name,
        'tamanho_mb': caminho.stat().st_size / 1024 / 1024,
        'tipo': type(dados).__name__
    }
    
    if isinstance(dados, dict):
        print(f"Chaves principais: {list(dados.keys())}")
        stats['chaves'] = list(dados.keys())
        
        if 'documentos' in dados:
            docs = dados['documentos']
            stats['num_documentos'] = len(docs)
            print(f"Número de documentos: {len(docs)}")
            
            if docs:
                primeiro = docs[0]
                print(f"\nEstrutura do primeiro documento:")
                print(f"  Chaves: {list(primeiro.keys())}")
                
                stats['campos_documento'] = list(primeiro.keys())
                
                # Estatísticas de conteúdo
                tamanhos = []
                arquivos_unicos = set()
                
                for doc in docs:
                    conteudo = doc.get('conteudo_completo', '') or doc.get('conteudo', '')
                    tamanhos.append(len(conteudo))
                    arquivos_unicos.add(doc.get('arquivo', 'sem_nome'))
                
                stats['arquivos_unicos'] = len(arquivos_unicos)
                stats['tamanho_medio_conteudo'] = sum(tamanhos) / len(tamanhos) if tamanhos else 0
                stats['tamanho_min'] = min(tamanhos) if tamanhos else 0
                stats['tamanho_max'] = max(tamanhos) if tamanhos else 0
                
                print(f"\nEstatísticas de conteúdo:")
                print(f"  Arquivos únicos: {len(arquivos_unicos)}")
                print(f"  Tamanho médio: {stats['tamanho_medio_conteudo']:.0f} caracteres")
                print(f"  Tamanho mínimo: {stats['tamanho_min']} caracteres")
                print(f"  Tamanho máximo: {stats['tamanho_max']} caracteres")
                
                # Exemplo de primeiro documento
                print(f"\nExemplo (primeiro documento):")
                print(f"  Arquivo: {primeiro.get('arquivo', 'N/A')}")
                conteudo_exemplo = primeiro.get('conteudo_completo', '') or primeiro.get('conteudo', '')
                print(f"  Preview: {conteudo_exemplo[:200]}...")
                
                stats['exemplo_arquivo'] = primeiro.get('arquivo', 'N/A')
                
                # Verificar se tem embeddings
                if 'embedding' in primeiro or 'embeddings' in primeiro:
                    stats['tem_embeddings'] = True
                    print(f"  ✅ Contém embeddings pré-computados")
                else:
                    stats['tem_embeddings'] = False
                    print(f"  ❌ Sem embeddings pré-computados")
    
    elif isinstance(dados, list):
        stats['num_itens'] = len(dados)
        print(f"Número de itens: {len(dados)}")
        if dados and isinstance(dados[0], dict):
            print(f"Chaves do primeiro item: {list(dados[0].keys())}")
            stats['campos_item'] = list(dados[0].keys())
    
    return stats


def comparar_bases(stats1, stats2):
    """Compara duas bases e gera relatório."""
    print(f"\n{'=' * 70}")
    print("📊 COMPARAÇÃO DAS BASES")
    print(f"{'=' * 70}")
    
    print(f"\n1. Tamanho do arquivo:")
    print(f"   {stats1['arquivo']}: {stats1['tamanho_mb']:.2f} MB")
    print(f"   {stats2['arquivo']}: {stats2['tamanho_mb']:.2f} MB")
    diff_mb = stats2['tamanho_mb'] - stats1['tamanho_mb']
    print(f"   Diferença: {diff_mb:+.2f} MB ({diff_mb/stats1['tamanho_mb']*100:+.1f}%)")
    
    if 'num_documentos' in stats1 and 'num_documentos' in stats2:
        print(f"\n2. Número de documentos:")
        print(f"   {stats1['arquivo']}: {stats1['num_documentos']}")
        print(f"   {stats2['arquivo']}: {stats2['num_documentos']}")
        diff_docs = stats2['num_documentos'] - stats1['num_documentos']
        print(f"   Diferença: {diff_docs:+d} documentos")
        
        if 'arquivos_unicos' in stats1 and 'arquivos_unicos' in stats2:
            print(f"\n3. Arquivos únicos:")
            print(f"   {stats1['arquivo']}: {stats1['arquivos_unicos']}")
            print(f"   {stats2['arquivo']}: {stats2['arquivos_unicos']}")
        
        print(f"\n4. Tamanho médio do conteúdo:")
        print(f"   {stats1['arquivo']}: {stats1['tamanho_medio_conteudo']:.0f} chars")
        print(f"   {stats2['arquivo']}: {stats2['tamanho_medio_conteudo']:.0f} chars")
        
        print(f"\n5. Embeddings pré-computados:")
        print(f"   {stats1['arquivo']}: {'✅ Sim' if stats1.get('tem_embeddings') else '❌ Não'}")
        print(f"   {stats2['arquivo']}: {'✅ Sim' if stats2.get('tem_embeddings') else '❌ Não'}")
    
    # Recomendação
    print(f"\n{'=' * 70}")
    print("💡 RECOMENDAÇÃO")
    print(f"{'=' * 70}")
    
    if 'num_documentos' in stats1 and 'num_documentos' in stats2:
        if stats2['num_documentos'] > stats1['num_documentos']:
            print(f"✅ A base '{stats2['arquivo']}' tem MAIS documentos ({stats2['num_documentos']} vs {stats1['num_documentos']})")
            print(f"   Pode ser mais completa e atualizada.")
        elif stats1['num_documentos'] > stats2['num_documentos']:
            print(f"✅ A base '{stats1['arquivo']}' tem MAIS documentos ({stats1['num_documentos']} vs {stats2['num_documentos']})")
            print(f"   Pode ser mais completa.")
        else:
            print(f"⚖️  Ambas têm o mesmo número de documentos ({stats1['num_documentos']})")
        
        if not stats1.get('tem_embeddings') and stats2.get('tem_embeddings'):
            print(f"\n✅ A base '{stats2['arquivo']}' tem embeddings pré-computados!")
            print(f"   Isso acelera o processamento FAISS.")
        elif stats1.get('tem_embeddings') and not stats2.get('tem_embeddings'):
            print(f"\n✅ A base '{stats1['arquivo']}' tem embeddings pré-computados!")
            print(f"   Isso acelera o processamento FAISS.")
        
        print(f"\n📝 Próximos passos:")
        print(f"   1. Verifique se a base mais recente cobre os mesmos tópicos")
        print(f"   2. Teste a recuperação com algumas perguntas de exemplo")
        print(f"   3. Compare a qualidade das respostas RAG com ambas as bases")


def main():
    base_dir = Path(__file__).parent
    
    base1 = base_dir / "base_conhecimento_indice.json"
    base2 = base_dir / "base_conhecimento.json"
    
    stats1 = None
    stats2 = None
    
    if base1.exists():
        stats1 = analisar_base(base1)
    else:
        print(f"❌ Arquivo não encontrado: {base1}")
    
    if base2.exists():
        stats2 = analisar_base(base2)
    else:
        print(f"❌ Arquivo não encontrado: {base2}")
    
    if stats1 and stats2:
        comparar_bases(stats1, stats2)
        
        # Salvar relatório
        relatorio = {
            'base_antiga': stats1,
            'base_nova': stats2,
        }
        
        with open(base_dir / 'comparacao_bases.json', 'w', encoding='utf-8') as f:
            json.dump(relatorio, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 Relatório salvo em: comparacao_bases.json")


if __name__ == "__main__":
    main()
