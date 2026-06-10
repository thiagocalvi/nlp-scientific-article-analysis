# Trabalho 2 de IIA – Processamento de Linguagem Natural
## Universidade Estadual de Maringá – Departamento de Informática
### Tema: **Segurança Cibernética** (Scopus)

---

## Estrutura do Projeto

```
cybersec_nlp/
├── artigos/                  ← Coloque aqui os 12 PDFs científicos
├── saida/                    ← Resultados gerados automaticamente
│   ├── resultados_pipeline.json
│   ├── avaliacao.json
│   ├── avaliacao_relatorio.txt
│   ├── 1_wordcloud.png
│   ├── 2_top_termos.png
│   ├── 3_heatmap.png
│   ├── 4_coocorrencia.png
│   ├── 5_similaridade.png
│   ├── 6_termos_futuros.png
│   └── 7_arvore_palavras.png
├── ontologia/                ← Ontologias geradas
│   ├── artigos_cybersec.ttl      (Turtle)
│   ├── artigos_cybersec.jsonld   (JSON-LD)
│   └── artigos_cybersec.owl      (OWL/XML)
├── main.py                   ← Ponto de entrada (executa tudo)
├── pipeline.py               ← Etapas 1 e 2
├── ontologia.py              ← Etapa 3
├── avaliacao.py              ← Etapa 4
├── visualizacoes.py          ← Visualizações (observações)
```

---

## Instalação

Crie e ative a venv
```bash
python -m venv venv
source venv/bin/activate
```
Instale as dependências
```bash
pip install -r requirements.txt
python -m nltk.downloader stopwords punkt wordnet punkt_tab
```

---

## Execução

```bash
# Coloque os 12 PDFs na pasta artigos/ e execute:
python main.py

# Ou especifique outro diretório:
python main.py --dir /caminho/para/pdfs
```

---

## O que o sistema faz

### Etapa 1 – Leitura e pré-processamento
- Lê todos os PDFs do diretório com **fitz**
- Separa automaticamente o **corpo** das **referências bibliográficas**
- Pré-processamento: tokenização, remoção de stop-words, **lematização** (WordNetLemmatizer)
- Modelos de linguagem: **Bag-of-Words** e **N-gramas** (bi e trigramas)
- Identifica os **10 termos mais citados** por artigo e globalmente
- Extrai as **referências bibliográficas**

### Etapa 2 – Extração de informações estruturadas
Usa padrões heurísticos regex para identificar:
- **Objetivo** (`objective`, `aim`, `goal`, `this paper...`)
- **Problema** (`challenge`, `problem`, `limitation`, `overcome`)
- **Metodologia** (`methodology`, `we conduct`, `experiment`, `evaluation`)
- **Contribuições** (`contributes to`, `our contribution`, `novel`)

### Etapa 3 – Ontologia
Exporta os dados extraídos em três formatos:
| Formato | Arquivo |
|---------|---------|
| **Turtle** | `ontologia/artigos_cybersec.ttl` |
| **JSON-LD** | `ontologia/artigos_cybersec.jsonld` |
| **OWL/XML** | `ontologia/artigos_cybersec.owl` |

Classes da ontologia: `ArtigoCientifico`, `Objetivo`, `Problema`, `Metodologia`, `Contribuicao`, `Referencia`, `Termo`

### Etapa 4 – Avaliação de desempenho
- Cobertura de campos (% de artigos com cada campo extraído)
- Precisão, Recall e F1-Score (quando gabarito manual está preenchido em `avaliacao.py`)

### Visualizações
1. Nuvem de palavras
2. Barras – Top 15 termos
3. Heatmap – frequência por artigo
4. Grafo de coocorrência de bigramas
5. Similaridade de Jaccard entre artigos
6. Termos frequentes em n-gramas
7. Árvore de palavras (similaridade de prefixo)

---

## Bibliotecas utilizadas (Adicionar)
