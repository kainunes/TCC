import os
import glob
import re
import unicodedata
from collections import Counter
import pandas as pd

# NLP
import nltk
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer   # stemmer para português

nltk.download('stopwords', quiet=True)
nltk.download('rslp', quiet=True)

# -----------------------------------------------
# 1. BLACKLIST EM PORTUGUÊS
# -----------------------------------------------

palavras_ia_pt = {
    "inteligencia", "artificial", "aprendizado", "maquina",
    "modelo", "modelos", "rede", "redes", "neurais", "neural",
    "profundo", "algoritmo", "algoritmos", "treinamento",
    "bert", "claude", "gemini", "llama", "chatgpt", "openai",
    "transformador", "transformadores", "atencao", "codificador",
    "ajuste", "ajustefino", "pretreinamento", "conjunto",
    "corpus", "benchmark", "incorporacao", "token", "tokens",
    "inferencia", "prompt", "recuperacao",
    "predicao", "previsao", "geracao", "gerativo", "generativo",
    "classificacao", "reforco", "recompensa", "agente", "agentes",
    "chatbot", "chatbots", "conversacional", "dialogo",
    "multimodal", "alucinacao",
    "linguagem", "grande", "texto", "textos", "baseado",
    "usando", "utilizado", "proposto", "artigo", "estudo",
    "pesquisa", "resultados", "metodo", "abordagem", "sistema",
    "sistemas", "trabalho", "analise", "dados", "base",
    "uso", "partir", "tarefa", "tarefas", "desempenho",
    "experimento", "experimentos", "avaliacao", "apresenta",
    "introduz", "recente", "atual", "existente", "anterior",
    "estado", "arte", "supera", "demonstra", "mostra",
}

stop_pt = set(stopwords.words('portuguese'))

blacklist_pt = palavras_ia_pt | stop_pt

stemmer_pt = RSLPStemmer()


# -----------------------------------------------
# 2. LEITURA DE ARQUIVOS LOCAIS
# -----------------------------------------------
def coletar_de_pasta(caminho, extensoes=('.txt', '.pdf')):
    """
    Lê todos os arquivos com extensões especificadas dentro da pasta (recursivo).
    Para PDF, tenta extrair texto com pdfplumber; se não disponível, usa PyPDF2.
    Para TXT, lê como UTF-8.
    Retorna lista de textos completos.
    """
    print(f"\n[Local] Lendo arquivos de: {caminho}")
    textos = []
    arquivos = []
    for ext in extensoes:
        arquivos.extend(glob.glob(os.path.join(caminho, '**', f'*{ext}'), recursive=True))

    if not arquivos:
        print(f"  Nenhum arquivo com extensões {extensoes} encontrado em {caminho}")
        return textos

    print(f"  Encontrados {len(arquivos)} arquivos.")
    for arq in arquivos:
        try:
            if arq.lower().endswith('.pdf'):
                texto = extrair_texto_pdf(arq)
            else:
                with open(arq, 'r', encoding='utf-8', errors='ignore') as f:
                    texto = f.read()
            if texto and len(texto.strip()) > 50:
                textos.append(texto.strip())
        except Exception as e:
            print(f"  Erro ao ler {os.path.basename(arq)}: {e}")
    print(f"  {len(textos)} textos válidos extraídos.")
    return textos

def extrair_texto_pdf(caminho_pdf):
    """Tenta extrair texto de um PDF usando pdfplumber, fallback para PyPDF2."""
    try:
        import pdfplumber
        with pdfplumber.open(caminho_pdf) as pdf:
            texto = ''.join(pagina.extract_text() or '' for pagina in pdf.pages)
        return texto
    except ImportError:
        try:
            import PyPDF2
            with open(caminho_pdf, 'rb') as f:
                leitor = PyPDF2.PdfReader(f)
                texto = ''.join(pagina.extract_text() or '' for pagina in leitor.pages)
            return texto
        except ImportError:
            print(f"  Aviso: nem pdfplumber nem PyPDF2 instalados. Não é possível ler PDFs.")
            return ""
    except Exception as e:
        print(f"  Erro ao extrair PDF {caminho_pdf}: {e}")
        return ""


# -----------------------------------------------
# 3. PRÉ-PROCESSAMENTO
# -----------------------------------------------
def limpar_pt(texto):
    texto = texto.lower()
    texto = unicodedata.normalize('NFD', texto)
    texto = ''.join(c for c in texto if unicodedata.category(c) != 'Mn')
    texto = re.sub(r'[^\w\s]', '', texto)
    texto = re.sub(r'\d+', '', texto)
    return texto

def tokenizar_e_filtrar(texto, blacklist):
    texto = limpar_pt(texto)
    tokens = []
    for p in texto.split():
        if p in blacklist or len(p) <= 2:
            continue
        tokens.append(stemmer_pt.stem(p))
    return tokens

def contar_palavras(textos, blacklist):
    counter = Counter()
    for t in textos:
        counter.update(tokenizar_e_filtrar(t, blacklist))
    return counter


# -----------------------------------------------
# 4. COMPARAÇÃO  (pre_2022  vs  pos_2022)
# -----------------------------------------------
def comparar(counter_pre, counter_pos, min_freq=10):
    palavras = set(counter_pre.keys()) | set(counter_pos.keys())
    dados = []
    for p in palavras:
        f1 = counter_pre.get(p, 0)
        f2 = counter_pos.get(p, 0)
        if f1 + f2 < min_freq:
            continue
        diff = f2 - f1
        crescimento = (f2 - f1) / f1 if f1 > 0 else float('inf')
        dados.append((p, f1, f2, diff, crescimento))

    df = pd.DataFrame(dados, columns=[
        "Palavra", "Pre_2022", "Pos_2022", "Diferença", "Crescimento_%"
    ])
    return df.sort_values("Crescimento_%", ascending=False)


# -----------------------------------------------
# 5. RELATÓRIO
# -----------------------------------------------
def gerar_relatorio(df, prefixo=""):
    finito = df[df["Crescimento_%"] != float('inf')]
    top    = finito.head(30)
    novas  = df[df["Pre_2022"] == 0].head(30)

    print(f"\n🔥 [{prefixo}] Palavras com maior crescimento (top 30):\n")
    print(top.to_string(index=False))

    print(f"\n🆕 [{prefixo}] Palavras que surgiram pós-2022 (top 30):\n")
    print(novas.to_string(index=False))

    df.to_csv(f"analise_completa_{prefixo}.csv", index=False)
    top.to_csv(f"top_crescimento_{prefixo}.csv", index=False)
    novas.to_csv(f"palavras_novas_{prefixo}.csv", index=False)

    print(f"\n✅ Arquivos gerados: analise_completa_{prefixo}.csv | "
          f"top_crescimento_{prefixo}.csv | palavras_novas_{prefixo}.csv")


# -----------------------------------------------
# MAIN  —  apenas duas pastas: pre_2022  e  pos_2022
# -----------------------------------------------
if __name__ == "__main__":

    # ---- Caminhos das duas pastas ----
    PASTA_PRE_2022 = "C:/Users/kaiqu/OneDrive/Documentos/repositorio_tccs/pre_2022"
    PASTA_POS_2022 = "C:/Users/kaiqu/OneDrive/Documentos/repositorio_tccs/pos_2022"

    # ---- Blacklist final ----
    blacklist_final = blacklist_pt
    print(f"  Blacklist final com {len(blacklist_final)} termos.")

    # ---- Leitura dos TCCs locais ----
    textos_pre = coletar_de_pasta(PASTA_PRE_2022)
    textos_pos = coletar_de_pasta(PASTA_POS_2022)

    if not textos_pre or not textos_pos:
        print("ERRO: Uma das pastas não retornou textos. Verifique os caminhos.")
        exit()

    # ---- Contagem e comparação ----
    print("\nProcessando textos...")
    freq_pre = contar_palavras(textos_pre, blacklist_final)
    freq_pos = contar_palavras(textos_pos, blacklist_final)

    df = comparar(freq_pre, freq_pos)
    gerar_relatorio(df, prefixo="manual")

    print("\n" + "="*60)
    print("RESUMO FINAL")
    print("="*60)
    print("Total de documentos analisados:")
    print(f"  Pre-2022 : {len(textos_pre)}")
    print(f"  Pos-2022 : {len(textos_pos)}")
    print("\nArquivos CSV gerados com prefixo 'manual'.")