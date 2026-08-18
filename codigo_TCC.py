import requests
import xml.etree.ElementTree as ET
import re
from collections import Counter
import pandas as pd

# NLP
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')

# -------------------------------
# 1. LISTA DE PALAVRAS DE IA (BLACKLIST)
# -------------------------------
palavras_ia = {
    "ai", "artificial", "intelligence", "machine", "learning",
    "model", "models", "neural", "network", "networks",
    "deep", "llm", "llms", "chatgpt", "gpt",
    "transformer", "training", "dataset", "datasets",
    "algorithm", "algorithms", "predict", "prediction",
    "generative", "generation"
}

# -------------------------------
# 2. STOPWORDS + STEMMER
# -------------------------------
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# junta tudo
blacklist_total = palavras_ia.union(stop_words)

# -------------------------------
# 3. COLETAR DADOS COM ANO
# -------------------------------
def coletar_arxiv_por_ano(query, max_results=500):
    url = f"http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results={max_results}"
    
    response = requests.get(url)
    root = ET.fromstring(response.content)
    
    textos_antes = []
    textos_depois = []
    
    for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
        resumo = entry.find("{http://www.w3.org/2005/Atom}summary").text
        data = entry.find("{http://www.w3.org/2005/Atom}published").text
        
        ano = int(data[:4])
        
        if ano < 2022:
            textos_antes.append(resumo)
        else:
            textos_depois.append(resumo)
    
    return textos_antes, textos_depois

# -------------------------------
# 4. LIMPEZA + NORMALIZAÇÃO
# -------------------------------
def limpar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r'[^\w\s]', '', texto)
    return texto

def normalizar(palavra):
    return stemmer.stem(palavra)

# -------------------------------
# 5. CONTAGEM COM FILTRO
# -------------------------------
def contar_palavras(textos):
    palavras = []
    
    for t in textos:
        t = limpar_texto(t)
        tokens = t.split()
        
        for p in tokens:
            if p in blacklist_total:
                continue
            
            p = normalizar(p)
            palavras.append(p)
    
    return Counter(palavras)

# -------------------------------
# 6. COMPARAÇÃO AVANÇADA
# -------------------------------
def comparar(counter_antes, counter_depois, min_freq=5):
    
    palavras = set(counter_antes.keys()) | set(counter_depois.keys())
    dados = []
    
    for p in palavras:
        f1 = counter_antes.get(p, 0)
        f2 = counter_depois.get(p, 0)
        
        if f1 + f2 < min_freq:
            continue
        
        diff = f2 - f1
        
        if f1 > 0:
            crescimento = (f2 - f1) / f1
        else:
            crescimento = float('inf')
        
        dados.append((p, f1, f2, diff, crescimento))
    
    df = pd.DataFrame(dados, columns=[
        "Palavra", "Antes_2022", "Depois_2022", "Diferença", "Crescimento_%"
    ])
    
    df = df.sort_values(by="Crescimento_%", ascending=False)
    
    return df

# -------------------------------
# 7. RELATÓRIO
# -------------------------------
def gerar_relatorio(df):
    
    top_crescimento = df.head(30)
    
    print("\n🔥 Palavras que mais cresceram (sem termos de IA):\n")
    print(top_crescimento)
    
    novas = df[df["Antes_2022"] == 0].head(30)
    
    print("\n🆕 Palavras novas após 2022:\n")
    print(novas)
    
    df.to_csv("analise_completa.csv", index=False)
    top_crescimento.to_csv("top_crescimento.csv", index=False)
    novas.to_csv("palavras_novas.csv", index=False)
    
    print("\nArquivos gerados:")
    print("- analise_completa.csv")
    print("- top_crescimento.csv")
    print("- palavras_novas.csv")

# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":
    
    print("Coletando dados...")
    
    textos_antes, textos_depois = coletar_arxiv_por_ano(
        "academic writing OR research paper",
        500
    )
    
    print(f"Antes de 2022: {len(textos_antes)} textos")
    print(f"Depois de 2022: {len(textos_depois)} textos")
    
    print("Processando textos...")
    freq_antes = contar_palavras(textos_antes)
    freq_depois = contar_palavras(textos_depois)
    
    print("Comparando...")
    df = comparar(freq_antes, freq_depois)
    
    gerar_relatorio(df)
