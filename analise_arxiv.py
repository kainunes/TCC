

from __future__ import annotations

import argparse
import calendar
import json
import math
import random
import re
import statistics
import sys
import time
import unicodedata
import urllib.parse
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer


def garantir_nltk() -> None:
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)


# ----------------------- BLACKLISTS -----------------------
ANCORAS_IA = {
    "llm", "llms", "gpt", "chatgpt", "bert", "roberta", "t5", "bart",
    "transformer", "transformers", "attention", "tokenizer", "token",
    "tokens", "embedding", "embeddings", "fine-tuning", "finetuning",
    "pretraining", "prompt", "prompts", "rag", "openai", "gemini",
    "llama", "claude", "mistral", "deepseek",
}
DOMINIO_COMPARTILHADO = {
    "ai", "artificial", "intelligence", "machine", "learning",
    "model", "models", "neural", "network", "networks",
    "deep", "training", "dataset", "datasets",
    "algorithm", "algorithms", "predict", "prediction",
    "generative", "generation",
}
FILLER = {
    "system", "systems", "data", "text", "texts", "approach",
    "method", "methods", "result", "results", "paper", "study",
    "work", "research", "analysis", "based", "using", "used",
    "propose", "proposed", "recent", "existing", "state", "art",
    "performance", "experiment", "experiments",
}
LATEX_RUIDO = {"textbf", "textit", "mathrm", "frac", "emph", "cite"}

# Âncoras usadas apenas para a associação com IA (contadas fora da blacklist).
ANCORAS_IA_ESTRITAS_EN = {
    "llm", "llms", "gpt", "chatgpt", "openai",
    "davinci", "opt", "flan", "instructgpt",
    "llama", "claude", "gemini", "mistral", "deepseek",
    "hallucination", "hallucinations",
    "prompt", "prompts", "rag",
    "fine-tuning", "finetuning", "pretraining",
}

# Compostos bloqueados no modo "completa" (após concatenar hífens).
COMPOSTOS_SUBSTR = {
    w.replace("-", "") for w in ANCORAS_IA if len(w.replace("-", "")) >= 5
}
COMPOSTOS_PREFIXOS = ("llm", "gpt")


# ----------------------- ESTATÍSTICA -----------------------
def g2_keyness(f1, n1, f2, n2):
    a, b, c, d = f1, f2, n1 - f1, n2 - f2
    n = a + b + c + d
    if n == 0:
        return 0.0
    row1, row2 = a + b, c + d
    col1, col2 = a + c, b + d
    g2 = 0.0
    for obs, row, col in ((a, row1, col1), (b, row1, col2),
                          (c, row2, col1), (d, row2, col2)):
        if obs > 0:
            e = row * col / n
            if e > 0:
                g2 += obs * math.log(obs / e)
    return 2.0 * g2


def log_odds_prior(f1, n1, f2, n2, alpha=0.5):
    """Razão de log-odds suavizada com prior uniforme (não é a versão
    informativa de Monroe et al. 2008)."""
    p1 = (f1 + alpha) / (n1 + 2 * alpha) if n1 > 0 else 0.5
    p2 = (f2 + alpha) / (n2 + 2 * alpha) if n2 > 0 else 0.5
    logit1 = math.log(p1 / (1 - p1))
    logit2 = math.log(p2 / (1 - p2))
    delta = logit2 - logit1
    var = (1 / (f1 + alpha) + 1 / (n1 - f1 + alpha) +
           1 / (f2 + alpha) + 1 / (n2 - f2 + alpha))
    z = delta / math.sqrt(var) if var > 0 else 0.0
    return delta, z


def z_para_p_bilateral(z):
    try:
        return math.erfc(abs(float(z)) / math.sqrt(2.0))
    except (ValueError, TypeError):
        return 1.0


def bh_fdr(pvalues, alpha=0.05):
    """Benjamini-Hochberg. Retorna (rejeitados, qvalues)."""
    n = len(pvalues)
    if n == 0:
        return [], []
    indexed = sorted(enumerate(pvalues), key=lambda x: x[1])
    p_sorted = [p for _, p in indexed]
    q_sorted = [0.0] * n
    prev_min = 1.0
    for i in range(n - 1, -1, -1):
        q = min(1.0, p_sorted[i] * n / (i + 1), prev_min)
        q_sorted[i] = q
        prev_min = q
    qvalues = [0.0] * n
    for i, (orig_idx, _) in enumerate(indexed):
        qvalues[orig_idx] = q_sorted[i]
    return [q < alpha for q in qvalues], qvalues


def mann_whitney_p(x, y):
    """Mann-Whitney bilateral, aproximação normal com correção de empates
    e de continuidade. x, y: arrays de frequência relativa por documento."""
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return 1.0
    todos = np.concatenate([x, y])
    ranks = pd.Series(todos).rank(method="average").to_numpy()
    u1 = ranks[:n1].sum() - n1 * (n1 + 1) / 2.0
    mu = n1 * n2 / 2.0
    n = n1 + n2
    _, t = np.unique(todos, return_counts=True)
    empates = float((t ** 3 - t).sum())
    var = n1 * n2 / 12.0 * ((n + 1) - empates / (n * (n - 1)))
    if var <= 0:
        return 1.0
    z = max(0.0, abs(u1 - mu) - 0.5) / math.sqrt(var)
    return math.erfc(z / math.sqrt(2.0))


def spearman_manual(xs, ys):
    n = len(xs)
    if n < 3 or len(ys) != n:
        return 0.0

    def ranks(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n - 1 and v[order[j + 1]] == v[order[j]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    dx = math.sqrt(sum((rx[i] - mx) ** 2 for i in range(n)))
    dy = math.sqrt(sum((ry[i] - my) ** 2 for i in range(n)))
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)


# ----------------------- ASSOCIAÇÃO COM IA -----------------------
def contar_ancoras_por_doc(textos, stemmer, anchors_stems):
    """Contagem de âncoras por documento (índice = doc_id), direto do texto
    limpo: sem blacklist e sem filtro parece_palavra_en."""
    saida = {}
    for doc_id, texto in enumerate(textos):
        total = 0
        for p in limpar_texto_en(texto).split():
            if stemmer.stem(p) in anchors_stems:
                total += 1
        saida[doc_id] = total
    return saida


def classificar_associacao_ia(corpus_pos, ancora_por_doc, anchors_stems,
                              stems_interesse=None, min_docs=5,
                              lift_alto=1.5, lift_baixo=0.67):
    """
    Para cada radical w, nos documentos do PÓS:
      - P_Ancora = fração dos docs com w que têm >=1 âncora
      - P_Base   = fração de todos os docs com >=1 âncora (taxa-base)
      - Lift     = P_Ancora / P_Base
      - Spearman_IA = correlação freq_w(doc) x freq_âncora(doc)
    O próprio w é excluído das âncoras quando w é uma âncora.
    """
    doc_wc = corpus_pos["doc_word_count"]
    if not doc_wc:
        return pd.DataFrame()
    doc_ids = list(doc_wc.keys())
    stems = stems_interesse if stems_interesse is not None else set(corpus_pos["token_count"])
    base_global = sum(1 for d in doc_ids if ancora_por_doc.get(d, 0) > 0) / len(doc_ids)

    linhas = []
    for stem in stems:
        docs_com_w = [d for d in doc_ids if doc_wc[d].get(stem, 0) > 0]
        if not docs_com_w:
            continue
        if stem in anchors_stems:
            freq = {d: max(0, ancora_por_doc.get(d, 0) - doc_wc[d].get(stem, 0))
                    for d in doc_ids}
            base = sum(1 for d in doc_ids if freq[d] > 0) / len(doc_ids)
        else:
            freq = {d: ancora_por_doc.get(d, 0) for d in doc_ids}
            base = base_global

        n_com = sum(1 for d in docs_com_w if freq[d] > 0)
        p_anc = n_com / len(docs_com_w)
        lift = p_anc / base if base > 0 else None
        rho = spearman_manual([doc_wc[d].get(stem, 0) for d in doc_ids],
                              [freq[d] for d in doc_ids])

        if len(docs_com_w) < min_docs or lift is None:
            classe = f"Poucos documentos (n<{min_docs})"
        elif lift >= lift_alto:
            classe = "Associada a âncoras de IA"
        elif lift <= lift_baixo:
            classe = "Evita âncoras de IA"
        else:
            classe = "Neutra"

        linhas.append({
            "Radical": stem,
            "NDocs_Pos": len(docs_com_w),
            "DocFreq_Pos_Com_Ancora": n_com,
            "P_Ancora": round(p_anc, 4),
            "P_Base": round(base, 4),
            "Lift": round(lift, 3) if lift is not None else None,
            "Spearman_IA": round(rho, 4),
            "Classificacao_IA": classe,
        })
    return pd.DataFrame(linhas)


# ----------------------- COLETA ARXIV -----------------------
API = "https://export.arxiv.org/api/query?"
NS_ATOM = "{http://www.w3.org/2005/Atom}"
NS_OS = "{http://a9.com/-/spec/opensearch/1.1/}"


def _fazer_requisicao(params, email, tentativas=5):
    url = API + urllib.parse.urlencode(params)
    headers = {"User-Agent": f"TCCResearch/1.0 (mailto:{email})"}
    for i in range(tentativas):
        try:
            r = requests.get(url, headers=headers, timeout=60)
            if r.status_code == 200:
                return r.content
            if r.status_code in (429, 503):
                time.sleep(5 * (2 ** i))
                continue
            print(f"  arXiv {r.status_code}: {r.text[:200]}")
            return None
        except requests.RequestException as e:
            print(f"  Erro de rede: {e}")
            time.sleep(5 * (2 ** i))
    return None


def _total_results(xml_bytes):
    try:
        root = ET.fromstring(xml_bytes)
        tr = root.find(f"{NS_OS}totalResults")
        return int(tr.text) if tr is not None and tr.text else 0
    except (ET.ParseError, ValueError):
        return 0


def _extrair_registros(xml_bytes):
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return []
    saida = []
    for entry in root.findall(f"{NS_ATOM}entry"):
        try:
            resumo = (entry.find(f"{NS_ATOM}summary").text or "").strip()
            aid = entry.find(f"{NS_ATOM}id").text or ""
            pub = entry.find(f"{NS_ATOM}published").text or ""
            if resumo:
                saida.append({"resumo": resumo, "id": aid, "published": pub})
        except AttributeError:
            continue
    return saida


def _lista_meses(inicio, fim):
    """('20190101','20211231') -> [(AAAAMMDD_ini, AAAAMMDD_fim), ...] por mês."""
    a_i, m_i = int(inicio[:4]), int(inicio[4:6])
    a_f, m_f = int(fim[:4]), int(fim[4:6])
    meses = []
    a, m = a_i, m_i
    while (a, m) <= (a_f, m_f):
        ult = calendar.monthrange(a, m)[1]
        ini = max(inicio, f"{a:04d}{m:02d}01")
        f = min(fim, f"{a:04d}{m:02d}{ult:02d}")
        meses.append((ini, f))
        m += 1
        if m > 12:
            a, m = a + 1, 1
    return meses


def coletar_arxiv(query, inicio, fim, max_results=500, seed=42, pausa=3.0,
                  email="seu.email@exemplo.com", pool_fator=10):
    """
    Amostragem estratificada por mês: para cada mês, descobre o total de
    resultados, escolhe um deslocamento aleatório, baixa um bloco de
    alvo*pool_fator resumos e sorteia `alvo` deles. Garante cobertura de toda
    a janela (e não só das primeiras semanas).
    """
    print(f"\n[arXiv] {query} | {inicio} -> {fim}")
    rng = random.Random(seed)
    meses = _lista_meses(inicio, fim)
    alvo = math.ceil(max_results / len(meses))
    registros, vistos = [], set()

    for ini, f in meses:
        q = f"{query} AND submittedDate:[{ini}0000 TO {f}2359]"
        base = {"search_query": q, "sortBy": "submittedDate", "sortOrder": "ascending"}
        xml = _fazer_requisicao({**base, "start": 0, "max_results": 1}, email)
        time.sleep(pausa)
        if xml is None:
            print(f"  {ini[:6]}: falha ao obter total")
            continue
        total = _total_results(xml)
        if total == 0:
            print(f"  {ini[:6]}: 0 resultados")
            continue
        pool = min(total, alvo * pool_fator, 1000)
        start = rng.randint(0, total - pool)
        xml = _fazer_requisicao({**base, "start": start, "max_results": pool}, email)
        time.sleep(pausa)
        if xml is None:
            print(f"  {ini[:6]}: falha ao baixar bloco")
            continue
        lote = [r for r in _extrair_registros(xml) if r["id"] not in vistos]
        rng.shuffle(lote)
        escolhidos = lote[:alvo]
        for r in escolhidos:
            vistos.add(r["id"])
            registros.append(r)
        print(f"  {ini[:6]}: total={total} bloco={len(lote)} amostrados={len(escolhidos)}")

    if len(registros) > max_results:
        rng.shuffle(registros)
        registros = registros[:max_results]
    registros.sort(key=lambda r: r["published"])
    if registros:
        print(f"  coletados: {len(registros)} | primeiro={registros[0]['published'][:10]} "
              f"último={registros[-1]['published'][:10]}")
        por_ano = Counter(r["published"][:4] for r in registros)
        print(f"  por ano: {dict(sorted(por_ano.items()))}")
    return registros


def coletar_com_cache(nome, query, janela, args, out_dir):
    cache = out_dir / f"cache_{nome}_{janela[0]}_{janela[1]}_{args.max_results}_{args.seed}.json"
    if cache.exists() and not args.refazer_coleta:
        print(f"\n[cache] usando {cache.name}")
        return json.loads(cache.read_text(encoding="utf-8"))
    regs = coletar_arxiv(query, janela[0], janela[1], args.max_results,
                         args.seed, email=args.email)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(regs, ensure_ascii=False), encoding="utf-8")
    return regs


# ----------------------- PRÉ-PROCESSAMENTO -----------------------
def limpar_texto_en(texto, debug=False):
    original = texto[:200] if debug else None
    texto = texto.lower()
    texto = texto.replace("seq2seq", "sequencetosequence")
    texto = texto.replace("\u2014", " ").replace("\u2013", "-")  # travessões
    texto = texto.replace("-\n", "").replace("-\r\n", "")
    texto = unicodedata.normalize("NFD", texto)
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
    texto_ascii = "".join(c for c in texto if ord(c) < 128)
    if debug:
        print(f"  [limpar] antes: {repr(original[:100])}")
        print(f"  [limpar] depois ASCII: {repr(texto_ascii[:100])}")
    texto = texto_ascii
    texto = texto.replace("$", " ").replace("\\", " ")
    # concatena hífens intralinha (fine-tuning -> finetuning)
    texto = re.sub(r"(?<=[a-z])-(?=[a-z])", "", texto)
    texto = "".join(c if c.isalpha() or c.isspace() else " " for c in texto)
    return " ".join(texto.split())


_RE_CID = re.compile(r"^cid[a-z]{1,3}$")
_RE_CONS = re.compile(r"[bcdfghjklmnpqrstvwxyz]{5,}")
_RE_VOG = re.compile(r"[aeiou]{4,}")


def parece_artefato_pdf(tok):
    if _RE_CID.match(tok):
        return True
    if _RE_CONS.search(tok):
        return True
    if _RE_VOG.search(tok):
        return True
    if len(tok) > 4 and re.search(r"[bcdfgjktpqvwyz]{2,}$", tok):
        return True
    return False


def parece_palavra_en(tok):
    if len(tok) < 3 or len(tok) > 22:
        return False
    if not tok.isascii() or not tok.isalpha():
        return False
    if parece_artefato_pdf(tok):
        return False
    vogais = sum(1 for c in tok if c in "aeiou")
    if vogais == 0:
        return False
    ratio = vogais / len(tok)
    if ratio < 0.20 or ratio > 0.80:
        return False
    v_distintas = len(set(c for c in tok if c in "aeiou"))
    if v_distintas < 2 and vogais < 3 and len(tok) > 3:
        return False
    return True


def eh_composto_bloqueado(tok, substr, prefixos):
    if any(tok.startswith(p) for p in prefixos):
        return True
    return any(s in tok and tok != s for s in substr)


def tokenizar_en(texto, blacklist_raw, blacklist_stems, stemmer,
                 debug_limpeza=False, substr=frozenset(), prefixos=()):
    texto = limpar_texto_en(texto, debug=debug_limpeza)
    pares = []
    for p in texto.split():
        if len(p) < 3:
            continue
        if not parece_palavra_en(p):
            continue
        if p in blacklist_raw:
            continue
        if (substr or prefixos) and eh_composto_bloqueado(p, substr, prefixos):
            continue
        st = stemmer.stem(p)
        if st in blacklist_stems:
            continue
        if len(st) < 3 or not parece_palavra_en(st):
            continue
        pares.append((st, p))
    return pares


def detectar_invertidas(counter, min_freq=2):
    para_remover = set()
    for tok, freq in list(counter.items()):
        if freq < min_freq or len(tok) < 4:
            continue
        rev = tok[::-1]
        if rev == tok:
            continue
        if rev in counter and counter[rev] > freq * 1.5:
            para_remover.add(tok)
    novo = Counter(counter)
    for tok in para_remover:
        del novo[tok]
    return novo, para_remover


# ----------------------- ANÁLISE -----------------------
def analisar_corpus(textos, nomes, blacklist_raw, blacklist_stems, stemmer,
                    debug_limpeza=False, substr=frozenset(), prefixos=()):
    total_tokens = 0
    token_count = Counter()
    doc_freq = Counter()
    surface_forms = defaultdict(Counter)
    tokens_por_doc = []
    doc_word_count = {}
    n_docs, n_vazios = 0, 0

    for doc_id, (texto, nome) in enumerate(zip(textos, nomes)):
        pares = tokenizar_en(texto, blacklist_raw, blacklist_stems, stemmer,
                             debug_limpeza, substr, prefixos)
        if not pares:
            n_vazios += 1
            continue
        n_docs += 1
        total_tokens += len(pares)
        tokens_por_doc.append((nome, len(pares)))
        doc_counter = Counter()
        for st, surf in pares:
            token_count[st] += 1
            surface_forms[st][surf] += 1
            doc_counter[st] += 1
        doc_word_count[doc_id] = doc_counter
        for st in doc_counter:
            doc_freq[st] += 1

    token_count, invertidas = detectar_invertidas(token_count)

    return {
        "total_tokens": total_tokens,
        "token_count": token_count,
        "doc_freq": doc_freq,
        "surface_forms": surface_forms,
        "n_docs": n_docs,
        "n_docs_vazios": n_vazios,
        "tokens_por_doc": tokens_por_doc,
        "invertidas_removidas": invertidas,
        "doc_word_count": doc_word_count,
    }


def diagnosticar_corpus(corpus, nome, limiar_outlier=3.0):
    tpd = corpus["tokens_por_doc"]
    if not tpd:
        return []
    contagens = [n for _, n in tpd]
    mediana = statistics.median(contagens)
    limite = mediana * limiar_outlier
    outliers = sorted([(a, n) for a, n in tpd if n > limite], key=lambda x: -x[1])
    print(f"\n[Diagnóstico: {nome}]")
    print(f"  docs: {len(tpd)} | tokens: {corpus['total_tokens']:,}")
    print(f"  tokens/doc: min={min(contagens):,} mediana={int(mediana):,} max={max(contagens):,}")
    if outliers:
        print(f"  {len(outliers)} outlier(s) (> {limiar_outlier}x mediana):")
        for a, n in outliers[:5]:
            print(f"     - {a}: {n:,} tokens")
    if corpus["invertidas_removidas"]:
        amostra = list(corpus["invertidas_removidas"])[:5]
        print(f"  invertidas removidas: {len(corpus['invertidas_removidas'])} (ex.: {amostra})")
    return outliers


def _docs_com_tamanho(corpus):
    return [(c, sum(c.values())) for c in corpus["doc_word_count"].values()]


def comparar(corpus_pre, corpus_pos, min_freq_total=10,
             min_freq_periodo=5, min_doc_freq_novo=3,
             anchors_stems=None, ancora_por_doc_pos=None, min_docs_ancora=5):
    n1, n2 = corpus_pre["total_tokens"], corpus_pos["total_tokens"]
    stems = set(corpus_pre["token_count"]) | set(corpus_pos["token_count"])
    linhas = []
    for stem in stems:
        f1 = corpus_pre["token_count"].get(stem, 0)
        f2 = corpus_pos["token_count"].get(stem, 0)
        if f1 + f2 < min_freq_total or max(f1, f2) < min_freq_periodo:
            continue
        p1 = f1 / n1 if n1 > 0 else 0.0
        p2 = f2 / n2 if n2 > 0 else 0.0
        cresc = ((p2 - p1) / p1 * 100.0) if p1 > 0 else None
        df1 = corpus_pre["doc_freq"].get(stem, 0)
        df2 = corpus_pos["doc_freq"].get(stem, 0)
        novo = (f1 == 0 and df2 >= min_doc_freq_novo)
        g2 = g2_keyness(f1, n1, f2, n2)
        delta, z = log_odds_prior(f1, n1, f2, n2)

        surf_c = Counter(corpus_pre["surface_forms"].get(stem, Counter()))
        surf_c.update(corpus_pos["surface_forms"].get(stem, Counter()))
        surf = surf_c.most_common(1)[0][0] if surf_c else ""

        linhas.append({
            "Radical": stem, "Forma": surf,
            "Pre_n": f1, "Pos_n": f2,
            "Pre_pct": p1 * 100.0, "Pos_pct": p2 * 100.0,
            "Cresc_pct": cresc, "Novo": novo,
            "G2": g2, "LogOdds": delta, "Z": z,
            "DocFreq_Pre": df1, "DocFreq_Pos": df2,
        })

    df = pd.DataFrame(linhas)
    if df.empty:
        return df

    # (a) teste em nível de token (log-odds Z) + BH-FDR
    pvals = [z_para_p_bilateral(z) for z in df["Z"]]
    _, q = bh_fdr(pvals, 0.05)
    df["p_valor"] = pvals
    df["p_FDR"] = q
    df["Signif_FDR"] = [x < 0.05 for x in q]

    # (b) teste em nível de documento (Mann-Whitney) + BH-FDR
    pre_docs = _docs_com_tamanho(corpus_pre)
    pos_docs = _docs_com_tamanho(corpus_pos)
    p_doc = []
    for stem in df["Radical"]:
        x = np.fromiter((c.get(stem, 0) / L for c, L in pre_docs), float, len(pre_docs))
        y = np.fromiter((c.get(stem, 0) / L for c, L in pos_docs), float, len(pos_docs))
        p_doc.append(mann_whitney_p(x, y))
    _, q_doc = bh_fdr(p_doc, 0.05)
    df["p_doc"] = p_doc
    df["p_FDR_doc"] = q_doc
    df["Signif_FDR_doc"] = [x < 0.05 for x in q_doc]
    df["Robusto"] = df["Signif_FDR"] & df["Signif_FDR_doc"]

    # (c) associação com âncoras de IA (taxa-base e lift)
    if anchors_stems and ancora_por_doc_pos is not None:
        print("\n  Classificando associação com IA...")
        classif = classificar_associacao_ia(
            corpus_pos, ancora_por_doc_pos, anchors_stems,
            stems_interesse=set(df["Radical"]), min_docs=min_docs_ancora)
        if not classif.empty:
            df = df.merge(classif, on="Radical", how="left")
            df["Classificacao_IA"] = df["Classificacao_IA"].fillna("Sem dados")
    return df


# ----------------------- RELATÓRIO -----------------------
def gerar_relatorio(df, prefixo, ts, out_dir, stats_pre, stats_pos,
                    corpus_pre, corpus_pos, outliers_pre, outliers_pos,
                    min_doc_freq_novo):
    """Grava um único CSV, com as colunas do formato antigo na mesma ordem
    e as colunas novas (teste por documento, lift) no final."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if df.empty:
        print(f"[{prefixo}] nada passou nos filtros.")
        return
    antigas = ["Radical", "Forma", "Pre_n", "Pos_n", "Pre_pct", "Pos_pct",
               "Cresc_pct", "Novo", "G2", "LogOdds", "Z", "DocFreq_Pre",
               "DocFreq_Pos", "p_valor", "p_FDR", "Signif_FDR",
               "DocFreq_Pos_Com_Ancora", "P_Ancora", "Spearman_IA",
               "Classificacao_IA"]
    novas = ["p_doc", "p_FDR_doc", "Signif_FDR_doc", "Robusto",
             "NDocs_Pos", "P_Base", "Lift"]
    cols = [c for c in antigas + novas if c in df.columns]
    caminho = out_dir / f"analise_completa_{prefixo}_{ts}.csv"
    df[cols].to_csv(caminho, index=False)
    print(f"\n[{prefixo}] {len(df)} radicais | FDR token: {int(df['Signif_FDR'].sum())}"
          f" | FDR documento: {int(df['Signif_FDR_doc'].sum())}"
          f" | robustos: {int(df['Robusto'].sum())}")
    print(f"  salvo em {caminho}")


# ----------------------- MAIN -----------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--query", default="cat:cs.CL")
    p.add_argument("--max-results", type=int, default=500)
    p.add_argument("--pre", nargs=2, default=["20190101", "20211231"],
                   metavar=("INICIO", "FIM"))
    p.add_argument("--pos", nargs=2, default=["20230101", "20251231"],
                   metavar=("INICIO", "FIM"))
    p.add_argument("--blacklist", choices=["minima", "completa", "ambas"],
                   default="ambas")
    p.add_argument("--min-freq", type=int, default=10)
    p.add_argument("--min-freq-periodo", type=int, default=5)
    p.add_argument("--min-doc-freq-novo", type=int, default=3)
    p.add_argument("--min-docs-ancora", type=int, default=5)
    p.add_argument("--excluir-outliers", action="store_true")
    p.add_argument("--limiar-outlier", type=float, default=3.0)
    p.add_argument("--debug-limpeza", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--email", default="seu.email@exemplo.com",
                   help="e-mail para o User-Agent da API do arXiv")
    p.add_argument("--refazer-coleta", action="store_true",
                   help="ignora o cache e baixa de novo")
    p.add_argument("--out-dir", default="resultados_arxiv")
    return p.parse_args()


def construir_blacklist_en(modo):
    base = set(stopwords.words("english")) | LATEX_RUIDO
    if modo == "minima":
        return base
    if modo == "completa":
        conjunto = base | ANCORAS_IA | DOMINIO_COMPARTILHADO | FILLER
        conjunto = conjunto | {w.replace("-", "") for w in conjunto if "-" in w}
        return conjunto
    raise ValueError(modo)


def filtrar_outliers(textos, nomes, corpus, limiar):
    tpd = corpus["tokens_por_doc"]
    if not tpd:
        return textos, nomes
    mediana = statistics.median([n for _, n in tpd])
    limite = mediana * limiar
    outliers = {nome for nome, n in tpd if n > limite}
    if not outliers:
        return textos, nomes
    print(f"  removendo {len(outliers)} outlier(s)")
    novos_t, novos_n = [], []
    for t, n in zip(textos, nomes):
        if n not in outliers:
            novos_t.append(t)
            novos_n.append(n)
    return novos_t, novos_n


def main():
    args = parse_args()
    garantir_nltk()
    stemmer = SnowballStemmer("english")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    anchors_stems = {stemmer.stem(w.replace("-", "")) for w in ANCORAS_IA_ESTRITAS_EN}

    q_pre = args.query
    q_pos = args.query
    regs_pre = coletar_com_cache("pre", q_pre, args.pre, args, out_dir)
    regs_pos = coletar_com_cache("pos", q_pos, args.pos, args, out_dir)
    if not regs_pre or not regs_pos:
        print("ERRO: uma das janelas não retornou abstracts.", file=sys.stderr)
        sys.exit(1)

    textos_pre = [r["resumo"] for r in regs_pre]
    textos_pos = [r["resumo"] for r in regs_pos]

    nomes_pre = [f"pre_{i}" for i in range(len(textos_pre))]
    nomes_pos = [f"pos_{i}" for i in range(len(textos_pos))]
    stats_pre = {"fonte": "arxiv", "janela": args.pre, "coletados": len(textos_pre),
                 "primeiro": regs_pre[0]["published"][:10], "ultimo": regs_pre[-1]["published"][:10]}
    stats_pos = {"fonte": "arxiv", "janela": args.pos, "coletados": len(textos_pos),
                 "primeiro": regs_pos[0]["published"][:10], "ultimo": regs_pos[-1]["published"][:10]}

    modos = ["minima", "completa"] if args.blacklist == "ambas" else [args.blacklist]

    for modo in modos:
        print("\n" + "=" * 70)
        print(f"MODO: {modo}")
        print("=" * 70)
        raw = construir_blacklist_en(modo)
        stems_bl = {stemmer.stem(w) for w in raw}
        substr = frozenset(COMPOSTOS_SUBSTR) if modo == "completa" else frozenset()
        pref = COMPOSTOS_PREFIXOS if modo == "completa" else ()
        print(f"  blacklist: {len(raw)} termos + {len(stems_bl)} radicais"
              f" | compostos bloqueados: {len(substr) + len(pref)}")

        t_pre, n_pre, t_pos, n_pos = textos_pre, nomes_pre, textos_pos, nomes_pos
        cp = analisar_corpus(t_pre, n_pre, raw, stems_bl, stemmer,
                             args.debug_limpeza, substr, pref)
        cpos = analisar_corpus(t_pos, n_pos, raw, stems_bl, stemmer,
                               args.debug_limpeza, substr, pref)

        o_pre = diagnosticar_corpus(cp, "PRÉ", args.limiar_outlier)
        o_pos = diagnosticar_corpus(cpos, "PÓS", args.limiar_outlier)

        if args.excluir_outliers and (o_pre or o_pos):
            print("\n[Re-analisando sem outliers...]")
            t_pre, n_pre = filtrar_outliers(textos_pre, nomes_pre, cp, args.limiar_outlier)
            t_pos, n_pos = filtrar_outliers(textos_pos, nomes_pos, cpos, args.limiar_outlier)
            cp = analisar_corpus(t_pre, n_pre, raw, stems_bl, stemmer,
                                 args.debug_limpeza, substr, pref)
            cpos = analisar_corpus(t_pos, n_pos, raw, stems_bl, stemmer,
                                   args.debug_limpeza, substr, pref)
            o_pre, o_pos = [], []

        print(f"\n  PRÉ: {cp['n_docs']} docs | {cp['total_tokens']:,} tokens")
        print(f"  PÓS: {cpos['n_docs']} docs | {cpos['total_tokens']:,} tokens")

        # âncoras contadas no texto bruto limpo do PÓS (índices alinhados a t_pos)
        ancora_pos = contar_ancoras_por_doc(t_pos, stemmer, anchors_stems)
        com_anc = sum(1 for v in ancora_pos.values() if v > 0)
        print(f"  PÓS com >=1 âncora de IA: {com_anc}/{len(ancora_pos)} "
              f"({com_anc / max(1, len(ancora_pos)):.1%})")

        df = comparar(cp, cpos, args.min_freq, args.min_freq_periodo,
                      args.min_doc_freq_novo, anchors_stems=anchors_stems,
                      ancora_por_doc_pos=ancora_pos,
                      min_docs_ancora=args.min_docs_ancora)

        gerar_relatorio(df, f"arxiv_{modo}", ts, out_dir,
                        stats_pre, stats_pos, cp, cpos, o_pre, o_pos,
                        args.min_doc_freq_novo)


if __name__ == "__main__":
    main()
