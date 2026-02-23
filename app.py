# app.py — Assistente UILCOM IPZS SUPER UI CHATGPT

import os
import json
import re
from typing import List, Dict, Optional

import numpy as np
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI


# ============================================================
# CONFIG
# ============================================================

APP_TITLE = "Assistente Contrattuale UILCOM IPZS"

PDF_PATH = os.path.join("documenti", "ccnl.pdf")
IPZS_PERMESSI_PATH = os.path.join("documenti", "PERMESSI_IPZS_COMPLETO_FINALE.txt")
LOGO_PATH = os.path.join("logo", "logo_uilcom.png")

INDEX_DIR = "index_ccnl"
VEC_PATH = os.path.join(INDEX_DIR, "vectors.npy")
META_PATH = os.path.join(INDEX_DIR, "chunks.json")

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150

LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0


# ============================================================
# STREAMLIT STYLE CHATGPT
# ============================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🟦",
    layout="centered"
)

st.markdown("""
<style>
.block-container {
    max-width: 900px;
}

.chat-message {
    padding: 12px 16px;
    border-radius: 12px;
    margin-bottom: 10px;
    line-height: 1.5;
}

.user-msg {
    background-color: #DCF8C6;
    text-align: right;
}

.bot-msg {
    background-color: #F1F0F0;
}

.logo {
    text-align: center;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# LOGO
# ============================================================

if os.path.exists(LOGO_PATH):
    st.image(LOGO_PATH, width=180)

st.title(APP_TITLE)


# ============================================================
# UTILS
# ============================================================

def clean_text(s: str) -> str:
    return " ".join((s or "").replace("\ufeff", "").split())


def normalize_key(s: str) -> str:
    s = clean_text(s).lower()
    return re.sub(r"[^a-z0-9]+", "", s)


# ============================================================
# PERMESSI IPZS
# ============================================================

def load_ipzs_permessi(path: str):

    if not os.path.exists(path):
        return []

    raw = open(path, "r", encoding="utf-8", errors="ignore").read()
    lines = raw.splitlines()

    entries = []
    i = 0

    while i < len(lines):

        title = clean_text(lines[i])

        if not title:
            i += 1
            continue

        j = i + 1

        if j < len(lines) and set(clean_text(lines[j])) <= set("-"):
            j += 1

        if j < len(lines) and clean_text(lines[j]).lower().startswith("descrizione"):
            j += 1

        desc_lines = []

        while j < len(lines):

            nxt = clean_text(lines[j])

            if nxt.lower().startswith("fine documento"):
                break

            if nxt and (j + 1 < len(lines)) and set(clean_text(lines[j + 1])) <= set("-"):
                break

            desc_lines.append(lines[j])
            j += 1

        desc = clean_text(" ".join(desc_lines))

        if title and desc:
            entries.append({
                "title": title,
                "desc": desc,
                "norm": normalize_key(title)
            })

        i = j

    return entries


def short_desc(desc: str, n=170):
    if len(desc) <= n:
        return desc
    return desc[:n] + "..."


def wants_permessi_list(q: str) -> bool:

    q = q.lower()

    triggers = [
        "a quali permessi ha diritto",
        "a quali permessi hanno diritto",
        "quali permessi",
        "elenco permessi",
        "lista permessi",
        "tutti i permessi",
        "permessi disponibili",
    ]

    return any(t in q for t in triggers)


def search_permesso(query: str, entries):

    q_norm = normalize_key(query)
    q_low = query.lower()

    for e in entries:
        if e["norm"] == q_norm:
            return e

    for e in entries:
        if e["title"].lower() in q_low:
            return e

    for e in entries:
        if q_norm in e["norm"] or e["norm"] in q_norm:
            return e

    words = [w for w in q_low.split() if len(w) >= 4]

    if words:
        for e in entries:
            text = e["desc"].lower()
            if all(w in text for w in words[:2]):
                return e

    return None


# ============================================================
# SECRETS
# ============================================================

def get_secret(key: str):
    try:
        return st.secrets[key]
    except Exception:
        return os.getenv(key)


OPENAI_API_KEY = get_secret("OPENAI_API_KEY")
UILCOM_PASSWORD = get_secret("UILCOM_PASSWORD")


# ============================================================
# AUTH
# ============================================================

if "auth_ok" not in st.session_state:
    st.session_state.auth_ok = False

if UILCOM_PASSWORD:

    pwd = st.text_input("Password UILCOM", type="password")

    if st.button("Accedi"):

        if pwd == UILCOM_PASSWORD:
            st.session_state.auth_ok = True
        else:
            st.error("Password errata")

if not st.session_state.auth_ok:
    st.stop()


# ============================================================
# SESSION
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "ipzs_permessi" not in st.session_state:
    st.session_state.ipzs_permessi = load_ipzs_permessi(IPZS_PERMESSI_PATH)


# ============================================================
# CHAT DISPLAY
# ============================================================

for msg in st.session_state.messages:

    if msg["role"] == "user":
        st.markdown(f'<div class="chat-message user-msg">{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-message bot-msg">{msg["content"]}</div>', unsafe_allow_html=True)


# ============================================================
# INPUT
# ============================================================

user_input = st.chat_input("Scrivi una domanda...")

if not user_input:
    st.stop()


# ============================================================
# ROUTER PERMESSI IPZS
# ============================================================

entries = st.session_state.ipzs_permessi

if entries and wants_permessi_list(user_input):

    risposta = "📋 **Permessi disponibili per i lavoratori IPZS:**\n\n"

    for i, e in enumerate(entries, 1):
        risposta += f"**{i}. {e['title']}** — {short_desc(e['desc'])}\n\n"

    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.messages.append({"role": "assistant", "content": risposta})

    st.rerun()


if entries:

    hit = search_permesso(user_input, entries)

    if hit:

        risposta = f"🏭 **{hit['title']}**\n\n{hit['desc']}"

        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.messages.append({"role": "assistant", "content": risposta})

        st.rerun()


# ============================================================
# CCNL RAG
# ============================================================

if not os.path.exists(VEC_PATH):

    st.warning("Indicizza prima il CCNL.")

    if st.button("Indicizza CCNL"):

        loader = PyPDFLoader(PDF_PATH)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )

        chunks = splitter.split_documents(docs)

        texts = [c.page_content for c in chunks]
        pages = [(int(c.metadata.get("page", 0)) + 1) for c in chunks]

        emb = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

        vectors = emb.embed_documents(texts)
        vectors = np.array(vectors, dtype=np.float32)

        os.makedirs(INDEX_DIR, exist_ok=True)

        np.save(VEC_PATH, vectors)

        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump(
                [{"page": p, "text": t} for p, t in zip(pages, texts)],
                f,
                ensure_ascii=False,
            )

        st.success("Indicizzazione completata")

    st.stop()


vectors = np.load(VEC_PATH)

with open(META_PATH, "r", encoding="utf-8") as f:
    meta = json.load(f)


def normalize_rows(mat):
    return mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)


mat_norm = normalize_rows(vectors)

emb = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

qvec = np.array(emb.embed_query(user_input), dtype=np.float32)
qvec = qvec / (np.linalg.norm(qvec) + 1e-12)

scores = mat_norm @ qvec

top_idx = np.argsort(-scores)[:10]

selected = [meta[i] for i in top_idx]

context = "\n\n".join([c["text"] for c in selected])


llm = ChatOpenAI(
    model=LLM_MODEL,
    temperature=LLM_TEMPERATURE,
    api_key=OPENAI_API_KEY
)

prompt = f"""
Rispondi usando SOLO il contesto CCNL.

Domanda:
{user_input}

Contesto:
{context}
"""

response = llm.invoke(prompt).content


st.session_state.messages.append({"role": "user", "content": user_input})
st.session_state.messages.append({"role": "assistant", "content": response})

st.rerun()    margin-bottom: 10px;
    line-height: 1.5;
}

.user-msg {
    background-color: #DCF8C6;
    text-align: right;
}

.bot-msg {
    background-color: #F1F0F0;
}

.logo {
    text-align: center;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# LOGO
# ============================================================

if os.path.exists(LOGO_PATH):
    st.image(LOGO_PATH, width=180)

st.title(APP_TITLE)


# ============================================================
# UTILS
# ============================================================

def clean_text(s: str) -> str:
    return " ".join((s or "").replace("\ufeff", "").split())


def normalize_key(s: str) -> str:
    s = clean_text(s).lower()
    return re.sub(r"[^a-z0-9]+", "", s)


# ============================================================
# PERMESSI IPZS
# ============================================================

def load_ipzs_permessi(path: str):

    if not os.path.exists(path):
        return []

    raw = open(path, "r", encoding="utf-8", errors="ignore").read()
    lines = raw.splitlines()

    entries = []
    i = 0

    while i < len(lines):

        title = clean_text(lines[i])

        if not title:
            i += 1
            continue

        j = i + 1

        if j < len(lines) and set(clean_text(lines[j])) <= set("-"):
            j += 1

        if j < len(lines) and clean_text(lines[j]).lower().startswith("descrizione"):
            j += 1

        desc_lines = []

        while j < len(lines):

            nxt = clean_text(lines[j])

            if nxt.lower().startswith("fine documento"):
                break

            if nxt and (j + 1 < len(lines)) and set(clean_text(lines[j + 1])) <= set("-"):
                break

            desc_lines.append(lines[j])
            j += 1

        desc = clean_text(" ".join(desc_lines))

        if title and desc:
            entries.append({
                "title": title,
                "desc": desc,
                "norm": normalize_key(title)
            })

        i = j

    return entries


def short_desc(desc: str, n=170):
    if len(desc) <= n:
        return desc
    return desc[:n] + "..."


def wants_permessi_list(q: str) -> bool:

    q = q.lower()

    triggers = [
        "a quali permessi ha diritto",
        "a quali permessi hanno diritto",
        "quali permessi",
        "elenco permessi",
        "lista permessi",
        "tutti i permessi",
        "permessi disponibili",
    ]

    return any(t in q for t in triggers)


def search_permesso(query: str, entries):

    q_norm = normalize_key(query)
    q_low = query.lower()

    for e in entries:
        if e["norm"] == q_norm:
            return e

    for e in entries:
        if e["title"].lower() in q_low:
            return e

    for e in entries:
        if q_norm in e["norm"] or e["norm"] in q_norm:
            return e

    words = [w for w in q_low.split() if len(w) >= 4]

    if words:
        for e in entries:
            text = e["desc"].lower()
            if all(w in text for w in words[:2]):
                return e

    return None


# ============================================================
# SECRETS
# ============================================================

def get_secret(key: str):
    try:
        return st.secrets[key]
    except Exception:
        return os.getenv(key)


OPENAI_API_KEY = get_secret("OPENAI_API_KEY")
UILCOM_PASSWORD = get_secret("UILCOM_PASSWORD")


# ============================================================
# AUTH
# ============================================================

if "auth_ok" not in st.session_state:
    st.session_state.auth_ok = False

if UILCOM_PASSWORD:

    pwd = st.text_input("Password UILCOM", type="password")

    if st.button("Accedi"):

        if pwd == UILCOM_PASSWORD:
            st.session_state.auth_ok = True
        else:
            st.error("Password errata")

if not st.session_state.auth_ok:
    st.stop()


# ============================================================
# SESSION
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "ipzs_permessi" not in st.session_state:
    st.session_state.ipzs_permessi = load_ipzs_permessi(IPZS_PERMESSI_PATH)


# ============================================================
# CHAT DISPLAY
# ============================================================

for msg in st.session_state.messages:

    if msg["role"] == "user":
        st.markdown(f'<div class="chat-message user-msg">{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-message bot-msg">{msg["content"]}</div>', unsafe_allow_html=True)


# ============================================================
# INPUT
# ============================================================

user_input = st.chat_input("Scrivi una domanda...")

if not user_input:
    st.stop()


# ============================================================
# ROUTER PERMESSI IPZS
# ============================================================

entries = st.session_state.ipzs_permessi

if entries and wants_permessi_list(user_input):

    risposta = "📋 **Permessi disponibili per i lavoratori IPZS:**\n\n"

    for i, e in enumerate(entries, 1):
        risposta += f"**{i}. {e['title']}** — {short_desc(e['desc'])}\n\n"

    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.messages.append({"role": "assistant", "content": risposta})

    st.rerun()


if entries:

    hit = search_permesso(user_input, entries)

    if hit:

        risposta = f"🏭 **{hit['title']}**\n\n{hit['desc']}"

        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.messages.append({"role": "assistant", "content": risposta})

        st.rerun()


# ============================================================
# CCNL RAG
# ============================================================

if not os.path.exists(VEC_PATH):

    st.warning("Indicizza prima il CCNL.")

    if st.button("Indicizza CCNL"):

        loader = PyPDFLoader(PDF_PATH)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )

        chunks = splitter.split_documents(docs)

        texts = [c.page_content for c in chunks]
        pages = [(int(c.metadata.get("page", 0)) + 1) for c in chunks]

        emb = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

        vectors = emb.embed_documents(texts)
        vectors = np.array(vectors, dtype=np.float32)

        os.makedirs(INDEX_DIR, exist_ok=True)

        np.save(VEC_PATH, vectors)

        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump(
                [{"page": p, "text": t} for p, t in zip(pages, texts)],
                f,
                ensure_ascii=False,
            )

        st.success("Indicizzazione completata")

    st.stop()


vectors = np.load(VEC_PATH)

with open(META_PATH, "r", encoding="utf-8") as f:
    meta = json.load(f)


def normalize_rows(mat):
    return mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)


mat_norm = normalize_rows(vectors)

emb = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

qvec = np.array(emb.embed_query(user_input), dtype=np.float32)
qvec = qvec / (np.linalg.norm(qvec) + 1e-12)

scores = mat_norm @ qvec

top_idx = np.argsort(-scores)[:10]

selected = [meta[i] for i in top_idx]

context = "\n\n".join([c["text"] for c in selected])


llm = ChatOpenAI(
    model=LLM_MODEL,
    temperature=LLM_TEMPERATURE,
    api_key=OPENAI_API_KEY
)

prompt = f"""
Rispondi usando SOLO il contesto CCNL.

Domanda:
{user_input}

Contesto:
{context}
"""

response = llm.invoke(prompt).content


st.session_state.messages.append({"role": "user", "content": user_input})
st.session_state.messages.append({"role": "assistant", "content": response})

st.rerun()
