# app.py — Assistente Contrattuale UILCOM IPZS (CCNL + IPZS Permessi)
# ✅ UI stile ChatGPT (pulita, chiara) + sidebar apribile (tasto ☰ visibile)
# ✅ CCNL con citazioni pagine
# ✅ IPZS Permessi (RAO/ROL ecc) + elenco completo quando richiesto
# ✅ Guardrail HARD: se retrieval debole -> "Non ho trovato..."
# ✅ Admin: reindicizza CCNL + Permessi + debug chunk/pagine usate
# ✅ Feedback 👍/👎 + log richieste
# ✅ Auto-apprendimento controllato: Q/A approvate indicizzate (admin)

import os
import re
import json
import time
import hashlib
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# Optional (precision boost): rank-bm25
try:
    from rank_bm25 import BM25Okapi  # type: ignore
    BM25_AVAILABLE = True
except Exception:
    BM25_AVAILABLE = False


# ============================================================
# CONFIG
# ============================================================
APP_TITLE = "🟦 Assistente Contrattuale UILCOM IPZS"

DOC_DIR = "documenti"
PDF_PATH = os.path.join(DOC_DIR, "ccnl.pdf")
IPZS_PERMESSI_PATH = os.path.join(DOC_DIR, "PERMESSI_IPZS_COMPLETO_FINALE.txt")

# Logo (opzionale)
LOGO_PATH_1 = os.path.join("logo", "logo_uilcom.png")   # consigliato
LOGO_PATH_2 = "logo_uilcom.png"                         # fallback se lo metti in root

# Storage
STORAGE_DIR = "storage"
LOGS_FILE = os.path.join(STORAGE_DIR, "logs.jsonl")
FEEDBACK_FILE = os.path.join(STORAGE_DIR, "feedback.jsonl")
APPROVED_QA_FILE = os.path.join(STORAGE_DIR, "approved_qa.jsonl")

# Indici
INDEX_CCNL_DIR = "index_ccnl"
CCNL_VEC_PATH = os.path.join(INDEX_CCNL_DIR, "vectors.npy")
CCNL_META_PATH = os.path.join(INDEX_CCNL_DIR, "chunks.json")

INDEX_IPZS_DIR = "index_ipzs_permessi"
IPZS_VEC_PATH = os.path.join(INDEX_IPZS_DIR, "vectors.npy")
IPZS_META_PATH = os.path.join(INDEX_IPZS_DIR, "chunks.json")

INDEX_APPROVED_DIR = "index_approved"
APPROVED_VEC_PATH = os.path.join(INDEX_APPROVED_DIR, "vectors.npy")
APPROVED_META_PATH = os.path.join(INDEX_APPROVED_DIR, "chunks.json")

# Chunking
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150

TOP_K_PER_QUERY = 12
TOP_K_FINAL = 18
MAX_MULTI_QUERIES = 8

# Guardrail retrieval
MIN_BEST_SIMILARITY = 0.24
MIN_SELECTED_CHUNKS = 3

# Admin
MAX_EVIDENCE_LINES = 18

# LLM
LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0

# Chat memory (per evitare “si trascina” argomenti vecchi)
MAX_HISTORY_TURNS = 6  # user+assistant, ultimi turni


# ============================================================
# SECRETS
# ============================================================
def get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    try:
        if key in st.secrets:  # type: ignore
            return str(st.secrets[key])  # type: ignore
    except Exception:
        pass
    return os.getenv(key, default)

def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

UILCOM_PASSWORD = get_secret("UILCOM_PASSWORD")        # password iscritti
ADMIN_PASSWORD = get_secret("ADMIN_PASSWORD")          # password admin
OPENAI_API_KEY = get_secret("OPENAI_API_KEY")          # obbligatoria


# ============================================================
# FILE IO (jsonl)
# ============================================================
def ensure_dirs():
    os.makedirs(STORAGE_DIR, exist_ok=True)
    os.makedirs(INDEX_CCNL_DIR, exist_ok=True)
    os.makedirs(INDEX_IPZS_DIR, exist_ok=True)
    os.makedirs(INDEX_APPROVED_DIR, exist_ok=True)
    os.makedirs(DOC_DIR, exist_ok=True)

def append_jsonl(path: str, obj: Dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def read_jsonl(path: str, limit: int = 2000) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out

def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


# ============================================================
# UI SETUP (ChatGPT-like) — IMPORTANT: NON NASCONDIAMO L'HEADER
# ============================================================
ensure_dirs()
st.set_page_config(
    page_title="Assistente UILCOM IPZS",
    page_icon="🟦",
    layout="centered",
    initial_sidebar_state="collapsed",  # tasto ☰ visibile, sidebar chiusa di default
)

# CSS: NON nascondere l'header (altrimenti sparisce il bottone ☰)
st.markdown(
    """
    <style>
      .block-container { max-width: 900px; padding-top: 1.2rem; }

      /* NON TOCCARE header: lì c'è il pulsante ☰ della sidebar su mobile */

      .stChatMessage { padding: 0.15rem 0; }
      div.stButton>button { border-radius: 10px; padding: 0.45rem 0.8rem; }
      section[data-testid="stSidebar"] { padding-top: 0.8rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Logo (non deve mai rompere l'app)
logo_to_use = LOGO_PATH_1 if os.path.exists(LOGO_PATH_1) else (LOGO_PATH_2 if os.path.exists(LOGO_PATH_2) else None)
if logo_to_use:
    try:
        st.image(logo_to_use, width=140)
    except Exception:
        pass

st.title(APP_TITLE)
st.caption(
    "Strumento informativo per consultare **CCNL** e **IPZS Permessi/Giustificativi**. "
    "Risposte basate solo sui documenti indicizzati. Per casi specifici → RSU/UILCOM o HR."
)
st.divider()


# ============================================================
# AUTH: ISCRITTI
# ============================================================
if "auth_ok" not in st.session_state:
    st.session_state.auth_ok = False

if UILCOM_PASSWORD:
    with st.expander("🔒 Accesso iscritti UILCOM", expanded=not st.session_state.auth_ok):
        pwd_in = st.text_input("Password iscritti", type="password", placeholder="Inserisci password iscritti")
        if st.button("Entra", use_container_width=True):
            if pwd_in == UILCOM_PASSWORD:
                st.session_state.auth_ok = True
                st.success("Accesso consentito.")
            else:
                st.session_state.auth_ok = False
                st.error("Password non corretta.")
else:
    st.warning("Password iscritti non impostata. Imposta UILCOM_PASSWORD in Secrets.")
    # st.session_state.auth_ok = True  # per sviluppo locale

if not st.session_state.auth_ok:
    st.stop()


# ============================================================
# ADMIN SIDEBAR
# ============================================================
if "is_admin" not in st.session_state:
    st.session_state.is_admin = False

if "chat_id" not in st.session_state:
    st.session_state.chat_id = sha256_text(str(time.time()))[:12]

# ============================================================
# CACHED RESOURCES
# ============================================================
@st.cache_resource(show_spinner=False)
def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(api_key=OPENAI_API_KEY)

@st.cache_resource(show_spinner=False)
def get_llm() -> ChatOpenAI:
    return ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE, api_key=OPENAI_API_KEY)


# ============================================================
# HELPERS: VECTOR SEARCH (ottimizzati)
# ============================================================
def normalize_rows(mat: np.ndarray) -> np.ndarray:
    return mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)

def cosine_scores(query_vec: np.ndarray, mat_norm: np.ndarray) -> np.ndarray:
    q = query_vec / (np.linalg.norm(query_vec) + 1e-12)
    return mat_norm @ q

@st.cache_data(show_spinner=False)
def load_index_cached(vec_path: str, meta_path: str) -> Tuple[np.ndarray, List[Dict[str, Any]], np.ndarray]:
    vectors = np.load(vec_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta_raw = json.load(f)

    meta: List[Dict[str, Any]] = []
    for item in meta_raw:
        if isinstance(item, dict) and "text" in item:
            meta.append({"page": item.get("page", "?"), "text": item.get("text", "")})
        else:
            meta.append({"page": "?", "text": str(item)})

    mat_norm = normalize_rows(vectors.astype(np.float32))
    return vectors.astype(np.float32), meta, mat_norm

def bm25_rerank(query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not BM25_AVAILABLE or not candidates:
        return candidates
    corpus = [(c.get("text") or "").lower().split() for c in candidates]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(query.lower().split())
    idx = np.argsort(-np.array(scores))
    return [candidates[int(i)] for i in idx]

def dedup_by_text(chunks: List[Dict[str, Any]], max_out: int) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for c in chunks:
        txt = (c.get("text") or "").strip()
        key = sha256_text(txt[:500])
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
        if len(out) >= max_out:
            break
    return out

def unique_pages_ccnl(chunks: List[Dict[str, Any]], max_pages: int = 8) -> List[int]:
    pages: List[int] = []
    for c in chunks:
        try:
            p = int(c.get("page", 0))
        except Exception:
            continue
        if p and p not in pages:
            pages.append(p)
        if len(pages) >= max_pages:
            break
    return pages

def format_ccnl_citations(pages: List[int]) -> str:
    if not pages:
        return ""
    pages_sorted = sorted(pages)
    if len(pages_sorted) == 1:
        return f"**Fonte:** CCNL (pag. {pages_sorted[0]})"
    return f"**Fonte:** CCNL (pagg. {', '.join(map(str, pages_sorted))})"

def extract_evidence(chunks: List[Dict[str, Any]], max_lines: int = 18) -> List[str]:
    patterns = [
        r"rao", r"r\.a\.o", r"rol", r"r\.o\.l", r"riduzion",
        r"permess", r"conged", r"retribuit", r"non retribuit",
        r"donazion", r"decesso", r"lutto", r"studio", r"esam", r"104",
        r"cure termali", r"elettoral", r"testimonianza",
        r"inquadrament", r"livell", r"categoria", r"mansioni superiori",
    ]
    out = []
    for c in chunks:
        page = c.get("page", "?")
        text = c.get("text", "") or ""
        for ln in [x.strip() for x in text.splitlines() if x.strip()]:
            low = ln.lower()
            if any(re.search(p, low) for p in patterns):
                clean = " ".join(ln.split())
                if 20 <= len(clean) <= 420:
                    out.append(f"({page}) {clean}")
            if len(out) >= max_lines:
                return out
    return out[:max_lines]


# ============================================================
# IPZS PERMESSI: PARSER ELENCO COMPLETO (deterministico)
# ============================================================
def parse_ipzs_permessi_sections(raw_text: str) -> List[Dict[str, str]]:
    lines = raw_text.splitlines()
    sections: List[Dict[str, str]] = []

    def is_title(line: str) -> bool:
        s = line.strip()
        if not s:
            return False
        if set(s) <= set("-_= "):
            return False
        if s in ["RAO", "R.O.L.", "ROL", "R.A.O."]:
            return True
        letters = [ch for ch in s if ch.isalpha()]
        if len(letters) >= 6 and sum(ch.isupper() for ch in letters) / max(len(letters), 1) > 0.8:
            return True
        if re.search(r"\bL\.\s*\d+/\d+\b", s):
            return True
        return False

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if is_title(line):
            title = line
            j = i + 1
            while j < len(lines) and (lines[j].strip() == "" or set(lines[j].strip()) <= set("-_= ")):
                j += 1

            desc_lines = []
            if j < len(lines) and lines[j].strip().lower().startswith("descrizione"):
                desc_lines.append(lines[j].strip())
                j += 1

            while j < len(lines):
                nxt = lines[j].strip()
                if is_title(nxt):
                    break
                desc_lines.append(lines[j].rstrip())
                j += 1

            desc = "\n".join([x for x in desc_lines if x.strip()]).strip()
            if desc:
                sections.append({"title": title.strip(), "desc": desc})
            i = j
        else:
            i += 1

    out = []
    seen = set()
    for s in sections:
        t = re.sub(r"\s+", " ", s["title"]).strip()
        key = t.lower()
        if key not in seen:
            out.append({"title": t, "desc": s["desc"]})
            seen.add(key)
    return out

def ipzs_source_line() -> str:
    return "**Fonte:** Documento IPZS Permessi/Giustificativi (PERMESSI_IPZS_COMPLETO_FINALE.txt)"

def is_generic_permessi_list_question(q: str) -> bool:
    ql = q.lower()
    triggers = [
        "a quali permessi ha diritto",
        "quali permessi ha diritto",
        "quali permessi ha a disposizione",
        "elenco permessi",
        "tutti i permessi",
        "lista permessi",
        "permessi disponibili",
        "permessi ipzs",
        "giustificativi",
    ]
    return any(t in ql for t in triggers)

def extract_specific_permesso_name(q: str, sections_titles: List[str]) -> Optional[str]:
    ql = q.lower()
    if re.search(r"\brao\b|\br\.a\.o\b", ql):
        return "RAO"
    if re.search(r"\brol\b|\br\.o\.l\b", ql):
        return "R.O.L."

    for t in sections_titles:
        tl = t.lower()
        if len(tl) < 4:
            continue
        if tl in ql:
            return t

    keywords_map = {
        "donazione": ["DONAZIONE SANGUE", "DONAZIONE"],
        "sangue": ["DONAZIONE SANGUE"],
        "cure termali": ["CURE TERMALI"],
        "elettoral": ["PERMESSI ELETTORALI"],
        "decesso": ["DECESSO", "DECESSO FAMILIARI"],
        "testimon": ["TESTIMONIANZA CIVILE"],
        "esami": ["PERM. ESAMI", "ESAMI"],
        "104": ["L. 104/92", "L.104/92", "104"],
        "maternit": ["MATERNITA' OBBLIGATORIA", "MATERNITA'"],
        "padre": ["CONG.OBBLIGATORIO PADRE"],
        "nido": ["INSERIM.NIDO", "ACC.NIDO", "NIDO"],
    }
    for k, cand in keywords_map.items():
        if k in ql:
            for c in cand:
                for real in sections_titles:
                    if c.lower() in real.lower() or real.lower() in c.lower():
                        return real
    return None


# ============================================================
# ROUTING: IPZS vs CCNL (più solido)
# ============================================================
def detect_route(q: str) -> str:
    ql = q.lower()
    if is_generic_permessi_list_question(ql):
        return "IPZS_LIST"
    if any(x in ql for x in ["rao", "r.a.o", "rol", "r.o.l", "donazione", "cure termali",
                             "permessi elettorali", "testimonianza", "decesso", "104", "esami", "maternit", "nido"]):
        return "IPZS_SINGLE"
    if "permess" in ql or "giustific" in ql:
        return "MIXED"
    return "CCNL"


# ============================================================
# MULTI-QUERY BUILDER
# ============================================================
def build_queries(q: str) -> List[str]:
    q0 = q.strip()
    qs = [
        q0,
        f"{q0} CCNL",
        f"{q0} definizione",
        f"{q0} procedura",
        f"{q0} requisiti",
    ]
    out, seen = [], set()
    for x in qs:
        x = x.strip()
        if x and x not in seen:
            out.append(x)
            seen.add(x)
    return out[:MAX_MULTI_QUERIES]


# ============================================================
# RETRIEVE (generic)
# ============================================================
def retrieve_from_index(
    query: str,
    vec_path: str,
    meta_path: str,
    emb: OpenAIEmbeddings,
    top_k_per_query: int = TOP_K_PER_QUERY,
    top_k_final: int = TOP_K_FINAL,
) -> Tuple[List[Dict[str, Any]], float, List[str], List[str]]:
    _, meta, mat_norm = load_index_cached(vec_path, meta_path)

    queries = build_queries(query)
    scores_best: Dict[int, float] = {}
    best_similarity = 0.0

    for q in queries:
        qvec = np.array(emb.embed_query(q), dtype=np.float32)
        sims = cosine_scores(qvec, mat_norm)
        top_idx = np.argsort(-sims)[:top_k_per_query]
        for i in top_idx:
            s = float(sims[int(i)])
            best_similarity = max(best_similarity, s)
            if (int(i) not in scores_best) or (s > scores_best[int(i)]):
                scores_best[int(i)] = s

    final_idx = sorted(scores_best.keys(), key=lambda i: scores_best[i], reverse=True)[:top_k_final]
    selected = [meta[i] for i in final_idx]
    selected = bm25_rerank(query, selected)
    selected = dedup_by_text(selected, max_out=top_k_final)

    evidence = extract_evidence(selected, max_lines=MAX_EVIDENCE_LINES)
    return selected, best_similarity, queries, evidence


def hard_not_found_ccnl() -> str:
    return "Non ho trovato la risposta nel CCNL caricato."

def hard_not_found_generic() -> str:
    return "Non ho trovato la risposta nei documenti caricati."


# ============================================================
# CCNL LLM PROMPT
# ============================================================
RULES_CCNL = """
Sei l’assistente UILCOM per lavoratori IPZS.
Rispondi in modo chiaro e pratico basandoti SOLO sul contesto (estratti CCNL) fornito.
Non inventare informazioni.

REGOLE:
1) Se non trovi nel contesto, scrivi: "Non ho trovato la risposta nel CCNL caricato."
2) Chiudi SEMPRE con: "Fonte: CCNL (pag. ...)" usando solo le pagine presenti nel contesto.
3) Se l’utente chiede permessi ma nel contesto non ci sono dettagli, non inventare.
"""

def split_public(text: str) -> str:
    m = re.search(r"<PUBLIC>(.*?)</PUBLIC>", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return text.strip()


# ============================================================
# INDICIZZAZIONE (CCNL + IPZS + Q/A approvate)
# ============================================================
def do_reindex_all():
    emb = get_embeddings()
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    # ---- CCNL PDF ----
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"Non trovo il PDF CCNL: {PDF_PATH}")

    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()
    ccnl_chunks = splitter.split_documents(docs)

    ccnl_texts = [c.page_content for c in ccnl_chunks]
    ccnl_pages = [(int(c.metadata.get("page", 0)) + 1) for c in ccnl_chunks]  # 1-based

    ccnl_vectors = np.array(emb.embed_documents(ccnl_texts), dtype=np.float32)
    np.save(CCNL_VEC_PATH, ccnl_vectors)
    with open(CCNL_META_PATH, "w", encoding="utf-8") as f:
        json.dump([{"page": p, "text": t} for p, t in zip(ccnl_pages, ccnl_texts)], f, ensure_ascii=False)

    # ---- IPZS Permessi TXT ----
    if not os.path.exists(IPZS_PERMESSI_PATH):
        raise FileNotFoundError(f"Non trovo IPZS Permessi: {IPZS_PERMESSI_PATH}")

    with open(IPZS_PERMESSI_PATH, "r", encoding="utf-8", errors="ignore") as f:
        ipzs_raw = f.read()

    ipzs_chunks = splitter.split_text(ipzs_raw)
    ipzs_vectors = np.array(emb.embed_documents(ipzs_chunks), dtype=np.float32)
    np.save(IPZS_VEC_PATH, ipzs_vectors)
    with open(IPZS_META_PATH, "w", encoding="utf-8") as f:
        json.dump([{"page": "IPZS", "text": t} for t in ipzs_chunks], f, ensure_ascii=False)

    # ---- Q/A approvate (autoapprendimento controllato) ----
    approved_rows = read_jsonl(APPROVED_QA_FILE, limit=5000)
    qa_texts: List[str] = []
    qa_meta: List[Dict[str, Any]] = []

    for r in approved_rows:
        q = (r.get("question") or "").strip()
        a = (r.get("answer") or "").strip()
        if not q or not a:
            continue
        txt = f"DOMANDA: {q}\nRISPOSTA APPROVATA: {a}"
        qa_texts.append(txt)
        qa_meta.append({"page": "APPROVED", "text": txt})

    if qa_texts:
        qa_vec = np.array(emb.embed_documents(qa_texts), dtype=np.float32)
        np.save(APPROVED_VEC_PATH, qa_vec)
        with open(APPROVED_META_PATH, "w", encoding="utf-8") as f:
            json.dump(qa_meta, f, ensure_ascii=False)
    else:
        if os.path.exists(APPROVED_VEC_PATH):
            os.remove(APPROVED_VEC_PATH)
        if os.path.exists(APPROVED_META_PATH):
            os.remove(APPROVED_META_PATH)


# ============================================================
# SIDEBAR CONTENT (ora il tasto ☰ si vede sempre)
# ============================================================
with st.sidebar:
    st.header("⚙️ Controlli")

    if st.button("🧹 Nuova chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_id = sha256_text(str(time.time()))[:12]
        st.rerun()

    st.divider()
    st.subheader("📦 Indici documenti")

    ok_ccnl = os.path.exists(CCNL_VEC_PATH) and os.path.exists(CCNL_META_PATH)
    ok_ipzs = os.path.exists(IPZS_VEC_PATH) and os.path.exists(IPZS_META_PATH)
    ok_approved = os.path.exists(APPROVED_VEC_PATH) and os.path.exists(APPROVED_META_PATH)

    st.write("Indice CCNL:", "✅" if ok_ccnl else "❌")
    st.write("Indice IPZS Permessi:", "✅" if ok_ipzs else "❌")
    st.write("Indice Q/A approvate:", "✅" if ok_approved else "❌")

    st.caption("Se aggiorni documenti o Q/A approvate → reindicizza.")

    st.divider()
    st.subheader("🧠 Admin")
    if ADMIN_PASSWORD:
        admin_in = st.text_input("Password admin", type="password", placeholder="Solo admin UILCOM", key="admin_pwd")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Login", use_container_width=True):
                st.session_state.is_admin = (admin_in == ADMIN_PASSWORD)
                if st.session_state.is_admin:
                    st.success("Admin attivo.")
                else:
                    st.error("Password errata.")
        with c2:
            if st.button("Logout", use_container_width=True):
                st.session_state.is_admin = False
    else:
        st.caption("ADMIN_PASSWORD non impostata (Secrets).")

    st.divider()
    if st.button("Indicizza / Reindicizza (CCNL + IPZS + Q/A)", use_container_width=True):
        if not OPENAI_API_KEY:
            st.error("Manca OPENAI_API_KEY nei Secrets.")
        else:
            try:
                with st.spinner("Indicizzazione in corso..."):
                    do_reindex_all()
                st.success("Indicizzazione completata ✅")
                st.cache_data.clear()
                st.cache_resource.clear()
                st.rerun()
            except Exception as e:
                st.error(str(e))

    if st.session_state.is_admin:
        with st.expander("📄 Log & Feedback", expanded=False):
            if st.button("Mostra ultimi log", use_container_width=True):
                st.write(read_jsonl(LOGS_FILE, limit=200)[-30:])
            if st.button("Mostra ultimi feedback", use_container_width=True):
                st.write(read_jsonl(FEEDBACK_FILE, limit=200)[-30:])

        with st.expander("✅ Auto-apprendimento controllato (Q/A)", expanded=False):
            st.caption("Incolla domanda+risposta che vuoi rendere 'conoscenza approvata'. Poi reindicizza.")
            q = st.text_area("Domanda approvata", height=80, key="qa_q")
            a = st.text_area("Risposta approvata", height=120, key="qa_a")
            if st.button("Salva Q/A approvata", use_container_width=True):
                if q.strip() and a.strip():
                    append_jsonl(APPROVED_QA_FILE, {"ts": now_iso(), "question": q.strip(), "answer": a.strip()})
                    st.success("Salvata. Ora premi Reindicizza per renderla attiva.")
                else:
                    st.warning("Inserisci sia domanda che risposta.")

    st.caption("Suggerimento: su mobile apri la sidebar con ☰ in alto a sinistra.")


# ============================================================
# HARD FAIL IF NO OPENAI KEY
# ============================================================
if not OPENAI_API_KEY:
    st.error(
        "Manca la variabile **OPENAI_API_KEY**.\n\n"
        "Streamlit Cloud: **Settings → Secrets → OPENAI_API_KEY**"
    )
    st.stop()


# ============================================================
# CHAT STATE
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

def trimmed_history(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if len(messages) <= MAX_HISTORY_TURNS * 2:
        return messages
    return messages[-MAX_HISTORY_TURNS * 2:]


# Render chat history
for i, m in enumerate(st.session_state.messages):
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

        if m["role"] == "assistant" and m.get("citations"):
            st.caption(m["citations"])

        if m["role"] == "assistant":
            cols = st.columns(10)
            with cols[0]:
                if st.button("👍", key=f"up_{i}"):
                    append_jsonl(FEEDBACK_FILE, {
                        "ts": now_iso(),
                        "chat_id": st.session_state.chat_id,
                        "msg_index": i,
                        "rating": "up",
                        "question": m.get("question", ""),
                        "answer": m.get("content", ""),
                    })
                    st.toast("Feedback 👍 salvato")
            with cols[1]:
                if st.button("👎", key=f"down_{i}"):
                    append_jsonl(FEEDBACK_FILE, {
                        "ts": now_iso(),
                        "chat_id": st.session_state.chat_id,
                        "msg_index": i,
                        "rating": "down",
                        "question": m.get("question", ""),
                        "answer": m.get("content", ""),
                    })
                    st.toast("Feedback 👎 salvato")

        if st.session_state.is_admin and m["role"] == "assistant":
            dbg = m.get("debug")
            if dbg:
                with st.expander("🧠 Admin: debug", expanded=False):
                    st.write("**Route:**", dbg.get("route"))
                    st.write("**Best similarity:**", dbg.get("best_similarity"))
                    st.write("**Query usate:**")
                    st.code("\n".join(dbg.get("queries", [])) or "(nessuna)")
                    st.write("**Evidenze:**")
                    st.code("\n".join(dbg.get("evidence", [])) or "(nessuna)")
                    if dbg.get("pages_ccnl"):
                        st.write("**Pagine CCNL:**", dbg.get("pages_ccnl"))
                    st.write("**Chunk selezionati (prime righe):**")
                    for c in (dbg.get("selected") or [])[:4]:
                        st.write(f"**Pagina/Origine: {c.get('page')}**")
                        txt = (c.get("text") or "")
                        st.write(txt[:700] + ("..." if len(txt) > 700 else ""))
                        st.divider()


user_input = st.chat_input("Scrivi una domanda (CCNL / permessi IPZS: RAO, ROL, donazione sangue, ecc.)")
if not user_input:
    st.stop()

user_input = user_input.strip()
if user_input.lower() in ("/nuova", "/reset", "/new"):
    st.session_state.messages = []
    st.session_state.chat_id = sha256_text(str(time.time()))[:12]
    st.rerun()


# Require indexes
need_ccnl = not (os.path.exists(CCNL_VEC_PATH) and os.path.exists(CCNL_META_PATH))
need_ipzs = not (os.path.exists(IPZS_VEC_PATH) and os.path.exists(IPZS_META_PATH))
have_approved = os.path.exists(APPROVED_VEC_PATH) and os.path.exists(APPROVED_META_PATH)

if need_ccnl or need_ipzs:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Prima devo indicizzare i documenti: apri la barra laterale (☰) e clicca **Indicizza / Reindicizza (CCNL + IPZS + Q/A)**.",
        "question": user_input,
    })
    st.rerun()

# Log domanda
append_jsonl(LOGS_FILE, {"ts": now_iso(), "chat_id": st.session_state.chat_id, "event": "question", "text": user_input})

st.session_state.messages.append({"role": "user", "content": user_input})

emb = get_embeddings()
route = detect_route(user_input)


# ============================================================
# IPZS LIST
# ============================================================
if route == "IPZS_LIST":
    try:
        with open(IPZS_PERMESSI_PATH, "r", encoding="utf-8", errors="ignore") as f:
            ipzs_raw = f.read()
        sections = parse_ipzs_permessi_sections(ipzs_raw)

        if not sections:
            public_ans = hard_not_found_generic()
        else:
            lines = []
            lines.append("Ecco l’**elenco dei permessi/giustificativi** presenti nel documento IPZS, con una breve spiegazione:")
            lines.append("")
            for s in sections:
                title = s["title"]
                desc = s["desc"]
                desc_short = []
                for ln in desc.splitlines():
                    ln = ln.strip()
                    if not ln:
                        continue
                    if ln.lower() == "descrizione:":
                        continue
                    desc_short.append(ln)
                    if len(desc_short) >= 2:
                        break
                preview = " ".join(desc_short).strip() or "Vedi descrizione nel documento."
                lines.append(f"- **{title}**: {preview}")

            lines.append("")
            lines.append(ipzs_source_line())
            public_ans = "\n".join(lines).strip()

        assistant_payload = {
            "role": "assistant",
            "content": public_ans,
            "citations": ipzs_source_line(),
            "question": user_input,
        }

        st.session_state.messages.append(assistant_payload)
        append_jsonl(LOGS_FILE, {"ts": now_iso(), "chat_id": st.session_state.chat_id, "event": "answer", "route": route})
        st.rerun()

    except Exception:
        assistant_payload = {"role": "assistant", "content": hard_not_found_generic(), "question": user_input}
        st.session_state.messages.append(assistant_payload)
        st.rerun()


# ============================================================
# IPZS SINGLE / MIXED
# ============================================================
def answer_from_ipzs(user_q: str) -> Tuple[Optional[str], Dict[str, Any]]:
    selected_ipzs, best_ipzs, queries_ipzs, evidence_ipzs = retrieve_from_index(
        user_q, IPZS_VEC_PATH, IPZS_META_PATH, emb
    )
    retrieval_ok_ipzs = (best_ipzs >= MIN_BEST_SIMILARITY) and (len(selected_ipzs) >= MIN_SELECTED_CHUNKS)

    try:
        with open(IPZS_PERMESSI_PATH, "r", encoding="utf-8", errors="ignore") as f:
            ipzs_raw = f.read()
        sections = parse_ipzs_permessi_sections(ipzs_raw)
        titles = [s["title"] for s in sections]
        wanted = extract_specific_permesso_name(user_q, titles)
    except Exception:
        sections, wanted = [], None

    if not retrieval_ok_ipzs:
        return None, {
            "route": "IPZS",
            "best_similarity": best_ipzs,
            "queries": queries_ipzs,
            "evidence": evidence_ipzs,
            "selected": selected_ipzs[:6],
            "note": "retrieval_ok=False",
        }

    if sections and wanted:
        match = None
        if wanted == "RAO":
            for s in sections:
                if s["title"].strip().lower() in ["rao", "r.a.o."] or "rao" in s["title"].lower():
                    match = s
                    break
        elif wanted in ["R.O.L.", "ROL"]:
            for s in sections:
                if "rol" in s["title"].lower() or "r.o.l" in s["title"].lower():
                    match = s
                    break
        else:
            for s in sections:
                if s["title"].lower() == wanted.lower() or wanted.lower() in s["title"].lower():
                    match = s
                    break

        if match:
            public_ans = f"**{match['title']}**\n\n{match['desc'].strip()}\n\n{ipzs_source_line()}"
            return public_ans, {
                "route": "IPZS",
                "best_similarity": best_ipzs,
                "queries": queries_ipzs,
                "evidence": evidence_ipzs,
                "selected": selected_ipzs[:6],
                "note": f"Permesso specifico richiesto: {wanted}",
            }

    context = "\n\n---\n\n".join([c.get("text", "") for c in selected_ipzs[:5]])
    public_ans = f"Ho trovato riferimenti nel documento IPZS:\n\n{context}\n\n{ipzs_source_line()}"
    return public_ans, {
        "route": "IPZS",
        "best_similarity": best_ipzs,
        "queries": queries_ipzs,
        "evidence": evidence_ipzs,
        "selected": selected_ipzs[:6],
        "note": "Fallback su chunk IPZS",
    }

if route in ["IPZS_SINGLE", "MIXED"]:
    ipzs_ans, _ = answer_from_ipzs(user_input)
    if ipzs_ans:
        assistant_payload = {
            "role": "assistant",
            "content": ipzs_ans,
            "citations": ipzs_source_line(),
            "question": user_input,
        }
        st.session_state.messages.append(assistant_payload)
        append_jsonl(LOGS_FILE, {"ts": now_iso(), "chat_id": st.session_state.chat_id, "event": "answer", "route": "IPZS"})
        st.rerun()
    if route != "MIXED":
        assistant_payload = {"role": "assistant", "content": hard_not_found_generic(), "question": user_input}
        st.session_state.messages.append(assistant_payload)
        st.rerun()


# ============================================================
# CCNL (+ Q/A approvate come booster)
# ============================================================
selected_ccnl, best_ccnl, queries_ccnl, evidence_ccnl = retrieve_from_index(
    user_input, CCNL_VEC_PATH, CCNL_META_PATH, emb
)
retrieval_ok_ccnl = (best_ccnl >= MIN_BEST_SIMILARITY) and (len(selected_ccnl) >= MIN_SELECTED_CHUNKS)

approved_dbg = None
if have_approved:
    selected_qa, best_qa, queries_qa, evidence_qa = retrieve_from_index(
        user_input, APPROVED_VEC_PATH, APPROVED_META_PATH, emb,
        top_k_per_query=8, top_k_final=10
    )
    if best_qa >= max(MIN_BEST_SIMILARITY, 0.28):
        selected_ccnl = dedup_by_text(selected_qa + selected_ccnl, max_out=TOP_K_FINAL)
        evidence_ccnl = extract_evidence(selected_ccnl, max_lines=MAX_EVIDENCE_LINES)
        approved_dbg = {"best_qa": best_qa, "queries_qa": queries_qa, "evidence_qa": evidence_qa}

if not retrieval_ok_ccnl:
    assistant_payload = {"role": "assistant", "content": hard_not_found_ccnl(), "question": user_input}
    st.session_state.messages.append(assistant_payload)
    append_jsonl(LOGS_FILE, {"ts": now_iso(), "chat_id": st.session_state.chat_id, "event": "answer", "route": "CCNL_BLOCKED"})
    st.rerun()

pages = unique_pages_ccnl(selected_ccnl, max_pages=8)
cit_line = format_ccnl_citations(pages)
context = "\n\n---\n\n".join([f"[Pagina {c.get('page','?')}] {c.get('text','')}" for c in selected_ccnl])

llm = get_llm()
hist = trimmed_history([m for m in st.session_state.messages if m["role"] in ("user", "assistant")])

prompt = f"""
{RULES_CCNL}

STORIA CHAT (ultimi turni, se utili):
{json.dumps(hist, ensure_ascii=False)}

DOMANDA (UTENTE):
{user_input}

CONTESTO CCNL (estratti):
{context}

SCRIVI UNA RISPOSTA CHIARA E PRATICA.
CHIUDI SEMPRE CON:
{cit_line}

FORMATO:
<PUBLIC>
...testo...
{cit_line}
</PUBLIC>
"""

try:
    raw = llm.invoke(prompt).content
except Exception as e:
    raw = f"<PUBLIC>Errore nel generare la risposta: {e}\n\n{cit_line}</PUBLIC>"

public_ans = split_public(raw).strip()
if cit_line and ("fonte" not in public_ans.lower()):
    public_ans = public_ans.rstrip() + "\n\n" + cit_line

assistant_payload = {
    "role": "assistant",
    "content": public_ans,
    "citations": cit_line,
    "question": user_input,
}

st.session_state.messages.append(assistant_payload)
append_jsonl(LOGS_FILE, {"ts": now_iso(), "chat_id": st.session_state.chat_id, "event": "answer", "route": "CCNL_OK", "pages": pages})
st.rerun()DOC_DIR = "documenti"
PDF_PATH = os.path.join(DOC_DIR, "ccnl.pdf")
IPZS_PERMESSI_PATH = os.path.join(DOC_DIR, "PERMESSI_IPZS_COMPLETO_FINALE.txt")

# Logo (opzionale)
LOGO_PATH_1 = os.path.join("logo", "logo_uilcom.png")   # consigliato
LOGO_PATH_2 = "logo_uilcom.png"                         # fallback se lo metti in root

# Storage
STORAGE_DIR = "storage"
LOGS_FILE = os.path.join(STORAGE_DIR, "logs.jsonl")
FEEDBACK_FILE = os.path.join(STORAGE_DIR, "feedback.jsonl")
APPROVED_QA_FILE = os.path.join(STORAGE_DIR, "approved_qa.jsonl")

# Indici
INDEX_CCNL_DIR = "index_ccnl"
CCNL_VEC_PATH = os.path.join(INDEX_CCNL_DIR, "vectors.npy")
CCNL_META_PATH = os.path.join(INDEX_CCNL_DIR, "chunks.json")

INDEX_IPZS_DIR = "index_ipzs_permessi"
IPZS_VEC_PATH = os.path.join(INDEX_IPZS_DIR, "vectors.npy")
IPZS_META_PATH = os.path.join(INDEX_IPZS_DIR, "chunks.json")

INDEX_APPROVED_DIR = "index_approved"
APPROVED_VEC_PATH = os.path.join(INDEX_APPROVED_DIR, "vectors.npy")
APPROVED_META_PATH = os.path.join(INDEX_APPROVED_DIR, "chunks.json")

# Chunking
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150

TOP_K_PER_QUERY = 12
TOP_K_FINAL = 18
MAX_MULTI_QUERIES = 8

# Guardrail retrieval
MIN_BEST_SIMILARITY = 0.24
MIN_SELECTED_CHUNKS = 3

# Admin
MAX_EVIDENCE_LINES = 18

# LLM
LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0

# Chat memory (per evitare “si trascina” argomenti vecchi)
MAX_HISTORY_TURNS = 6  # user+assistant, ultimi turni


# ============================================================
# SECRETS
# ============================================================
def get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    try:
        if key in st.secrets:  # type: ignore
            return str(st.secrets[key])  # type: ignore
    except Exception:
        pass
    return os.getenv(key, default)

def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

UILCOM_PASSWORD = get_secret("UILCOM_PASSWORD")        # password iscritti
ADMIN_PASSWORD = get_secret("ADMIN_PASSWORD")          # password admin
OPENAI_API_KEY = get_secret("OPENAI_API_KEY")          # obbligatoria


# ============================================================
# FILE IO (jsonl)
# ============================================================
def ensure_dirs():
    os.makedirs(STORAGE_DIR, exist_ok=True)
    os.makedirs(INDEX_CCNL_DIR, exist_ok=True)
    os.makedirs(INDEX_IPZS_DIR, exist_ok=True)
    os.makedirs(INDEX_APPROVED_DIR, exist_ok=True)
    os.makedirs(DOC_DIR, exist_ok=True)

def append_jsonl(path: str, obj: Dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def read_jsonl(path: str, limit: int = 2000) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out

def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


# ============================================================
# UI SETUP (ChatGPT-like)
# ============================================================
ensure_dirs()
st.set_page_config(page_title="Assistente UILCOM IPZS", page_icon="🟦", layout="centered")

st.markdown(
    """
    <style>
      .block-container { max-width: 900px; padding-top: 1.2rem; }
      header { visibility: hidden; height: 0; }
      .stChatMessage { padding: 0.15rem 0; }
      div.stButton>button { border-radius: 10px; padding: 0.45rem 0.8rem; }
      section[data-testid="stSidebar"] { padding-top: 0.8rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Logo (non deve mai rompere l'app)
logo_to_use = LOGO_PATH_1 if os.path.exists(LOGO_PATH_1) else (LOGO_PATH_2 if os.path.exists(LOGO_PATH_2) else None)
if logo_to_use:
    try:
        st.image(logo_to_use, width=140)
    except Exception:
        pass

st.title(APP_TITLE)
st.caption(
    "Strumento informativo per consultare **CCNL** e **IPZS Permessi/Giustificativi**. "
    "Risposte basate solo sui documenti indicizzati. Per casi specifici → RSU/UILCOM o HR."
)
st.divider()


# ============================================================
# AUTH: ISCRITTI
# ============================================================
if "auth_ok" not in st.session_state:
    st.session_state.auth_ok = False

if UILCOM_PASSWORD:
    with st.expander("🔒 Accesso iscritti UILCOM", expanded=not st.session_state.auth_ok):
        pwd_in = st.text_input("Password iscritti", type="password", placeholder="Inserisci password iscritti")
        if st.button("Entra", use_container_width=True):
            if pwd_in == UILCOM_PASSWORD:
                st.session_state.auth_ok = True
                st.success("Accesso consentito.")
            else:
                st.session_state.auth_ok = False
                st.error("Password non corretta.")
else:
    st.warning("Password iscritti non impostata. Imposta UILCOM_PASSWORD in Secrets.")
    # st.session_state.auth_ok = True  # per sviluppo locale

if not st.session_state.auth_ok:
    st.stop()


# ============================================================
# ADMIN SIDEBAR
# ============================================================
if "is_admin" not in st.session_state:
    st.session_state.is_admin = False

if "chat_id" not in st.session_state:
    st.session_state.chat_id = sha256_text(str(time.time()))[:12]

with st.sidebar:
    st.header("⚙️ Controlli")

    if st.button("🧹 Nuova chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_id = sha256_text(str(time.time()))[:12]
        st.rerun()

    st.divider()
    st.subheader("📦 Indici documenti")

    ok_ccnl = os.path.exists(CCNL_VEC_PATH) and os.path.exists(CCNL_META_PATH)
    ok_ipzs = os.path.exists(IPZS_VEC_PATH) and os.path.exists(IPZS_META_PATH)
    ok_approved = os.path.exists(APPROVED_VEC_PATH) and os.path.exists(APPROVED_META_PATH)

    st.write("Indice CCNL:", "✅" if ok_ccnl else "❌")
    st.write("Indice IPZS Permessi:", "✅" if ok_ipzs else "❌")
    st.write("Indice Q/A approvate:", "✅" if ok_approved else "❌")

    st.caption("Se aggiorni i documenti o aggiungi Q/A approvate → reindicizza.")

    # Admin login
    st.divider()
    st.subheader("🧠 Admin")
    if ADMIN_PASSWORD:
        admin_in = st.text_input("Password admin", type="password", placeholder="Solo admin UILCOM", key="admin_pwd")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Login", use_container_width=True):
                st.session_state.is_admin = (admin_in == ADMIN_PASSWORD)
                if st.session_state.is_admin:
                    st.success("Admin attivo.")
                else:
                    st.error("Password errata.")
        with c2:
            if st.button("Logout", use_container_width=True):
                st.session_state.is_admin = False
    else:
        st.caption("ADMIN_PASSWORD non impostata (Secrets).")

    st.divider()
    if st.button("Indicizza / Reindicizza (CCNL + IPZS + Q/A)", use_container_width=True):
        if not OPENAI_API_KEY:
            st.error("Manca OPENAI_API_KEY nei Secrets.")
        else:
            try:
                with st.spinner("Indicizzazione in corso..."):
                    do_reindex_all()
                st.success("Indicizzazione completata ✅")
                st.cache_data.clear()
                st.cache_resource.clear()
                st.rerun()
            except Exception as e:
                st.error(str(e))

    if st.session_state.is_admin:
        with st.expander("📄 Log & Feedback", expanded=False):
            if st.button("Mostra ultimi log", use_container_width=True):
                st.write(read_jsonl(LOGS_FILE, limit=200)[-30:])
            if st.button("Mostra ultimi feedback", use_container_width=True):
                st.write(read_jsonl(FEEDBACK_FILE, limit=200)[-30:])

        with st.expander("✅ Auto-apprendimento controllato (Q/A)", expanded=False):
            st.caption("Incolla domanda+risposta che vuoi rendere 'conoscenza approvata'. Poi reindicizza.")
            q = st.text_area("Domanda approvata", height=80, key="qa_q")
            a = st.text_area("Risposta approvata", height=120, key="qa_a")
            if st.button("Salva Q/A approvata", use_container_width=True):
                if q.strip() and a.strip():
                    append_jsonl(APPROVED_QA_FILE, {"ts": now_iso(), "question": q.strip(), "answer": a.strip()})
                    st.success("Salvata. Ora premi Reindicizza per renderla attiva.")
                else:
                    st.warning("Inserisci sia domanda che risposta.")

    st.caption("Suggerimento: dopo update su GitHub → Streamlit Cloud fa auto-redeploy.")


# ============================================================
# HARD FAIL IF NO OPENAI KEY
# ============================================================
if not OPENAI_API_KEY:
    st.error(
        "Manca la variabile **OPENAI_API_KEY**.\n\n"
        "Streamlit Cloud: **Settings → Secrets → OPENAI_API_KEY**"
    )
    st.stop()


# ============================================================
# CACHED RESOURCES
# ============================================================
@st.cache_resource(show_spinner=False)
def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(api_key=OPENAI_API_KEY)

@st.cache_resource(show_spinner=False)
def get_llm() -> ChatOpenAI:
    return ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE, api_key=OPENAI_API_KEY)


# ============================================================
# HELPERS: VECTOR SEARCH (ottimizzati)
# ============================================================
def normalize_rows(mat: np.ndarray) -> np.ndarray:
    return mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)

def cosine_scores(query_vec: np.ndarray, mat_norm: np.ndarray) -> np.ndarray:
    q = query_vec / (np.linalg.norm(query_vec) + 1e-12)
    return mat_norm @ q

@st.cache_data(show_spinner=False)
def load_index_cached(vec_path: str, meta_path: str) -> Tuple[np.ndarray, List[Dict[str, Any]], np.ndarray]:
    vectors = np.load(vec_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta_raw = json.load(f)

    meta: List[Dict[str, Any]] = []
    for item in meta_raw:
        if isinstance(item, dict) and "text" in item:
            meta.append({"page": item.get("page", "?"), "text": item.get("text", "")})
        else:
            meta.append({"page": "?", "text": str(item)})

    mat_norm = normalize_rows(vectors.astype(np.float32))
    return vectors.astype(np.float32), meta, mat_norm

def bm25_rerank(query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not BM25_AVAILABLE or not candidates:
        return candidates
    corpus = [(c.get("text") or "").lower().split() for c in candidates]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(query.lower().split())
    idx = np.argsort(-np.array(scores))
    return [candidates[int(i)] for i in idx]

def dedup_by_text(chunks: List[Dict[str, Any]], max_out: int) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for c in chunks:
        txt = (c.get("text") or "").strip()
        key = sha256_text(txt[:500])
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
        if len(out) >= max_out:
            break
    return out

def unique_pages_ccnl(chunks: List[Dict[str, Any]], max_pages: int = 8) -> List[int]:
    pages: List[int] = []
    for c in chunks:
        try:
            p = int(c.get("page", 0))
        except Exception:
            continue
        if p and p not in pages:
            pages.append(p)
        if len(pages) >= max_pages:
            break
    return pages

def format_ccnl_citations(pages: List[int]) -> str:
    if not pages:
        return ""
    pages_sorted = sorted(pages)
    if len(pages_sorted) == 1:
        return f"**Fonte:** CCNL (pag. {pages_sorted[0]})"
    return f"**Fonte:** CCNL (pagg. {', '.join(map(str, pages_sorted))})"

def extract_evidence(chunks: List[Dict[str, Any]], max_lines: int = 18) -> List[str]:
    patterns = [
        r"rao", r"r\.a\.o", r"rol", r"r\.o\.l", r"riduzion",
        r"permess", r"conged", r"retribuit", r"non retribuit",
        r"donazion", r"decesso", r"lutto", r"studio", r"esam", r"104",
        r"cure termali", r"elettoral", r"testimonianza",
        r"inquadrament", r"livell", r"categoria", r"mansioni superiori",
    ]
    out = []
    for c in chunks:
        page = c.get("page", "?")
        text = c.get("text", "") or ""
        for ln in [x.strip() for x in text.splitlines() if x.strip()]:
            low = ln.lower()
            if any(re.search(p, low) for p in patterns):
                clean = " ".join(ln.split())
                if 20 <= len(clean) <= 420:
                    out.append(f"({page}) {clean}")
            if len(out) >= max_lines:
                return out
    return out[:max_lines]


# ============================================================
# IPZS PERMESSI: PARSER ELENCO COMPLETO (deterministico)
# ============================================================
def parse_ipzs_permessi_sections(raw_text: str) -> List[Dict[str, str]]:
    lines = raw_text.splitlines()
    sections: List[Dict[str, str]] = []

    def is_title(line: str) -> bool:
        s = line.strip()
        if not s:
            return False
        if set(s) <= set("-_= "):
            return False
        if s in ["RAO", "R.O.L.", "ROL", "R.A.O."]:
            return True
        letters = [ch for ch in s if ch.isalpha()]
        if len(letters) >= 6 and sum(ch.isupper() for ch in letters) / max(len(letters), 1) > 0.8:
            return True
        if re.search(r"\bL\.\s*\d+/\d+\b", s):
            return True
        return False

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if is_title(line):
            title = line
            j = i + 1
            while j < len(lines) and (lines[j].strip() == "" or set(lines[j].strip()) <= set("-_= ")):
                j += 1

            desc_lines = []
            if j < len(lines) and lines[j].strip().lower().startswith("descrizione"):
                desc_lines.append(lines[j].strip())
                j += 1

            while j < len(lines):
                nxt = lines[j].strip()
                if is_title(nxt):
                    break
                desc_lines.append(lines[j].rstrip())
                j += 1

            desc = "\n".join([x for x in desc_lines if x.strip()]).strip()
            if desc:
                sections.append({"title": title.strip(), "desc": desc})
            i = j
        else:
            i += 1

    out = []
    seen = set()
    for s in sections:
        t = re.sub(r"\s+", " ", s["title"]).strip()
        key = t.lower()
        if key not in seen:
            out.append({"title": t, "desc": s["desc"]})
            seen.add(key)
    return out

def ipzs_source_line() -> str:
    return "**Fonte:** Documento IPZS Permessi/Giustificativi (PERMESSI_IPZS_COMPLETO_FINALE.txt)"

def is_generic_permessi_list_question(q: str) -> bool:
    ql = q.lower()
    triggers = [
        "a quali permessi ha diritto",
        "quali permessi ha diritto",
        "quali permessi ha a disposizione",
        "elenco permessi",
        "tutti i permessi",
        "lista permessi",
        "permessi disponibili",
        "permessi ipzs",
        "giustificativi",
    ]
    return any(t in ql for t in triggers)

def extract_specific_permesso_name(q: str, sections_titles: List[str]) -> Optional[str]:
    ql = q.lower()
    if re.search(r"\brao\b|\br\.a\.o\b", ql):
        return "RAO"
    if re.search(r"\brol\b|\br\.o\.l\b", ql):
        return "R.O.L."

    for t in sections_titles:
        tl = t.lower()
        if len(tl) < 4:
            continue
        if tl in ql:
            return t

    keywords_map = {
        "donazione": ["DONAZIONE SANGUE", "DONAZIONE"],
        "sangue": ["DONAZIONE SANGUE"],
        "cure termali": ["CURE TERMALI"],
        "elettoral": ["PERMESSI ELETTORALI"],
        "decesso": ["DECESSO", "DECESSO FAMILIARI"],
        "testimon": ["TESTIMONIANZA CIVILE"],
        "esami": ["PERM. ESAMI", "ESAMI"],
        "104": ["L. 104/92", "L.104/92", "104"],
        "maternit": ["MATERNITA' OBBLIGATORIA", "MATERNITA'"],
        "padre": ["CONG.OBBLIGATORIO PADRE"],
        "nido": ["INSERIM.NIDO", "ACC.NIDO", "NIDO"],
    }
    for k, cand in keywords_map.items():
        if k in ql:
            for c in cand:
                for real in sections_titles:
                    if c.lower() in real.lower() or real.lower() in c.lower():
                        return real
    return None


# ============================================================
# ROUTING: IPZS vs CCNL (più solido)
# ============================================================
def detect_route(q: str) -> str:
    ql = q.lower()
    if is_generic_permessi_list_question(ql):
        return "IPZS_LIST"
    if any(x in ql for x in ["rao", "r.a.o", "rol", "r.o.l", "donazione", "cure termali",
                             "permessi elettorali", "testimonianza", "decesso", "104", "esami", "maternit", "nido"]):
        return "IPZS_SINGLE"
    if "permess" in ql or "giustific" in ql:
        return "MIXED"
    return "CCNL"


# ============================================================
# MULTI-QUERY BUILDER
# ============================================================
def build_queries(q: str) -> List[str]:
    q0 = q.strip()
    qs = [
        q0,
        f"{q0} CCNL",
        f"{q0} definizione",
        f"{q0} procedura",
        f"{q0} requisiti",
    ]
    out, seen = [], set()
    for x in qs:
        x = x.strip()
        if x and x not in seen:
            out.append(x)
            seen.add(x)
    return out[:MAX_MULTI_QUERIES]


# ============================================================
# RETRIEVE (generic)
# ============================================================
def retrieve_from_index(
    query: str,
    vec_path: str,
    meta_path: str,
    emb: OpenAIEmbeddings,
    top_k_per_query: int = TOP_K_PER_QUERY,
    top_k_final: int = TOP_K_FINAL,
) -> Tuple[List[Dict[str, Any]], float, List[str], List[str]]:
    _, meta, mat_norm = load_index_cached(vec_path, meta_path)

    queries = build_queries(query)
    scores_best: Dict[int, float] = {}
    best_similarity = 0.0

    for q in queries:
        qvec = np.array(emb.embed_query(q), dtype=np.float32)
        sims = cosine_scores(qvec, mat_norm)
        top_idx = np.argsort(-sims)[:top_k_per_query]
        for i in top_idx:
            s = float(sims[int(i)])
            best_similarity = max(best_similarity, s)
            if (int(i) not in scores_best) or (s > scores_best[int(i)]):
                scores_best[int(i)] = s

    final_idx = sorted(scores_best.keys(), key=lambda i: scores_best[i], reverse=True)[:top_k_final]
    selected = [meta[i] for i in final_idx]
    selected = bm25_rerank(query, selected)
    selected = dedup_by_text(selected, max_out=top_k_final)

    evidence = extract_evidence(selected, max_lines=MAX_EVIDENCE_LINES)
    return selected, best_similarity, queries, evidence


def hard_not_found_ccnl() -> str:
    return "Non ho trovato la risposta nel CCNL caricato."

def hard_not_found_generic() -> str:
    return "Non ho trovato la risposta nei documenti caricati."


# ============================================================
# CCNL LLM PROMPT
# ============================================================
RULES_CCNL = """
Sei l’assistente UILCOM per lavoratori IPZS.
Rispondi in modo chiaro e pratico basandoti SOLO sul contesto (estratti CCNL) fornito.
Non inventare informazioni.

REGOLE:
1) Se non trovi nel contesto, scrivi: "Non ho trovato la risposta nel CCNL caricato."
2) Chiudi SEMPRE con: "Fonte: CCNL (pag. ...)" usando solo le pagine presenti nel contesto.
3) Se l’utente chiede permessi ma nel contesto non ci sono dettagli, non inventare.
"""

def split_public(text: str) -> str:
    m = re.search(r"<PUBLIC>(.*?)</PUBLIC>", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return text.strip()


# ============================================================
# INDICIZZAZIONE (CCNL + IPZS + Q/A approvate)
# ============================================================
def do_reindex_all():
    emb = get_embeddings()
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    # ---- CCNL PDF ----
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"Non trovo il PDF CCNL: {PDF_PATH}")

    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()
    ccnl_chunks = splitter.split_documents(docs)

    ccnl_texts = [c.page_content for c in ccnl_chunks]
    ccnl_pages = [(int(c.metadata.get("page", 0)) + 1) for c in ccnl_chunks]  # 1-based

    ccnl_vectors = np.array(emb.embed_documents(ccnl_texts), dtype=np.float32)
    np.save(CCNL_VEC_PATH, ccnl_vectors)
    with open(CCNL_META_PATH, "w", encoding="utf-8") as f:
        json.dump([{"page": p, "text": t} for p, t in zip(ccnl_pages, ccnl_texts)], f, ensure_ascii=False)

    # ---- IPZS Permessi TXT ----
    if not os.path.exists(IPZS_PERMESSI_PATH):
        raise FileNotFoundError(f"Non trovo IPZS Permessi: {IPZS_PERMESSI_PATH}")

    with open(IPZS_PERMESSI_PATH, "r", encoding="utf-8", errors="ignore") as f:
        ipzs_raw = f.read()

    ipzs_chunks = splitter.split_text(ipzs_raw)
    ipzs_vectors = np.array(emb.embed_documents(ipzs_chunks), dtype=np.float32)
    np.save(IPZS_VEC_PATH, ipzs_vectors)
    with open(IPZS_META_PATH, "w", encoding="utf-8") as f:
        json.dump([{"page": "IPZS", "text": t} for t in ipzs_chunks], f, ensure_ascii=False)

    # ---- Q/A approvate (autoapprendimento controllato) ----
    approved_rows = read_jsonl(APPROVED_QA_FILE, limit=5000)
    qa_texts: List[str] = []
    qa_meta: List[Dict[str, Any]] = []

    for r in approved_rows:
        q = (r.get("question") or "").strip()
        a = (r.get("answer") or "").strip()
        if not q or not a:
            continue
        txt = f"DOMANDA: {q}\nRISPOSTA APPROVATA: {a}"
        qa_texts.append(txt)
        qa_meta.append({"page": "APPROVED", "text": txt})

    if qa_texts:
        qa_vec = np.array(emb.embed_documents(qa_texts), dtype=np.float32)
        np.save(APPROVED_VEC_PATH, qa_vec)
        with open(APPROVED_META_PATH, "w", encoding="utf-8") as f:
            json.dump(qa_meta, f, ensure_ascii=False)
    else:
        # se vuoto, rimuovi indice vecchio per evitare confusione
        if os.path.exists(APPROVED_VEC_PATH):
            os.remove(APPROVED_VEC_PATH)
        if os.path.exists(APPROVED_META_PATH):
            os.remove(APPROVED_META_PATH)


# ============================================================
# CHAT STATE
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

def trimmed_history(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Manteniamo gli ultimi turni per ridurre “contaminazione”
    # (ChatOpenAI usa il contesto “messages” che gli passiamo, non quello che vede a schermo.)
    if len(messages) <= MAX_HISTORY_TURNS * 2:
        return messages
    return messages[-MAX_HISTORY_TURNS * 2:]


# Render chat history
for i, m in enumerate(st.session_state.messages):
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

        # Citazioni (se presenti)
        if m["role"] == "assistant" and m.get("citations"):
            st.caption(m["citations"])

        # Feedback sui messaggi assistant
        if m["role"] == "assistant":
            cols = st.columns(10)
            with cols[0]:
                if st.button("👍", key=f"up_{i}"):
                    append_jsonl(FEEDBACK_FILE, {
                        "ts": now_iso(),
                        "chat_id": st.session_state.chat_id,
                        "msg_index": i,
                        "rating": "up",
                        "question": m.get("question", ""),
                        "answer": m.get("content", ""),
                    })
                    st.toast("Feedback 👍 salvato")
            with cols[1]:
                if st.button("👎", key=f"down_{i}"):
                    append_jsonl(FEEDBACK_FILE, {
                        "ts": now_iso(),
                        "chat_id": st.session_state.chat_id,
                        "msg_index": i,
                        "rating": "down",
                        "question": m.get("question", ""),
                        "answer": m.get("content", ""),
                    })
                    st.toast("Feedback 👎 salvato")

        # Admin debug (come nel tuo, ma più pulito)
        if st.session_state.is_admin and m["role"] == "assistant":
            dbg = m.get("debug")
            if dbg:
                with st.expander("🧠 Admin: debug", expanded=False):
                    st.write("**Route:**", dbg.get("route"))
                    st.write("**Best similarity:**", dbg.get("best_similarity"))
                    st.write("**Query usate:**")
                    st.code("\n".join(dbg.get("queries", [])) or "(nessuna)")
                    st.write("**Evidenze:**")
                    st.code("\n".join(dbg.get("evidence", [])) or "(nessuna)")
                    if dbg.get("pages_ccnl"):
                        st.write("**Pagine CCNL:**", dbg.get("pages_ccnl"))
                    st.write("**Chunk selezionati (prime righe):**")
                    for c in (dbg.get("selected") or [])[:4]:
                        st.write(f"**Pagina/Origine: {c.get('page')}**")
                        txt = (c.get("text") or "")
                        st.write(txt[:700] + ("..." if len(txt) > 700 else ""))
                        st.divider()


user_input = st.chat_input("Scrivi una domanda (CCNL / permessi IPZS: RAO, ROL, donazione sangue, ecc.)")
if not user_input:
    st.stop()

user_input = user_input.strip()
if user_input.lower() in ("/nuova", "/reset", "/new"):
    st.session_state.messages = []
    st.session_state.chat_id = sha256_text(str(time.time()))[:12]
    st.rerun()


# Require indexes
need_ccnl = not (os.path.exists(CCNL_VEC_PATH) and os.path.exists(CCNL_META_PATH))
need_ipzs = not (os.path.exists(IPZS_VEC_PATH) and os.path.exists(IPZS_META_PATH))

# approved index è opzionale
have_approved = os.path.exists(APPROVED_VEC_PATH) and os.path.exists(APPROVED_META_PATH)

if need_ccnl or need_ipzs:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Prima devo indicizzare i documenti: apri la barra laterale e clicca **Indicizza / Reindicizza (CCNL + IPZS + Q/A)**.",
        "question": user_input,
    })
    st.rerun()

# Log domanda
append_jsonl(LOGS_FILE, {"ts": now_iso(), "chat_id": st.session_state.chat_id, "event": "question", "text": user_input})

st.session_state.messages.append({"role": "user", "content": user_input})

emb = get_embeddings()
route = detect_route(user_input)


# ============================================================
# IPZS: ELENCO COMPLETO (deterministico)
# ============================================================
if route == "IPZS_LIST":
    try:
        with open(IPZS_PERMESSI_PATH, "r", encoding="utf-8", errors="ignore") as f:
            ipzs_raw = f.read()
        sections = parse_ipzs_permessi_sections(ipzs_raw)

        if not sections:
            public_ans = hard_not_found_generic()
        else:
            lines = []
            lines.append("Ecco l’**elenco dei permessi/giustificativi** presenti nel documento IPZS, con una breve spiegazione:")
            lines.append("")
            for s in sections:
                title = s["title"]
                desc = s["desc"]
                desc_short = []
                for ln in desc.splitlines():
                    ln = ln.strip()
                    if not ln:
                        continue
                    if ln.lower() == "descrizione:":
                        continue
                    desc_short.append(ln)
                    if len(desc_short) >= 2:
                        break
                preview = " ".join(desc_short).strip() or "Vedi descrizione nel documento."
                lines.append(f"- **{title}**: {preview}")

            lines.append("")
            lines.append(ipzs_source_line())
            public_ans = "\n".join(lines).strip()

        assistant_payload = {
            "role": "assistant",
            "content": public_ans,
            "citations": ipzs_source_line(),
            "question": user_input,
        }

        if st.session_state.is_admin:
            assistant_payload["debug"] = {
                "route": route,
                "best_similarity": None,
                "queries": [],
                "evidence": [],
                "pages_ccnl": [],
                "selected": [],
                "note": "Elenco IPZS generato deterministico dal file completo.",
            }

        st.session_state.messages.append(assistant_payload)
        append_jsonl(LOGS_FILE, {"ts": now_iso(), "chat_id": st.session_state.chat_id, "event": "answer", "route": route})
        st.rerun()

    except Exception:
        assistant_payload = {"role": "assistant", "content": hard_not_found_generic(), "question": user_input}
        st.session_state.messages.append(assistant_payload)
        st.rerun()


# ============================================================
# IPZS: PERMESSO SPECIFICO + fallback CCNL se MIXED
# ============================================================
def answer_from_ipzs(user_q: str) -> Tuple[Optional[str], Dict[str, Any]]:
    selected_ipzs, best_ipzs, queries_ipzs, evidence_ipzs = retrieve_from_index(
        user_q, IPZS_VEC_PATH, IPZS_META_PATH, emb
    )
    retrieval_ok_ipzs = (best_ipzs >= MIN_BEST_SIMILARITY) and (len(selected_ipzs) >= MIN_SELECTED_CHUNKS)

    # carica sezioni per match del titolo
    try:
        with open(IPZS_PERMESSI_PATH, "r", encoding="utf-8", errors="ignore") as f:
            ipzs_raw = f.read()
        sections = parse_ipzs_permessi_sections(ipzs_raw)
        titles = [s["title"] for s in sections]
        wanted = extract_specific_permesso_name(user_q, titles)
    except Exception:
        sections, wanted = [], None

    if not retrieval_ok_ipzs:
        return None, {
            "route": "IPZS",
            "best_similarity": best_ipzs,
            "queries": queries_ipzs,
            "evidence": evidence_ipzs,
            "selected": selected_ipzs[:6],
            "note": "retrieval_ok=False",
        }

    if sections and wanted:
        match = None
        if wanted == "RAO":
            for s in sections:
                if s["title"].strip().lower() in ["rao", "r.a.o."] or "rao" in s["title"].lower():
                    match = s
                    break
        elif wanted in ["R.O.L.", "ROL"]:
            for s in sections:
                if "rol" in s["title"].lower() or "r.o.l" in s["title"].lower():
                    match = s
                    break
        else:
            for s in sections:
                if s["title"].lower() == wanted.lower() or wanted.lower() in s["title"].lower():
                    match = s
                    break

        if match:
            public_ans = f"**{match['title']}**\n\n{match['desc'].strip()}\n\n{ipzs_source_line()}"
            return public_ans, {
                "route": "IPZS",
                "best_similarity": best_ipzs,
                "queries": queries_ipzs,
                "evidence": evidence_ipzs,
                "selected": selected_ipzs[:6],
                "note": f"Permesso specifico richiesto: {wanted}",
            }

    # fallback: usa chunk recuperati (sempre con fonte)
    context = "\n\n---\n\n".join([c.get("text", "") for c in selected_ipzs[:5]])
    public_ans = f"Ho trovato riferimenti nel documento IPZS:\n\n{context}\n\n{ipzs_source_line()}"
    return public_ans, {
        "route": "IPZS",
        "best_similarity": best_ipzs,
        "queries": queries_ipzs,
        "evidence": evidence_ipzs,
        "selected": selected_ipzs[:6],
        "note": "Fallback su chunk IPZS",
    }

# Proviamo IPZS se route lo suggerisce
if route in ["IPZS_SINGLE", "MIXED"]:
    ipzs_ans, ipzs_dbg = answer_from_ipzs(user_input)
    if ipzs_ans:
        assistant_payload = {
            "role": "assistant",
            "content": ipzs_ans,
            "citations": ipzs_source_line(),
            "question": user_input,
        }
        if st.session_state.is_admin:
            assistant_payload["debug"] = {
                **ipzs_dbg,
                "pages_ccnl": [],
            }
        st.session_state.messages.append(assistant_payload)
        append_jsonl(LOGS_FILE, {"ts": now_iso(), "chat_id": st.session_state.chat_id, "event": "answer", "route": "IPZS"})
        st.rerun()
    # se MIXED e IPZS non ok, prosegui su CCNL (fallback)
    if route != "MIXED":
        assistant_payload = {"role": "assistant", "content": hard_not_found_generic(), "question": user_input}
        st.session_state.messages.append(assistant_payload)
        st.rerun()


# ============================================================
# CCNL (+ Q/A approvate come “booster”)
# ============================================================
selected_ccnl, best_ccnl, queries_ccnl, evidence_ccnl = retrieve_from_index(
    user_input, CCNL_VEC_PATH, CCNL_META_PATH, emb
)
retrieval_ok_ccnl = (best_ccnl >= MIN_BEST_SIMILARITY) and (len(selected_ccnl) >= MIN_SELECTED_CHUNKS)

# Se esiste indice Q/A approvate, proviamo anche quello e uniamo (utile per casi specifici “ripetitivi”)
approved_dbg = None
if have_approved:
    selected_qa, best_qa, queries_qa, evidence_qa = retrieve_from_index(
        user_input, APPROVED_VEC_PATH, APPROVED_META_PATH, emb,
        top_k_per_query=8, top_k_final=10
    )
    # unione: mettiamo prima Q/A se score buono, poi CCNL
    if best_qa >= max(MIN_BEST_SIMILARITY, 0.28):
        selected_ccnl = dedup_by_text(selected_qa + selected_ccnl, max_out=TOP_K_FINAL)
        evidence_ccnl = extract_evidence(selected_ccnl, max_lines=MAX_EVIDENCE_LINES)
        approved_dbg = {"best_qa": best_qa, "queries_qa": queries_qa, "evidence_qa": evidence_qa}

if not retrieval_ok_ccnl:
    assistant_payload = {"role": "assistant", "content": hard_not_found_ccnl(), "question": user_input}
    if st.session_state.is_admin:
        assistant_payload["debug"] = {
            "route": "CCNL",
            "best_similarity": best_ccnl,
            "queries": queries_ccnl,
            "evidence": evidence_ccnl,
            "pages_ccnl": unique_pages_ccnl(selected_ccnl),
            "selected": selected_ccnl[:6],
            "note": "retrieval_ok=False -> risposta bloccata",
            "approved_dbg": approved_dbg,
        }
    st.session_state.messages.append(assistant_payload)
    append_jsonl(LOGS_FILE, {"ts": now_iso(), "chat_id": st.session_state.chat_id, "event": "answer", "route": "CCNL_BLOCKED"})
    st.rerun()

pages = unique_pages_ccnl(selected_ccnl, max_pages=8)
cit_line = format_ccnl_citations(pages)

context = "\n\n---\n\n".join([f"[Pagina {c.get('page','?')}] {c.get('text','')}" for c in selected_ccnl])

llm = get_llm()

# Passiamo al modello SOLO l’ultimo pezzo di history per evitare “contaminazioni”
hist = trimmed_history([m for m in st.session_state.messages if m["role"] in ("user", "assistant")])

prompt = f"""
{RULES_CCNL}

STORIA CHAT (ultimi turni, se utili):
{json.dumps(hist, ensure_ascii=False)}

DOMANDA (UTENTE):
{user_input}

CONTESTO CCNL (estratti):
{context}

SCRIVI UNA RISPOSTA CHIARA E PRATICA.
CHIUDI SEMPRE CON:
{cit_line}

FORMATO:
<PUBLIC>
...testo...
{cit_line}
</PUBLIC>
"""

try:
    raw = llm.invoke(prompt).content
except Exception as e:
    raw = f"<PUBLIC>Errore nel generare la risposta: {e}\n\n{cit_line}</PUBLIC>"

public_ans = split_public(raw).strip()

# Forza citazione se manca
if cit_line and ("fonte" not in public_ans.lower()):
    public_ans = public_ans.rstrip() + "\n\n" + cit_line

assistant_payload = {
    "role": "assistant",
    "content": public_ans,
    "citations": cit_line,
    "question": user_input
}

if st.session_state.is_admin:
    assistant_payload["debug"] = {
        "route": "CCNL",
        "best_similarity": best_ccnl,
        "queries": queries_ccnl,
        "evidence": evidence_ccnl,
        "pages_ccnl": pages,
        "selected": selected_ccnl[:6],
        "bm25_available": BM25_AVAILABLE,
        "approved_dbg": approved_dbg,
    }

st.session_state.messages.append(assistant_payload)
append_jsonl(LOGS_FILE, {"ts": now_iso(), "chat_id": st.session_state.chat_id, "event": "answer", "route": "CCNL_OK", "pages": pages})
st.rerun()# Logo (opzionale)
LOGO_PATH_1 = os.path.join("logo", "logo_uilcom.png")   # consigliato
LOGO_PATH_2 = "logo_uilcom.png"                         # fallback se lo metti in root

# Indici
INDEX_CCNL_DIR = "index_ccnl"
CCNL_VEC_PATH = os.path.join(INDEX_CCNL_DIR, "vectors.npy")
CCNL_META_PATH = os.path.join(INDEX_CCNL_DIR, "chunks.json")

INDEX_IPZS_DIR = "index_ipzs_permessi"
IPZS_VEC_PATH = os.path.join(INDEX_IPZS_DIR, "vectors.npy")
IPZS_META_PATH = os.path.join(INDEX_IPZS_DIR, "chunks.json")

# Chunking
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150

TOP_K_PER_QUERY = 12
TOP_K_FINAL = 18
MAX_MULTI_QUERIES = 10

# Guardrail retrieval
MIN_BEST_SIMILARITY = 0.24
MIN_SELECTED_CHUNKS = 3

# Admin
MAX_EVIDENCE_LINES = 18

# LLM
LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0


# ============================================================
# SECRETS
# ============================================================
def get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    try:
        if key in st.secrets:  # type: ignore
            return str(st.secrets[key])  # type: ignore
    except Exception:
        pass
    return os.getenv(key, default)

def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

UILCOM_PASSWORD = get_secret("UILCOM_PASSWORD")        # password iscritti
ADMIN_PASSWORD = get_secret("ADMIN_PASSWORD")          # password admin
OPENAI_API_KEY = get_secret("OPENAI_API_KEY")          # obbligatoria


# ============================================================
# UI SETUP
# ============================================================
st.set_page_config(page_title="Assistente UILCOM IPZS", page_icon="🟦", layout="centered")

# Logo (non deve mai rompere l'app)
logo_to_use = LOGO_PATH_1 if os.path.exists(LOGO_PATH_1) else (LOGO_PATH_2 if os.path.exists(LOGO_PATH_2) else None)
if logo_to_use:
    try:
        st.image(logo_to_use, width=140)
    except Exception:
        pass

st.title(APP_TITLE)
st.markdown(
    "**Accesso riservato agli iscritti UILCOM**  \n"
    "Strumento informativo per facilitare la consultazione del **CCNL** e del documento **IPZS Permessi/Giustificativi**.  \n\n"
    "⚠️ Le risposte sono generate **solo** in base ai documenti caricati. "
    "Per casi specifici o interpretazioni, rivolgersi a RSU/UILCOM o HR."
)
st.divider()


# ============================================================
# AUTH: ISCRITTI
# ============================================================
if "auth_ok" not in st.session_state:
    st.session_state.auth_ok = False

if UILCOM_PASSWORD:
    with st.expander("🔒 Accesso iscritti UILCOM", expanded=not st.session_state.auth_ok):
        pwd_in = st.text_input("Password iscritti", type="password", placeholder="Inserisci password iscritti")
        if st.button("Entra", use_container_width=True):
            if pwd_in == UILCOM_PASSWORD:
                st.session_state.auth_ok = True
                st.success("Accesso consentito.")
            else:
                st.session_state.auth_ok = False
                st.error("Password non corretta.")
else:
    st.warning("Password iscritti non impostata. Imposta UILCOM_PASSWORD in Secrets.")
    # st.session_state.auth_ok = True  # per sviluppo locale

if not st.session_state.auth_ok:
    st.stop()


# ============================================================
# ADMIN SIDEBAR
# ============================================================
if "is_admin" not in st.session_state:
    st.session_state.is_admin = False

with st.sidebar:
    st.header("⚙️ Controlli")

    st.subheader("🧠 Admin (debug)")
    if ADMIN_PASSWORD:
        admin_in = st.text_input("Password admin", type="password", placeholder="Solo admin UILCOM", key="admin_pwd")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Login admin", use_container_width=True):
                if admin_in == ADMIN_PASSWORD:
                    st.session_state.is_admin = True
                    st.success("Admin attivo.")
                else:
                    st.session_state.is_admin = False
                    st.error("Password admin errata.")
        with c2:
            if st.button("Logout", use_container_width=True):
                st.session_state.is_admin = False
    else:
        st.caption("ADMIN_PASSWORD non impostata (Secrets).")

    st.divider()

    st.subheader("📦 Indici documenti")
    ok_ccnl = os.path.exists(CCNL_VEC_PATH) and os.path.exists(CCNL_META_PATH)
    ok_ipzs = os.path.exists(IPZS_VEC_PATH) and os.path.exists(IPZS_META_PATH)

    st.write("Indice CCNL:", "✅" if ok_ccnl else "❌")
    st.write("Indice IPZS Permessi:", "✅" if ok_ipzs else "❌")

    if st.button("Indicizza / Reindicizza (CCNL + IPZS)", use_container_width=True):
        if not OPENAI_API_KEY:
            st.error("Manca OPENAI_API_KEY nei Secrets.")
        else:
            try:
                with st.spinner("Indicizzazione in corso..."):
                    os.makedirs(INDEX_CCNL_DIR, exist_ok=True)
                    os.makedirs(INDEX_IPZS_DIR, exist_ok=True)

                    emb = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
                    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

                    # ---- CCNL PDF ----
                    if not os.path.exists(PDF_PATH):
                        raise FileNotFoundError(f"Non trovo il PDF CCNL: {PDF_PATH}")

                    loader = PyPDFLoader(PDF_PATH)
                    docs = loader.load()
                    ccnl_chunks = splitter.split_documents(docs)

                    ccnl_texts = [c.page_content for c in ccnl_chunks]
                    ccnl_pages = [(int(c.metadata.get("page", 0)) + 1) for c in ccnl_chunks]  # 1-based

                    ccnl_vectors = np.array(emb.embed_documents(ccnl_texts), dtype=np.float32)
                    np.save(CCNL_VEC_PATH, ccnl_vectors)
                    with open(CCNL_META_PATH, "w", encoding="utf-8") as f:
                        json.dump([{"page": p, "text": t} for p, t in zip(ccnl_pages, ccnl_texts)], f, ensure_ascii=False)

                    # ---- IPZS Permessi TXT ----
                    if not os.path.exists(IPZS_PERMESSI_PATH):
                        raise FileNotFoundError(f"Non trovo IPZS Permessi: {IPZS_PERMESSI_PATH}")

                    with open(IPZS_PERMESSI_PATH, "r", encoding="utf-8", errors="ignore") as f:
                        ipzs_raw = f.read()

                    # chunk a testo con "pseudo-page" (sezione)
                    # Spezziamo comunque con splitter per retrieval robusto
                    ipzs_docs = [{"page": "IPZS", "text": ipzs_raw}]
                    ipzs_texts = [ipzs_raw]
                    # split
                    ipzs_chunks = splitter.split_text(ipzs_raw)
                    ipzs_vectors = np.array(emb.embed_documents(ipzs_chunks), dtype=np.float32)
                    np.save(IPZS_VEC_PATH, ipzs_vectors)
                    with open(IPZS_META_PATH, "w", encoding="utf-8") as f:
                        json.dump([{"page": "IPZS", "text": t} for t in ipzs_chunks], f, ensure_ascii=False)

                st.success("Indicizzazione completata ✅")
                st.rerun()
            except Exception as e:
                st.error(str(e))

    if st.button("🧹 Nuova chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("Suggerimento: dopo aggiornamenti su GitHub → Streamlit Cloud fa auto-redeploy. Se no: **Reboot app**.")


# ============================================================
# HARD FAIL IF NO OPENAI KEY
# ============================================================
if not OPENAI_API_KEY:
    st.error(
        "Manca la variabile **OPENAI_API_KEY**.\n\n"
        "Streamlit Cloud: **Settings → Secrets → OPENAI_API_KEY**"
    )
    st.stop()


# ============================================================
# HELPERS: VECTOR SEARCH
# ============================================================
def normalize_rows(mat: np.ndarray) -> np.ndarray:
    return mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)

def cosine_scores(query_vec: np.ndarray, mat_norm: np.ndarray) -> np.ndarray:
    q = query_vec / (np.linalg.norm(query_vec) + 1e-12)
    return mat_norm @ q

def load_index(vec_path: str, meta_path: str) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    vectors = np.load(vec_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    fixed = []
    for item in meta:
        if isinstance(item, dict) and "text" in item:
            fixed.append({"page": item.get("page", "?"), "text": item.get("text", "")})
        else:
            fixed.append({"page": "?", "text": str(item)})
    return vectors, fixed

def bm25_rerank(query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not BM25_AVAILABLE or not candidates:
        return candidates
    corpus = [(c.get("text") or "").lower().split() for c in candidates]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(query.lower().split())
    idx = np.argsort(-np.array(scores))
    return [candidates[int(i)] for i in idx]

def unique_pages_ccnl(chunks: List[Dict[str, Any]], max_pages: int = 8) -> List[int]:
    pages: List[int] = []
    for c in chunks:
        try:
            p = int(c.get("page", 0))
        except Exception:
            continue
        if p and p not in pages:
            pages.append(p)
        if len(pages) >= max_pages:
            break
    return pages

def format_ccnl_citations(pages: List[int]) -> str:
    if not pages:
        return ""
    pages_sorted = sorted(pages)
    if len(pages_sorted) == 1:
        return f"**Fonte:** CCNL (pag. {pages_sorted[0]})"
    return f"**Fonte:** CCNL (pagg. {', '.join(map(str, pages_sorted))})"

def extract_evidence(chunks: List[Dict[str, Any]], max_lines: int = 18) -> List[str]:
    patterns = [
        r"rao", r"r\.a\.o", r"rol", r"r\.o\.l", r"riduzion",
        r"permess", r"conged", r"retribuit", r"non retribuit",
        r"donazion", r"decesso", r"lutto", r"studio", r"esam", r"104",
        r"cure termali", r"elettoral", r"testimonianza",
    ]
    out = []
    for c in chunks:
        page = c.get("page", "?")
        text = c.get("text", "") or ""
        for ln in [x.strip() for x in text.splitlines() if x.strip()]:
            low = ln.lower()
            if any(re.search(p, low) for p in patterns):
                clean = " ".join(ln.split())
                if 20 <= len(clean) <= 420:
                    out.append(f"({page}) {clean}")
            if len(out) >= max_lines:
                return out
    return out[:max_lines]


# ============================================================
# IPZS PERMESSI: PARSER ELENCO COMPLETO (deterministico)
# ============================================================
def parse_ipzs_permessi_sections(raw_text: str) -> List[Dict[str, str]]:
    """
    Prova a ricostruire un elenco di "permessi" dal file IPZS:
    - Riconosce titoli in MAIUSCOLO (es: DONAZIONE SANGUE, RAO, R.O.L., CURE TERMALI)
    - Prende il blocco "Descrizione:" sotto al titolo fino al prossimo titolo
    """
    lines = raw_text.splitlines()
    sections: List[Dict[str, str]] = []

    def is_title(line: str) -> bool:
        s = line.strip()
        if not s:
            return False
        # escludi righe decorative
        if set(s) <= set("-_= "):
            return False
        # titolo: molti caratteri maiuscoli + pochi segni
        # oppure RAO / R.O.L.
        if s in ["RAO", "R.O.L.", "ROL", "R.A.O."]:
            return True
        # se è quasi tutto uppercase
        letters = [ch for ch in s if ch.isalpha()]
        if len(letters) >= 6 and sum(ch.isupper() for ch in letters) / max(len(letters), 1) > 0.8:
            return True
        # contiene pattern tipici
        if re.search(r"\bL\.\s*\d+/\d+\b", s):
            return True
        return False

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if is_title(line):
            title = line
            # salta eventuali righe decorative sotto
            j = i + 1
            while j < len(lines) and (lines[j].strip() == "" or set(lines[j].strip()) <= set("-_= ")):
                j += 1

            # cerca descrizione
            desc_lines = []
            # se c'è "Descrizione:" la includiamo e prendiamo le righe successive
            if j < len(lines) and lines[j].strip().lower().startswith("descrizione"):
                desc_lines.append(lines[j].strip())
                j += 1

            while j < len(lines):
                nxt = lines[j].strip()
                if is_title(nxt):
                    break
                desc_lines.append(lines[j].rstrip())
                j += 1

            desc = "\n".join([x for x in desc_lines if x.strip()]).strip()
            if desc:
                sections.append({"title": title.strip(), "desc": desc})
            i = j
        else:
            i += 1

    # dedup titoli
    out = []
    seen = set()
    for s in sections:
        t = re.sub(r"\s+", " ", s["title"]).strip()
        key = t.lower()
        if key not in seen:
            out.append({"title": t, "desc": s["desc"]})
            seen.add(key)
    return out

def ipzs_source_line() -> str:
    return "**Fonte:** Documento IPZS Permessi/Giustificativi (PERMESSI_IPZS_COMPLETO_FINALE.txt)"

def is_generic_permessi_list_question(q: str) -> bool:
    ql = q.lower()
    triggers = [
        "a quali permessi ha diritto",
        "quali permessi ha diritto",
        "quali permessi ha a disposizione",
        "elenco permessi",
        "tutti i permessi",
        "lista permessi",
        "permessi disponibili",
        "permessi ipzs",
        "giustificativi",
    ]
    return any(t in ql for t in triggers)

def extract_specific_permesso_name(q: str, sections_titles: List[str]) -> Optional[str]:
    """
    Se l'utente chiede un permesso specifico, proviamo a matchare:
    - RAO / ROL
    - parole contenute in un titolo
    """
    ql = q.lower()

    # priorità RAO / ROL
    if re.search(r"\brao\b|\br\.a\.o\b", ql):
        return "RAO"
    if re.search(r"\brol\b|\br\.o\.l\b", ql):
        return "R.O.L."

    # match per inclusione (titolo dentro domanda o viceversa)
    for t in sections_titles:
        tl = t.lower()
        # evita titoli troppo generici
        if len(tl) < 4:
            continue
        if tl in ql:
            return t
    # match per parole chiave importanti
    keywords_map = {
        "donazione": ["DONAZIONE SANGUE", "DONAZIONE"],
        "sangue": ["DONAZIONE SANGUE"],
        "cure termali": ["CURE TERMALI"],
        "elettoral": ["PERMESSI ELETTORALI"],
        "decesso": ["DECESSO", "DECESSO FAMILIARI"],
        "testimon": ["TESTIMONIANZA CIVILE"],
        "esami": ["PERM. ESAMI", "ESAMI"],
        "104": ["L. 104/92", "L.104/92", "104"],
        "maternit": ["MATERNITA' OBBLIGATORIA", "MATERNITA'"],
        "padre": ["CONG.OBBLIGATORIO PADRE"],
        "nido": ["INSERIM.NIDO", "ACC.NIDO", "NIDO"],
    }
    for k, cand in keywords_map.items():
        if k in ql:
            # ritorna il primo che esiste davvero tra i titoli
            for c in cand:
                for real in sections_titles:
                    if c.lower() in real.lower() or real.lower() in c.lower():
                        return real
    return None


# ============================================================
# ROUTING: capire se rispondere con IPZS o CCNL
# ============================================================
def detect_route(q: str) -> str:
    """
    - Se domanda è lista permessi -> IPZS (elenco completo)
    - Se domanda su permesso specifico (rao/rol/donazione ecc) -> IPZS
    - Altrimenti -> CCNL
    """
    ql = q.lower()
    if is_generic_permessi_list_question(ql):
        return "IPZS_LIST"

    # segnali di permesso specifico
    if any(x in ql for x in ["rao", "r.a.o", "rol", "r.o.l", "donazione", "cure termali", "permessi elettorali", "testimonianza", "decesso", "104", "perm. esami", "maternit", "padre", "nido"]):
        return "IPZS_SINGLE"

    # se contiene "permessi" ma non chiede elenco, proviamo IPZS prima
    if "permess" in ql or "giustific" in ql:
        return "MIXED"

    return "CCNL"


# ============================================================
# MULTI-QUERY BUILDER
# ============================================================
def build_queries(q: str) -> List[str]:
    q0 = q.strip()
    qs = [q0, f"{q0} CCNL", f"{q0} regole condizioni", f"{q0} definizione procedura"]
    out, seen = [], set()
    for x in qs:
        x = x.strip()
        if x and x not in seen:
            out.append(x)
            seen.add(x)
    return out[:MAX_MULTI_QUERIES]


# ============================================================
# RETRIEVE (generic)
# ============================================================
def retrieve_from_index(
    query: str,
    vec_path: str,
    meta_path: str,
    emb: OpenAIEmbeddings,
    top_k_per_query: int = TOP_K_PER_QUERY,
    top_k_final: int = TOP_K_FINAL,
) -> Tuple[List[Dict[str, Any]], float, List[str], List[Dict[str, Any]]]:
    vectors, meta = load_index(vec_path, meta_path)
    mat_norm = normalize_rows(vectors)

    queries = build_queries(query)

    scores_best: Dict[int, float] = {}
    best_similarity = 0.0

    for q in queries:
        qvec = np.array(emb.embed_query(q), dtype=np.float32)
        sims = cosine_scores(qvec, mat_norm)
        top_idx = np.argsort(-sims)[:top_k_per_query]
        for i in top_idx:
            s = float(sims[int(i)])
            best_similarity = max(best_similarity, s)
            if (int(i) not in scores_best) or (s > scores_best[int(i)]):
                scores_best[int(i)] = s

    final_idx = sorted(scores_best.keys(), key=lambda i: scores_best[i], reverse=True)[:top_k_final]
    selected = [meta[i] for i in final_idx]
    selected = bm25_rerank(query, selected)

    evidence = extract_evidence(selected, max_lines=MAX_EVIDENCE_LINES)
    return selected, best_similarity, queries, selected


def hard_not_found() -> str:
    return "Non ho trovato la risposta nei documenti caricati."


# ============================================================
# CCNL LLM PROMPT
# ============================================================
RULES_CCNL = """
Sei l’assistente UILCOM per lavoratori IPZS.
Devi rispondere in modo chiaro e pratico basandoti SOLO sul contesto (estratti CCNL) fornito.
Non inventare informazioni.

REGOLE:
1) Se non trovi nel contesto, scrivi: "Non ho trovato la risposta nel CCNL caricato."
2) Chiudi SEMPRE con: "Fonte: CCNL (pag. ...)" usando solo le pagine presenti nel contesto.
3) Se l’utente chiede permessi ma nel contesto non ci sono tutti, non inventare.
"""

def split_public(text: str) -> str:
    # compatibile anche se il modello non usa tag
    m = re.search(r"<PUBLIC>(.*?)</PUBLIC>", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return text.strip()


# ============================================================
# CHAT STATE
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render chat history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if st.session_state.is_admin and m["role"] == "assistant":
            dbg = m.get("debug")
            if dbg:
                with st.expander("🧠 Admin: debug", expanded=False):
                    st.write("**Route:**", dbg.get("route"))
                    st.write("**Best similarity:**", dbg.get("best_similarity"))
                    st.write("**Query usate:**")
                    st.code("\n".join(dbg.get("queries", [])))
                    st.write("**Evidenze:**")
                    st.code("\n".join(dbg.get("evidence", [])) or "(nessuna)")
                    if dbg.get("pages_ccnl"):
                        st.write("**Pagine CCNL:**", dbg.get("pages_ccnl"))
                    st.write("**Chunk selezionati (prime righe):**")
                    for c in dbg.get("selected", [])[:4]:
                        st.write(f"**Pagina/Origine: {c.get('page')}**")
                        txt = (c.get("text") or "")
                        st.write(txt[:700] + ("..." if len(txt) > 700 else ""))
                        st.divider()

user_input = st.chat_input("Scrivi una domanda (CCNL / permessi IPZS: RAO, ROL, donazione sangue, ecc.)")

if not user_input:
    st.stop()

# Require indexes
need_ccnl = not (os.path.exists(CCNL_VEC_PATH) and os.path.exists(CCNL_META_PATH))
need_ipzs = not (os.path.exists(IPZS_VEC_PATH) and os.path.exists(IPZS_META_PATH))

if need_ccnl or need_ipzs:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Prima devo indicizzare i documenti: apri la barra laterale e clicca **Indicizza / Reindicizza (CCNL + IPZS)**.",
    })
    st.rerun()

st.session_state.messages.append({"role": "user", "content": user_input})

emb = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
route = detect_route(user_input)

# ============================================================
# IPZS: ELENCO COMPLETO (deterministico)
# ============================================================
if route == "IPZS_LIST":
    # Leggi file intero e costruisci elenco
    try:
        with open(IPZS_PERMESSI_PATH, "r", encoding="utf-8", errors="ignore") as f:
            ipzs_raw = f.read()
        sections = parse_ipzs_permessi_sections(ipzs_raw)

        if not sections:
            public_ans = hard_not_found()
        else:
            # elenco completo
            lines = []
            lines.append("Ecco l’**elenco dei permessi/giustificativi** presenti nel documento IPZS, con una breve spiegazione:")
            lines.append("")
            for s in sections:
                title = s["title"]
                desc = s["desc"]
                # prendi 1-3 righe utili
                desc_short = []
                for ln in desc.splitlines():
                    ln = ln.strip()
                    if not ln:
                        continue
                    # evita solo "Descrizione:"
                    if ln.lower() == "descrizione:":
                        continue
                    desc_short.append(ln)
                    if len(desc_short) >= 3:
                        break
                preview = " ".join(desc_short).strip()
                if not preview:
                    preview = "Vedi descrizione nel documento."
                lines.append(f"- **{title}**: {preview}")

            lines.append("")
            lines.append(ipzs_source_line())
            public_ans = "\n".join(lines).strip()

        assistant_payload = {"role": "assistant", "content": public_ans}

        if st.session_state.is_admin:
            # retrieval non serve qui: è elenco deterministico
            assistant_payload["debug"] = {
                "route": route,
                "best_similarity": None,
                "queries": [],
                "evidence": [],
                "pages_ccnl": [],
                "selected": [],
                "note": "Elenco IPZS generato deterministico dal file completo.",
            }

        st.session_state.messages.append(assistant_payload)
        st.rerun()
    except Exception:
        assistant_payload = {"role": "assistant", "content": hard_not_found()}
        st.session_state.messages.append(assistant_payload)
        st.rerun()


# ============================================================
# IPZS: PERMESSO SPECIFICO (RAO/ROL/altro)
# ============================================================
if route in ["IPZS_SINGLE", "MIXED"]:
    # prima prova IPZS
    selected_ipzs, best_ipzs, queries_ipzs, _ = retrieve_from_index(
        user_input, IPZS_VEC_PATH, IPZS_META_PATH, emb
    )
    retrieval_ok_ipzs = (best_ipzs >= MIN_BEST_SIMILARITY) and (len(selected_ipzs) >= MIN_SELECTED_CHUNKS)

    # carica tutte le sezioni per capire il "titolo" richiesto
    try:
        with open(IPZS_PERMESSI_PATH, "r", encoding="utf-8", errors="ignore") as f:
            ipzs_raw = f.read()
        sections = parse_ipzs_permessi_sections(ipzs_raw)
        titles = [s["title"] for s in sections]
        wanted = extract_specific_permesso_name(user_input, titles)
    except Exception:
        sections, titles, wanted = [], [], None

    if retrieval_ok_ipzs and sections:
        # Se l'utente ha chiesto un permesso specifico, rispondi SOLO quello
        if wanted:
            match = None
            # normalizza RAO/ROL
            if wanted == "RAO":
                for s in sections:
                    if s["title"].strip().lower() in ["rao", "r.a.o."]:
                        match = s
                        break
                if not match:
                    for s in sections:
                        if "rao" in s["title"].lower():
                            match = s
                            break
            elif wanted in ["R.O.L.", "ROL"]:
                for s in sections:
                    if "rol" in s["title"].lower() or "r.o.l" in s["title"].lower():
                        match = s
                        break
            else:
                for s in sections:
                    if s["title"].lower() == wanted.lower():
                        match = s
                        break
                if not match:
                    for s in sections:
                        if wanted.lower() in s["title"].lower():
                            match = s
                            break

            if match:
                public_ans = f"**{match['title']}**\n\n{match['desc'].strip()}\n\n{ipzs_source_line()}"
            else:
                # fallback: usa i chunk recuperati
                context = "\n\n---\n\n".join([c.get("text", "") for c in selected_ipzs[:6]])
                public_ans = f"Ho trovato riferimenti nel documento IPZS, ma non riesco a isolare la sezione esatta.\n\n{context}\n\n{ipzs_source_line()}"
        else:
            # Se NON è chiaro quale permesso, e la route è MIXED (generica),
            # allora non spariamo l'elenco: rispondi chiedendo parola chiave (ma senza bloccare troppo)
            if route == "MIXED" and ("permess" in user_input.lower()):
                public_ans = (
                    "Mi puoi dire **quale permesso** ti interessa (es. **RAO**, **ROL**, **donazione sangue**, **cure termali**, **104**, ecc.)?\n\n"
                    + ipzs_source_line()
                )
            else:
                public_ans = hard_not_found()

        assistant_payload = {"role": "assistant", "content": public_ans}

        if st.session_state.is_admin:
            assistant_payload["debug"] = {
                "route": route,
                "best_similarity": best_ipzs,
                "queries": queries_ipzs,
                "evidence": extract_evidence(selected_ipzs),
                "pages_ccnl": [],
                "selected": selected_ipzs[:6],
                "note": f"Permesso specifico richiesto: {wanted}",
            }

        st.session_state.messages.append(assistant_payload)
        st.rerun()

    # se IPZS debole e route MIXED, proviamo CCNL
    if route == "MIXED":
        # continua su CCNL sotto
        pass
    else:
        assistant_payload = {"role": "assistant", "content": hard_not_found()}
        st.session_state.messages.append(assistant_payload)
        st.rerun()


# ============================================================
# CCNL: Retrieval + LLM con citazioni
# ============================================================
selected_ccnl, best_ccnl, queries_ccnl, _ = retrieve_from_index(
    user_input, CCNL_VEC_PATH, CCNL_META_PATH, emb
)
retrieval_ok_ccnl = (best_ccnl >= MIN_BEST_SIMILARITY) and (len(selected_ccnl) >= MIN_SELECTED_CHUNKS)

if not retrieval_ok_ccnl:
    assistant_payload = {"role": "assistant", "content": "Non ho trovato la risposta nel CCNL caricato."}
    if st.session_state.is_admin:
        assistant_payload["debug"] = {
            "route": "CCNL",
            "best_similarity": best_ccnl,
            "queries": queries_ccnl,
            "evidence": extract_evidence(selected_ccnl),
            "pages_ccnl": unique_pages_ccnl(selected_ccnl),
            "selected": selected_ccnl[:6],
            "note": "retrieval_ok=False -> risposta bloccata",
        }
    st.session_state.messages.append(assistant_payload)
    st.rerun()

pages = unique_pages_ccnl(selected_ccnl, max_pages=8)
cit_line = format_ccnl_citations(pages)

context = "\n\n---\n\n".join([f"[Pagina {c.get('page','?')}] {c.get('text','')}" for c in selected_ccnl])

llm = ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE, api_key=OPENAI_API_KEY)

prompt = f"""
{RULES_CCNL}

DOMANDA (UTENTE):
{user_input}

CONTESTO CCNL (estratti):
{context}

SCRIVI UNA RISPOSTA CHIARA E PRATICA.
CHIUDI SEMPRE CON:
{cit_line}

FORMATO:
<PUBLIC>
...testo...
{cit_line}
</PUBLIC>
"""

try:
    raw = llm.invoke(prompt).content
except Exception as e:
    raw = f"<PUBLIC>Errore nel generare la risposta: {e}\n\n{cit_line}</PUBLIC>"

public_ans = split_public(raw).strip()

# Forza citazione se manca
if cit_line and ("fonte" not in public_ans.lower()):
    public_ans = public_ans.rstrip() + "\n\n" + cit_line

assistant_payload = {"role": "assistant", "content": public_ans}

if st.session_state.is_admin:
    assistant_payload["debug"] = {
        "route": "CCNL",
        "best_similarity": best_ccnl,
        "queries": queries_ccnl,
        "evidence": extract_evidence(selected_ccnl),
        "pages_ccnl": pages,
        "selected": selected_ccnl[:6],
        "bm25_available": BM25_AVAILABLE,
    }

st.session_state.messages.append(assistant_payload)
st.rerun()
INDEX_DIR = "index_uilcom"
VEC_PATH = os.path.join(INDEX_DIR, "vectors.npy")
META_PATH = os.path.join(INDEX_DIR, "chunks.json")

# Memoria (autoapprendimento controllato)
MEMORY_DIR = "memoria"
MEMORY_PATH = os.path.join(MEMORY_DIR, "memoria_uilcom.json")
MEMVEC_PATH = os.path.join(MEMORY_DIR, "memoria_vectors.npy")

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150

TOP_K_PER_QUERY = 10
TOP_K_FINAL = 14
MAX_MULTI_QUERIES = 10

# Memoria breve in chat (solo stesso topic)
MEMORY_USER_TURNS = 3

# Hard guardrail retrieval documenti
MIN_BEST_SIMILARITY = 0.24
MIN_SELECTED_CHUNKS = 3

# Autoapprendimento: soglia match su memoria validata admin
MEMORY_MATCH_THRESHOLD = 0.86  # alza/abbassa se necessario

# LLM
LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0


# ============================================================
# SECRETS
# ============================================================
def get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    try:
        if key in st.secrets:  # type: ignore
            return str(st.secrets[key])  # type: ignore
    except Exception:
        pass
    return os.getenv(key, default)

def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

UILCOM_PASSWORD = get_secret("UILCOM_PASSWORD")        # password iscritti
ADMIN_PASSWORD = get_secret("ADMIN_PASSWORD")          # password admin (B)
OPENAI_API_KEY = get_secret("OPENAI_API_KEY")          # obbligatoria


# ============================================================
# PAGE SETUP
# ============================================================
st.set_page_config(page_title="Assistente UILCOM IPZS", page_icon="🟦", layout="centered")

# Header (stile "chatgpt-like": pulito)
st.title(APP_TITLE)
st.markdown(
    "**Accesso riservato agli iscritti UILCOM**  \n"
    "Strumento informativo per consultare **CCNL Grafici Editoria** e **Documento IPZS (Permessi/Giustificativi)**.  \n\n"
    "⚠️ Le risposte sono generate **solo** in base ai documenti caricati. "
    "Le citazioni sono incluse per verifica diretta. "
    "Per casi specifici/interpretazioni: RSU/UILCOM o HR."
)
st.divider()


# ============================================================
# AUTH ISCRITTI
# ============================================================
if "auth_ok" not in st.session_state:
    st.session_state.auth_ok = False

if UILCOM_PASSWORD:
    with st.expander("🔒 Accesso iscritti UILCOM", expanded=not st.session_state.auth_ok):
        pwd_in = st.text_input("Password iscritti", type="password", placeholder="Inserisci password iscritti")
        if st.button("Entra", use_container_width=True):
            if pwd_in == UILCOM_PASSWORD:
                st.session_state.auth_ok = True
                st.success("Accesso consentito.")
            else:
                st.session_state.auth_ok = False
                st.error("Password non corretta.")
else:
    st.warning("UILCOM_PASSWORD non impostata. Imposta in Secrets (Streamlit) o variabile d’ambiente.")
    # sviluppo locale:
    # st.session_state.auth_ok = True

if not st.session_state.auth_ok:
    st.stop()


# ============================================================
# ADMIN (B) — password protetta
# ============================================================
if "is_admin" not in st.session_state:
    st.session_state.is_admin = False

with st.sidebar:
    st.header("⚙️ Controlli")

    st.subheader("🧠 Admin (protetto)")
    if ADMIN_PASSWORD:
        admin_in = st.text_input("Password admin", type="password", placeholder="Solo admin UILCOM", key="admin_pwd")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Login admin", use_container_width=True):
                if admin_in == ADMIN_PASSWORD:
                    st.session_state.is_admin = True
                    st.success("Admin attivo.")
                else:
                    st.session_state.is_admin = False
                    st.error("Password admin errata.")
        with c2:
            if st.button("Logout", use_container_width=True):
                st.session_state.is_admin = False
    else:
        st.caption("ADMIN_PASSWORD non impostata (Secrets).")

    st.divider()

    # Index management
    st.subheader("📦 Indice documenti")
    ok_index = os.path.exists(VEC_PATH) and os.path.exists(META_PATH)
    st.write("Indice presente:", "✅" if ok_index else "❌")

    if st.button("Indicizza / Reindicizza", use_container_width=True):
        try:
            with st.spinner("Indicizzazione in corso (CCNL + IPZS permessi)..."):
                if not os.path.exists(PDF_PATH):
                    raise FileNotFoundError(f"Non trovo il PDF: {PDF_PATH}")
                if not os.path.exists(IPZS_PERMESSI_TXT):
                    raise FileNotFoundError(f"Non trovo il TXT: {IPZS_PERMESSI_TXT}")

                if not OPENAI_API_KEY:
                    raise RuntimeError("Manca OPENAI_API_KEY in Secrets/variabili ambiente.")

                os.makedirs(INDEX_DIR, exist_ok=True)

                # ---- CCNL PDF ----
                loader = PyPDFLoader(PDF_PATH)
                docs_pdf = loader.load()  # each has metadata.page (0-based)

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
                )
                chunks_pdf = splitter.split_documents(docs_pdf)

                meta_all: List[Dict[str, Any]] = []
                texts_all: List[str] = []

                for c in chunks_pdf:
                    page_1based = int(c.metadata.get("page", 0)) + 1
                    text = c.page_content or ""
                    texts_all.append(text)
                    meta_all.append({
                        "source": "CCNL",
                        "page": page_1based,
                        "section": "",
                        "text": text
                    })

                # ---- IPZS TXT (Permessi) ----
                with open(IPZS_PERMESSI_TXT, "r", encoding="utf-8", errors="ignore") as f:
                    txt = f.read()

                # spezza in blocchi
                chunks_txt = splitter.create_documents([txt])
                # però vogliamo anche "sezioni" riconoscibili (titoli all caps) per citazioni migliori
                # aggiungiamo section heuristica per ogni chunk
                for c in chunks_txt:
                    t = c.page_content or ""
                    # cerca un titolo vicino all’inizio del chunk
                    m = re.search(r"(?m)^[A-Z0-9\.\'\(\)\s]{5,}$", t.strip())
                    section = (m.group(0).strip() if m else "IPZS_PERMESSI")
                    texts_all.append(t)
                    meta_all.append({
                        "source": "IPZS_PERMESSI",
                        "page": 0,
                        "section": section[:80],
                        "text": t
                    })

                # Embeddings
                emb = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
                vectors = emb.embed_documents(texts_all)
                vectors = np.array(vectors, dtype=np.float32)

                np.save(VEC_PATH, vectors)
                with open(META_PATH, "w", encoding="utf-8") as f:
                    json.dump(meta_all, f, ensure_ascii=False)

            st.success(f"Indicizzazione completata. Chunk totali: {len(meta_all)}")
            st.rerun()
        except Exception as e:
            st.error(str(e))

    if st.button("🧹 Nuova chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_topic = None
        st.rerun()

    st.divider()
    st.caption("Se Streamlit non aggiorna: Manage app → Reboot.")


# ============================================================
# HARD FAIL IF NO OPENAI KEY
# ============================================================
if not OPENAI_API_KEY:
    st.error(
        "Manca **OPENAI_API_KEY**.\n\n"
        "Streamlit Cloud: Settings → Secrets → OPENAI_API_KEY\n"
        "Locale: variabile d’ambiente OPENAI_API_KEY"
    )
    st.stop()


# ============================================================
# UTIL — VECTOR SEARCH
# ============================================================
def normalize_rows(mat: np.ndarray) -> np.ndarray:
    return mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)

def cosine_scores(query_vec: np.ndarray, mat_norm: np.ndarray) -> np.ndarray:
    q = query_vec / (np.linalg.norm(query_vec) + 1e-12)
    return mat_norm @ q

def load_index() -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    vectors = np.load(VEC_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    fixed = []
    for item in meta:
        if isinstance(item, dict) and "text" in item:
            fixed.append({
                "source": item.get("source", "?"),
                "page": item.get("page", 0),
                "section": item.get("section", ""),
                "text": item.get("text", "")
            })
        else:
            fixed.append({"source": "?", "page": 0, "section": "", "text": str(item)})
    return vectors, fixed

def unique_citations(chunks: List[Dict[str, Any]], max_items: int = 10) -> Dict[str, Any]:
    ccnl_pages: List[int] = []
    ipzs_sections: List[str] = []

    for c in chunks:
        src = c.get("source", "")
        if src == "CCNL":
            try:
                p = int(c.get("page", 0))
                if p and p not in ccnl_pages:
                    ccnl_pages.append(p)
            except Exception:
                pass
        elif src == "IPZS_PERMESSI":
            sec = (c.get("section", "") or "").strip()
            if sec and sec not in ipzs_sections:
                ipzs_sections.append(sec)

        if (len(ccnl_pages) + len(ipzs_sections)) >= max_items:
            break

    return {"ccnl_pages": sorted(ccnl_pages), "ipzs_sections": ipzs_sections[:max_items]}

def format_sources(cits: Dict[str, Any]) -> str:
    parts = []
    pages = cits.get("ccnl_pages", []) or []
    secs = cits.get("ipzs_sections", []) or []
    if pages:
        if len(pages) == 1:
            parts.append(f"**Fonte:** CCNL (pag. {pages[0]})")
        else:
            parts.append(f"**Fonte:** CCNL (pagg. {', '.join(map(str, pages))})")
    if secs:
        parts.append(f"**Fonte:** Documento IPZS Permessi (sez.: {', '.join(secs[:6])}{'…' if len(secs) > 6 else ''})")
    return "\n".join(parts).strip()


# ============================================================
# IPZS PERMESSI — PARSER DETERMINISTICO (PER ELENCO COMPLETO)
# ============================================================
def parse_ipzs_permessi_sections(txt_path: str) -> List[Dict[str, str]]:
    """
    Estrae sezioni dal TXT IPZS.
    Regola: riga titolo in MAIUSCOLO (o con punti) seguita da 'Descrizione:' e testo fino al prossimo titolo.
    """
    if not os.path.exists(txt_path):
        return []

    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()

    # normalizza
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")

    # individua i titoli (all caps)
    title_re = re.compile(r"(?m)^(?!FINE DOCUMENTO)([A-Z0-9][A-Z0-9\.\'\(\)\s]{3,})\s*$")
    titles = [(m.start(), m.group(1).strip()) for m in title_re.finditer(raw)]

    sections: List[Dict[str, str]] = []
    for i, (pos, title) in enumerate(titles):
        end = titles[i + 1][0] if i + 1 < len(titles) else len(raw)
        block = raw[pos:end].strip()

        # prova a togliere il titolo dal blocco
        block_wo_title = re.sub(r"(?m)^" + re.escape(title) + r"\s*$", "", block, count=1).strip()

        # prova a estrarre la descrizione da "Descrizione:"
        mdesc = re.search(r"(?is)descrizione\s*:\s*(.*)$", block_wo_title)
        desc = (mdesc.group(1).strip() if mdesc else block_wo_title.strip())

        # pulizia
        desc = re.sub(r"(?m)^\-+\s*$", "", desc).strip()
        desc = re.sub(r"\n{3,}", "\n\n", desc).strip()

        if len(title) >= 3 and len(desc) >= 10:
            sections.append({"title": title, "desc": desc})

    # de-dup titoli
    out, seen = [], set()
    for s in sections:
        key = s["title"].strip().lower()
        if key not in seen:
            out.append(s)
            seen.add(key)

    return out

def is_generic_permessi_question(q: str) -> bool:
    ql = q.lower()
    patterns = [
        "a quali permessi ha diritto",
        "quali permessi ha diritto",
        "quali permessi ha a disposizione",
        "elenco permessi",
        "tutti i permessi",
        "lista permessi",
        "che permessi ci sono",
        "tipologie di permessi",
        "permessi disponibili",
        "permessi ipzs",
    ]
    return any(p in ql for p in patterns)

def find_permesso_specific(sections: List[Dict[str, str]], user_q: str) -> Optional[Dict[str, str]]:
    ql = user_q.lower().strip()

    # se l’utente scrive RAO/ROL in modo secco
    direct_keys = ["rao", "r.a.o", "rol", "r.o.l", "donazione sangue", "permessi elettorali", "testimonianza civile"]
    for dk in direct_keys:
        if dk in ql:
            # cerca in titoli
            for s in sections:
                if dk.replace(".", "") in s["title"].lower().replace(".", ""):
                    return s

    # fallback: match per parole significative
    tokens = [t for t in re.split(r"[^a-zA-Z0-9àèéìòù]+", ql) if len(t) >= 3]
    if not tokens:
        return None

    best = None
    best_score = 0
    for s in sections:
        title_low = s["title"].lower()
        desc_low = s["desc"].lower()
        score = 0
        for t in tokens:
            if t in title_low:
                score += 3
            elif t in desc_low:
                score += 1
        if score > best_score:
            best_score = score
            best = s

    # soglia minima
    if best and best_score >= 3:
        return best
    return None

def render_permessi_list(sections: List[Dict[str, str]]) -> str:
    if not sections:
        return "Non ho trovato l’elenco dei permessi nel Documento IPZS caricato."

    lines = []
    lines.append("Ecco l’**elenco dei permessi/giustificativi** presenti nel **Documento IPZS** (con descrizione sintetica):\n")
    for i, s in enumerate(sections, start=1):
        title = s["title"].strip()
        desc = " ".join(s["desc"].split())
        # taglia descrizione per non fare muri
        if len(desc) > 260:
            desc = desc[:260].rstrip() + "…"
        lines.append(f"**{i}. {title}**  \n{desc}\n")
    lines.append("**Fonte:** Documento IPZS Permessi (elenco completo)")
    return "\n".join(lines).strip()

def render_permesso_specific(s: Dict[str, str]) -> str:
    title = s["title"].strip()
    desc = s["desc"].strip()
    return f"**{title}**\n\n{desc}\n\n**Fonte:** Documento IPZS Permessi (sez.: {title})"


# ============================================================
# TOPIC DETECTOR (per memoria breve e guardrail)
# ============================================================
MALATTIA_TRIGGERS = ["malattia", "comporto", "visita fiscale", "reperibil", "ricovero", "infortunio"]
PERMESSI_TRIGGERS = ["permess", "rao", "rol", "donazione", "elettoral", "lutto", "matrimonio", "studio", "esami", "104"]
STRAORD_TRIGGERS = ["straordin", "maggioraz", "notturn", "festiv"]
MANSIONI_TRIGGERS = ["mansioni superiori", "categoria superiore", "inquadramento superiore", "30 giorni", "60 giorni"]

def detect_topic(q: str) -> str:
    ql = q.lower()
    if any(t in ql for t in MALATTIA_TRIGGERS):
        return "malattia"
    if any(t in ql for t in MANSIONI_TRIGGERS):
        return "mansioni"
    if any(t in ql for t in PERMESSI_TRIGGERS):
        return "permessi"
    if any(t in ql for t in STRAORD_TRIGGERS):
        return "straordinari"
    return "altro"

def build_enriched_question(current_q: str, current_topic: str) -> str:
    if "messages" not in st.session_state:
        return current_q.strip()

    last_topic = st.session_state.get("last_topic", None)
    if last_topic and last_topic != current_topic:
        return current_q.strip()

    user_msgs = [m["content"] for m in st.session_state.messages if m.get("role") == "user" and m.get("content")]
    prev = user_msgs[:-1] if (user_msgs and user_msgs[-1].strip() == current_q.strip()) else user_msgs
    last = prev[-MEMORY_USER_TURNS:] if prev else []
    last = [x.strip() for x in last if x.strip()]
    if not last:
        return current_q.strip()

    return (
        "CONTESTO CONVERSAZIONE (ultime richieste utente, stesso argomento):\n"
        + "\n".join([f"- {x}" for x in last])
        + "\n\nDOMANDA ATTUALE:\n"
        + current_q.strip()
    )


# ============================================================
# QUERY BUILDER
# ============================================================
def build_queries(q: str) -> List[str]:
    q0 = q.strip()
    ql = q0.lower()

    qs = [q0, f"{q0} CCNL", f"{q0} documento IPZS permessi"]

    if "permess" in ql or "rao" in ql or "rol" in ql:
        qs += [
            "permessi IPZS elenco permessi giustificativi",
            "RAO permesso retribuito IPZS",
            "ROL riduzione orario permesso retribuito",
            "donazione sangue permesso giornaliero retribuito",
            "permessi elettorali permesso giornaliero retribuito",
        ]

    if "malatt" in ql:
        qs += [
            "malattia trattamento economico integrazione",
            "malattia periodo di comporto conservazione posto",
            "malattia reperibilità visite fiscali",
        ]

    if "straordin" in ql or "notturn" in ql:
        qs += [
            "lavoro straordinario maggiorazioni",
            "lavoro notturno maggiorazioni",
            "lavoro festivo maggiorazioni",
        ]

    if "mansioni superiori" in ql or "inquadramento superiore" in ql:
        qs += [
            "mansioni superiori 30 giorni consecutivi 60 giorni non consecutivi",
            "mansioni superiori trattamento corrispondente",
            "sostituzione conservazione del posto mansioni superiori",
        ]

    out, seen = [], set()
    for x in qs:
        x = x.strip()
        if x and x not in seen:
            out.append(x)
            seen.add(x)

    return out[:MAX_MULTI_QUERIES]


# ============================================================
# BM25 RERANK
# ============================================================
def bm25_rerank(query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not BM25_AVAILABLE or not candidates:
        return candidates
    corpus = [(c.get("text") or "").lower().split() for c in candidates]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(query.lower().split())
    idx = np.argsort(-np.array(scores))
    return [candidates[int(i)] for i in idx]


# ============================================================
# MEMORIA UILCOM (autoapprendimento controllato)
# ============================================================
def ensure_memory_files():
    os.makedirs(MEMORY_DIR, exist_ok=True)
    if not os.path.exists(MEMORY_PATH):
        with open(MEMORY_PATH, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False)
    if not os.path.exists(MEMVEC_PATH):
        np.save(MEMVEC_PATH, np.zeros((0, 1536), dtype=np.float32))  # dimensione tipica embeddings

def load_memory() -> Tuple[List[Dict[str, Any]], np.ndarray]:
    ensure_memory_files()
    with open(MEMORY_PATH, "r", encoding="utf-8") as f:
        items = json.load(f)
    try:
        vecs = np.load(MEMVEC_PATH)
    except Exception:
        vecs = np.zeros((0, 1536), dtype=np.float32)
    if not isinstance(items, list):
        items = []
    return items, vecs

def save_memory(items: List[Dict[str, Any]], vecs: np.ndarray):
    ensure_memory_files()
    with open(MEMORY_PATH, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    np.save(MEMVEC_PATH, vecs.astype(np.float32))

def memory_lookup(user_q: str, emb: OpenAIEmbeddings) -> Tuple[Optional[Dict[str, Any]], float]:
    items, vecs = load_memory()
    if vecs.shape[0] == 0 or len(items) == 0:
        return None, 0.0

    qvec = np.array(emb.embed_query(user_q), dtype=np.float32)
    mat = normalize_rows(vecs)
    sims = cosine_scores(qvec, mat)
    best_i = int(np.argmax(sims))
    best_s = float(sims[best_i])

    if best_s >= MEMORY_MATCH_THRESHOLD:
        return items[best_i], best_s
    return None, best_s


# ============================================================
# SYSTEM RULES
# ============================================================
RULES = """
Sei l’assistente UILCOM per lavoratori IPZS.
Devi rispondere in modo chiaro e pratico basandoti SOLO su:
- Contesto estratto dal CCNL (PDF)
- Contesto estratto dal Documento IPZS Permessi (TXT)
- Memoria UILCOM SOLO se "validata admin"

Non inventare informazioni.
Se non trovi nel contesto, scrivi: "Non ho trovato la risposta nei documenti caricati."

FORMATO:
Risposta diretta + citazioni finali (Fonte ...).
"""


# ============================================================
# CHAT STATE
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_topic" not in st.session_state:
    st.session_state.last_topic = None

# Render chat
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

        # Admin tools: correzione risposta
        if st.session_state.is_admin and m["role"] == "assistant":
            dbg = m.get("debug", None)
            if dbg:
                with st.expander("🧠 Admin debug", expanded=False):
                    st.write("**Topic:**", dbg.get("topic"))
                    st.write("**Best similarity docs:**", dbg.get("best_similarity"))
                    st.write("**Query usate:**")
                    st.code("\n".join(dbg.get("queries", [])))
                    st.write("**Citazioni:**")
                    st.code(dbg.get("cit_line", ""))

                # Pulsante correzione (autoapprendimento controllato)
                with st.expander("✅ Correggi e salva in memoria UILCOM (autoapprendimento)", expanded=False):
                    corr = st.text_area("Incolla qui la risposta corretta (quella che vuoi diventi definitiva)", value="", height=160, key=f"corr_{dbg.get('msg_id','')}")
                    if st.button("Salva risposta corretta", key=f"save_{dbg.get('msg_id','')}"):
                        try:
                            emb = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
                            items, vecs = load_memory()

                            q = dbg.get("user_q", "").strip()
                            if not q:
                                st.error("Domanda utente non trovata nel debug.")
                            elif not corr.strip():
                                st.error("Devi inserire una risposta corretta.")
                            else:
                                qvec = np.array(emb.embed_query(q), dtype=np.float32).reshape(1, -1)
                                # append
                                items.append({
                                    "question": q,
                                    "answer": corr.strip(),
                                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                                    "topic": dbg.get("topic", ""),
                                    "source_note": "Memoria UILCOM (validata admin)"
                                })
                                # vecs could be empty or different dim; handle robust
                                if vecs.shape[0] == 0:
                                    vecs = qvec
                                else:
                                    # if dim mismatch, re-init
                                    if vecs.shape[1] != qvec.shape[1]:
                                        vecs = np.vstack([vecs[:, :qvec.shape[1]], qvec]) if vecs.shape[1] > qvec.shape[1] else np.vstack([vecs, qvec])
                                    else:
                                        vecs = np.vstack([vecs, qvec])

                                save_memory(items, vecs)
                                st.success("Salvato ✅ Da ora la chatbot risponderà prima con questa risposta (se domanda simile).")
                        except Exception as e:
                            st.error(str(e))


user_input = st.chat_input("Scrivi una domanda (permessi IPZS, RAO/ROL, malattia, straordinari, ecc.)")

if not user_input:
    st.stop()

# Require index
if not (os.path.exists(VEC_PATH) and os.path.exists(META_PATH)):
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Prima devo indicizzare i documenti: apri la barra laterale e clicca **Indicizza / Reindicizza**.",
    })
    st.rerun()

# Append user msg
st.session_state.messages.append({"role": "user", "content": user_input})


# ============================================================
# PIPELINE
# ============================================================
topic = detect_topic(user_input)
enriched_q = build_enriched_question(user_input, topic)

emb = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

# 1) AUTOAPPRENDIMENTO (memoria validata admin) — PRIMA di tutto
mem_hit, mem_sim = memory_lookup(user_input, emb)
if mem_hit is not None:
    answer = mem_hit.get("answer", "").strip()
    stamp = mem_hit.get("timestamp", "")
    src_line = f"**Fonte:** Memoria UILCOM (validata admin, {stamp})"
    content = (answer + "\n\n" + src_line).strip()

    payload = {"role": "assistant", "content": content}
    if st.session_state.is_admin:
        payload["debug"] = {
            "msg_id": sha256_text(user_input + str(datetime.now().timestamp()))[:10],
            "topic": topic,
            "user_q": user_input,
            "queries": ["MEMORIA UILCOM (lookup)"],
            "best_similarity": mem_sim,
            "cit_line": src_line,
        }

    st.session_state.last_topic = topic
    st.session_state.messages.append(payload)
    st.rerun()

# 2) CASO SPECIALE: PERMESSI IPZS — elenco completo o singolo permesso (deterministico)
ipzs_sections = parse_ipzs_permessi_sections(IPZS_PERMESSI_TXT)

if topic == "permessi":
    # se domanda generica -> elenco completo
    if is_generic_permessi_question(user_input):
        content = render_permessi_list(ipzs_sections)

        payload = {"role": "assistant", "content": content}
        if st.session_state.is_admin:
            payload["debug"] = {
                "msg_id": sha256_text(user_input + str(datetime.now().timestamp()))[:10],
                "topic": topic,
                "user_q": user_input,
                "queries": ["IPZS TXT (parser elenco completo)"],
                "best_similarity": None,
                "cit_line": "**Fonte:** Documento IPZS Permessi (elenco completo)",
            }

        st.session_state.last_topic = topic
        st.session_state.messages.append(payload)
        st.rerun()

    # se domanda specifica -> prova a prendere una singola sezione IPZS
    spec = find_permesso_specific(ipzs_sections, user_input)
    if spec is not None:
        content = render_permesso_specific(spec)

        payload = {"role": "assistant", "content": content}
        if st.session_state.is_admin:
            payload["debug"] = {
                "msg_id": sha256_text(user_input + str(datetime.now().timestamp()))[:10],
                "topic": topic,
                "user_q": user_input,
                "queries": ["IPZS TXT (parser permesso specifico)"],
                "best_similarity": None,
                "cit_line": f"**Fonte:** Documento IPZS Permessi (sez.: {spec['title']})",
            }

        st.session_state.last_topic = topic
        st.session_state.messages.append(payload)
        st.rerun()

# 3) RETRIEVAL SU INDICE (CCNL + IPZS chunks) + LLM
vectors, meta = load_index()
mat_norm = normalize_rows(vectors)
queries = build_queries(enriched_q)

scores_best: Dict[int, float] = {}
best_similarity = 0.0

for q in queries:
    qvec = np.array(emb.embed_query(q), dtype=np.float32)
    sims = cosine_scores(qvec, mat_norm)
    top_idx = np.argsort(-sims)[:TOP_K_PER_QUERY]
    for i in top_idx:
        s = float(sims[int(i)])
        if s > best_similarity:
            best_similarity = s
        if (int(i) not in scores_best) or (s > scores_best[int(i)]):
            scores_best[int(i)] = s

final_idx = sorted(scores_best.keys(), key=lambda i: scores_best[i], reverse=True)[:TOP_K_FINAL]
selected = [meta[i] for i in final_idx]

selected = bm25_rerank(enriched_q, selected)

retrieval_ok = (best_similarity >= MIN_BEST_SIMILARITY) and (len(selected) >= MIN_SELECTED_CHUNKS)

cits = unique_citations(selected, max_items=10)
cit_line = format_sources(cits)

def hard_not_found_message() -> str:
    return "Non ho trovato la risposta nei documenti caricati."

if not retrieval_ok:
    payload = {"role": "assistant", "content": hard_not_found_message()}
    if st.session_state.is_admin:
        payload["debug"] = {
            "msg_id": sha256_text(user_input + str(datetime.now().timestamp()))[:10],
            "topic": topic,
            "user_q": user_input,
            "queries": queries,
            "best_similarity": best_similarity,
            "cit_line": cit_line,
            "note": "retrieval_ok=False -> risposta bloccata",
        }
    st.session_state.last_topic = topic
    st.session_state.messages.append(payload)
    st.rerun()

# LLM context
context = "\n\n---\n\n".join(
    [f"[{c.get('source','?')} | Pagina {c.get('page','?')} | Sez {c.get('section','')}] {c.get('text','')}" for c in selected]
)

llm = ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE, api_key=OPENAI_API_KEY)

prompt = f"""
{RULES}

DOMANDA (UTENTE):
{user_input}

DOMANDA ARRICCHITA (MEMORIA BREVE - solo stesso topic):
{enriched_q}

CONTESTO ESTRATTO DAI DOCUMENTI:
{context}

ISTRUZIONI IMPORTANTI:
- Rispondi SOLO usando ciò che trovi nel contesto.
- Se non c'è, di' che non hai trovato nei documenti caricati.
- Chiudi SEMPRE con le fonti (CCNL pagine / IPZS sezioni) se disponibili.
"""

try:
    raw = llm.invoke(prompt).content
except Exception as e:
    raw = f"Errore nel generare la risposta: {e}"

public_ans = (raw or "").strip()
if cit_line and ("fonte" not in public_ans.lower()):
    public_ans = public_ans.rstrip() + "\n\n" + cit_line

payload = {"role": "assistant", "content": public_ans}
if st.session_state.is_admin:
    payload["debug"] = {
        "msg_id": sha256_text(user_input + str(datetime.now().timestamp()))[:10],
        "topic": topic,
        "user_q": user_input,
        "queries": queries,
        "best_similarity": best_similarity,
        "cit_line": cit_line,
    }

st.session_state.last_topic = topic
st.session_state.messages.append(payload)
st.rerun()# Logo (opzionale)
LOGO_PATH_1 = os.path.join("logo", "logo_uilcom.png")   # consigliato
LOGO_PATH_2 = "logo_uilcom.png"                         # fallback se lo metti in root

# Indici
INDEX_CCNL_DIR = "index_ccnl"
CCNL_VEC_PATH = os.path.join(INDEX_CCNL_DIR, "vectors.npy")
CCNL_META_PATH = os.path.join(INDEX_CCNL_DIR, "chunks.json")

INDEX_IPZS_DIR = "index_ipzs_permessi"
IPZS_VEC_PATH = os.path.join(INDEX_IPZS_DIR, "vectors.npy")
IPZS_META_PATH = os.path.join(INDEX_IPZS_DIR, "chunks.json")

# Chunking
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150

TOP_K_PER_QUERY = 12
TOP_K_FINAL = 18
MAX_MULTI_QUERIES = 10

# Guardrail retrieval
MIN_BEST_SIMILARITY = 0.24
MIN_SELECTED_CHUNKS = 3

# Admin
MAX_EVIDENCE_LINES = 18

# LLM
LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0


# ============================================================
# SECRETS
# ============================================================
def get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    try:
        if key in st.secrets:  # type: ignore
            return str(st.secrets[key])  # type: ignore
    except Exception:
        pass
    return os.getenv(key, default)

def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

UILCOM_PASSWORD = get_secret("UILCOM_PASSWORD")        # password iscritti
ADMIN_PASSWORD = get_secret("ADMIN_PASSWORD")          # password admin
OPENAI_API_KEY = get_secret("OPENAI_API_KEY")          # obbligatoria


# ============================================================
# UI SETUP
# ============================================================
st.set_page_config(page_title="Assistente UILCOM IPZS", page_icon="🟦", layout="centered")

# Logo (non deve mai rompere l'app)
logo_to_use = LOGO_PATH_1 if os.path.exists(LOGO_PATH_1) else (LOGO_PATH_2 if os.path.exists(LOGO_PATH_2) else None)
if logo_to_use:
    try:
        st.image(logo_to_use, width=140)
    except Exception:
        pass

st.title(APP_TITLE)
st.markdown(
    "**Accesso riservato agli iscritti UILCOM**  \n"
    "Strumento informativo per facilitare la consultazione del **CCNL** e del documento **IPZS Permessi/Giustificativi**.  \n\n"
    "⚠️ Le risposte sono generate **solo** in base ai documenti caricati. "
    "Per casi specifici o interpretazioni, rivolgersi a RSU/UILCOM o HR."
)
st.divider()


# ============================================================
# AUTH: ISCRITTI
# ============================================================
if "auth_ok" not in st.session_state:
    st.session_state.auth_ok = False

if UILCOM_PASSWORD:
    with st.expander("🔒 Accesso iscritti UILCOM", expanded=not st.session_state.auth_ok):
        pwd_in = st.text_input("Password iscritti", type="password", placeholder="Inserisci password iscritti")
        if st.button("Entra", use_container_width=True):
            if pwd_in == UILCOM_PASSWORD:
                st.session_state.auth_ok = True
                st.success("Accesso consentito.")
            else:
                st.session_state.auth_ok = False
                st.error("Password non corretta.")
else:
    st.warning("Password iscritti non impostata. Imposta UILCOM_PASSWORD in Secrets.")
    # st.session_state.auth_ok = True  # per sviluppo locale

if not st.session_state.auth_ok:
    st.stop()


# ============================================================
# ADMIN SIDEBAR
# ============================================================
if "is_admin" not in st.session_state:
    st.session_state.is_admin = False

with st.sidebar:
    st.header("⚙️ Controlli")

    st.subheader("🧠 Admin (debug)")
    if ADMIN_PASSWORD:
        admin_in = st.text_input("Password admin", type="password", placeholder="Solo admin UILCOM", key="admin_pwd")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Login admin", use_container_width=True):
                if admin_in == ADMIN_PASSWORD:
                    st.session_state.is_admin = True
                    st.success("Admin attivo.")
                else:
                    st.session_state.is_admin = False
                    st.error("Password admin errata.")
        with c2:
            if st.button("Logout", use_container_width=True):
                st.session_state.is_admin = False
    else:
        st.caption("ADMIN_PASSWORD non impostata (Secrets).")

    st.divider()

    st.subheader("📦 Indici documenti")
    ok_ccnl = os.path.exists(CCNL_VEC_PATH) and os.path.exists(CCNL_META_PATH)
    ok_ipzs = os.path.exists(IPZS_VEC_PATH) and os.path.exists(IPZS_META_PATH)

    st.write("Indice CCNL:", "✅" if ok_ccnl else "❌")
    st.write("Indice IPZS Permessi:", "✅" if ok_ipzs else "❌")

    if st.button("Indicizza / Reindicizza (CCNL + IPZS)", use_container_width=True):
        if not OPENAI_API_KEY:
            st.error("Manca OPENAI_API_KEY nei Secrets.")
        else:
            try:
                with st.spinner("Indicizzazione in corso..."):
                    os.makedirs(INDEX_CCNL_DIR, exist_ok=True)
                    os.makedirs(INDEX_IPZS_DIR, exist_ok=True)

                    emb = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
                    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

                    # ---- CCNL PDF ----
                    if not os.path.exists(PDF_PATH):
                        raise FileNotFoundError(f"Non trovo il PDF CCNL: {PDF_PATH}")

                    loader = PyPDFLoader(PDF_PATH)
                    docs = loader.load()
                    ccnl_chunks = splitter.split_documents(docs)

                    ccnl_texts = [c.page_content for c in ccnl_chunks]
                    ccnl_pages = [(int(c.metadata.get("page", 0)) + 1) for c in ccnl_chunks]  # 1-based

                    ccnl_vectors = np.array(emb.embed_documents(ccnl_texts), dtype=np.float32)
                    np.save(CCNL_VEC_PATH, ccnl_vectors)
                    with open(CCNL_META_PATH, "w", encoding="utf-8") as f:
                        json.dump([{"page": p, "text": t} for p, t in zip(ccnl_pages, ccnl_texts)], f, ensure_ascii=False)

                    # ---- IPZS Permessi TXT ----
                    if not os.path.exists(IPZS_PERMESSI_PATH):
                        raise FileNotFoundError(f"Non trovo IPZS Permessi: {IPZS_PERMESSI_PATH}")

                    with open(IPZS_PERMESSI_PATH, "r", encoding="utf-8", errors="ignore") as f:
                        ipzs_raw = f.read()

                    # chunk a testo con "pseudo-page" (sezione)
                    # Spezziamo comunque con splitter per retrieval robusto
                    ipzs_docs = [{"page": "IPZS", "text": ipzs_raw}]
                    ipzs_texts = [ipzs_raw]
                    # split
                    ipzs_chunks = splitter.split_text(ipzs_raw)
                    ipzs_vectors = np.array(emb.embed_documents(ipzs_chunks), dtype=np.float32)
                    np.save(IPZS_VEC_PATH, ipzs_vectors)
                    with open(IPZS_META_PATH, "w", encoding="utf-8") as f:
                        json.dump([{"page": "IPZS", "text": t} for t in ipzs_chunks], f, ensure_ascii=False)

                st.success("Indicizzazione completata ✅")
                st.rerun()
            except Exception as e:
                st.error(str(e))

    if st.button("🧹 Nuova chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("Suggerimento: dopo aggiornamenti su GitHub → Streamlit Cloud fa auto-redeploy. Se no: **Reboot app**.")


# ============================================================
# HARD FAIL IF NO OPENAI KEY
# ============================================================
if not OPENAI_API_KEY:
    st.error(
        "Manca la variabile **OPENAI_API_KEY**.\n\n"
        "Streamlit Cloud: **Settings → Secrets → OPENAI_API_KEY**"
    )
    st.stop()


# ============================================================
# HELPERS: VECTOR SEARCH
# ============================================================
def normalize_rows(mat: np.ndarray) -> np.ndarray:
    return mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)

def cosine_scores(query_vec: np.ndarray, mat_norm: np.ndarray) -> np.ndarray:
    q = query_vec / (np.linalg.norm(query_vec) + 1e-12)
    return mat_norm @ q

def load_index(vec_path: str, meta_path: str) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    vectors = np.load(vec_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    fixed = []
    for item in meta:
        if isinstance(item, dict) and "text" in item:
            fixed.append({"page": item.get("page", "?"), "text": item.get("text", "")})
        else:
            fixed.append({"page": "?", "text": str(item)})
    return vectors, fixed

def bm25_rerank(query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not BM25_AVAILABLE or not candidates:
        return candidates
    corpus = [(c.get("text") or "").lower().split() for c in candidates]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(query.lower().split())
    idx = np.argsort(-np.array(scores))
    return [candidates[int(i)] for i in idx]

def unique_pages_ccnl(chunks: List[Dict[str, Any]], max_pages: int = 8) -> List[int]:
    pages: List[int] = []
    for c in chunks:
        try:
            p = int(c.get("page", 0))
        except Exception:
            continue
        if p and p not in pages:
            pages.append(p)
        if len(pages) >= max_pages:
            break
    return pages

def format_ccnl_citations(pages: List[int]) -> str:
    if not pages:
        return ""
    pages_sorted = sorted(pages)
    if len(pages_sorted) == 1:
        return f"**Fonte:** CCNL (pag. {pages_sorted[0]})"
    return f"**Fonte:** CCNL (pagg. {', '.join(map(str, pages_sorted))})"

def extract_evidence(chunks: List[Dict[str, Any]], max_lines: int = 18) -> List[str]:
    patterns = [
        r"rao", r"r\.a\.o", r"rol", r"r\.o\.l", r"riduzion",
        r"permess", r"conged", r"retribuit", r"non retribuit",
        r"donazion", r"decesso", r"lutto", r"studio", r"esam", r"104",
        r"cure termali", r"elettoral", r"testimonianza",
    ]
    out = []
    for c in chunks:
        page = c.get("page", "?")
        text = c.get("text", "") or ""
        for ln in [x.strip() for x in text.splitlines() if x.strip()]:
            low = ln.lower()
            if any(re.search(p, low) for p in patterns):
                clean = " ".join(ln.split())
                if 20 <= len(clean) <= 420:
                    out.append(f"({page}) {clean}")
            if len(out) >= max_lines:
                return out
    return out[:max_lines]


# ============================================================
# IPZS PERMESSI: PARSER ELENCO COMPLETO (deterministico)
# ============================================================
def parse_ipzs_permessi_sections(raw_text: str) -> List[Dict[str, str]]:
    """
    Prova a ricostruire un elenco di "permessi" dal file IPZS:
    - Riconosce titoli in MAIUSCOLO (es: DONAZIONE SANGUE, RAO, R.O.L., CURE TERMALI)
    - Prende il blocco "Descrizione:" sotto al titolo fino al prossimo titolo
    """
    lines = raw_text.splitlines()
    sections: List[Dict[str, str]] = []

    def is_title(line: str) -> bool:
        s = line.strip()
        if not s:
            return False
        # escludi righe decorative
        if set(s) <= set("-_= "):
            return False
        # titolo: molti caratteri maiuscoli + pochi segni
        # oppure RAO / R.O.L.
        if s in ["RAO", "R.O.L.", "ROL", "R.A.O."]:
            return True
        # se è quasi tutto uppercase
        letters = [ch for ch in s if ch.isalpha()]
        if len(letters) >= 6 and sum(ch.isupper() for ch in letters) / max(len(letters), 1) > 0.8:
            return True
        # contiene pattern tipici
        if re.search(r"\bL\.\s*\d+/\d+\b", s):
            return True
        return False

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if is_title(line):
            title = line
            # salta eventuali righe decorative sotto
            j = i + 1
            while j < len(lines) and (lines[j].strip() == "" or set(lines[j].strip()) <= set("-_= ")):
                j += 1

            # cerca descrizione
            desc_lines = []
            # se c'è "Descrizione:" la includiamo e prendiamo le righe successive
            if j < len(lines) and lines[j].strip().lower().startswith("descrizione"):
                desc_lines.append(lines[j].strip())
                j += 1

            while j < len(lines):
                nxt = lines[j].strip()
                if is_title(nxt):
                    break
                desc_lines.append(lines[j].rstrip())
                j += 1

            desc = "\n".join([x for x in desc_lines if x.strip()]).strip()
            if desc:
                sections.append({"title": title.strip(), "desc": desc})
            i = j
        else:
            i += 1

    # dedup titoli
    out = []
    seen = set()
    for s in sections:
        t = re.sub(r"\s+", " ", s["title"]).strip()
        key = t.lower()
        if key not in seen:
            out.append({"title": t, "desc": s["desc"]})
            seen.add(key)
    return out

def ipzs_source_line() -> str:
    return "**Fonte:** Documento IPZS Permessi/Giustificativi (PERMESSI_IPZS_COMPLETO_FINALE.txt)"

def is_generic_permessi_list_question(q: str) -> bool:
    ql = q.lower()
    triggers = [
        "a quali permessi ha diritto",
        "quali permessi ha diritto",
        "quali permessi ha a disposizione",
        "elenco permessi",
        "tutti i permessi",
        "lista permessi",
        "permessi disponibili",
        "permessi ipzs",
        "giustificativi",
    ]
    return any(t in ql for t in triggers)

def extract_specific_permesso_name(q: str, sections_titles: List[str]) -> Optional[str]:
    """
    Se l'utente chiede un permesso specifico, proviamo a matchare:
    - RAO / ROL
    - parole contenute in un titolo
    """
    ql = q.lower()

    # priorità RAO / ROL
    if re.search(r"\brao\b|\br\.a\.o\b", ql):
        return "RAO"
    if re.search(r"\brol\b|\br\.o\.l\b", ql):
        return "R.O.L."

    # match per inclusione (titolo dentro domanda o viceversa)
    for t in sections_titles:
        tl = t.lower()
        # evita titoli troppo generici
        if len(tl) < 4:
            continue
        if tl in ql:
            return t
    # match per parole chiave importanti
    keywords_map = {
        "donazione": ["DONAZIONE SANGUE", "DONAZIONE"],
        "sangue": ["DONAZIONE SANGUE"],
        "cure termali": ["CURE TERMALI"],
        "elettoral": ["PERMESSI ELETTORALI"],
        "decesso": ["DECESSO", "DECESSO FAMILIARI"],
        "testimon": ["TESTIMONIANZA CIVILE"],
        "esami": ["PERM. ESAMI", "ESAMI"],
        "104": ["L. 104/92", "L.104/92", "104"],
        "maternit": ["MATERNITA' OBBLIGATORIA", "MATERNITA'"],
        "padre": ["CONG.OBBLIGATORIO PADRE"],
        "nido": ["INSERIM.NIDO", "ACC.NIDO", "NIDO"],
    }
    for k, cand in keywords_map.items():
        if k in ql:
            # ritorna il primo che esiste davvero tra i titoli
            for c in cand:
                for real in sections_titles:
                    if c.lower() in real.lower() or real.lower() in c.lower():
                        return real
    return None


# ============================================================
# ROUTING: capire se rispondere con IPZS o CCNL
# ============================================================
def detect_route(q: str) -> str:
    """
    - Se domanda è lista permessi -> IPZS (elenco completo)
    - Se domanda su permesso specifico (rao/rol/donazione ecc) -> IPZS
    - Altrimenti -> CCNL
    """
    ql = q.lower()
    if is_generic_permessi_list_question(ql):
        return "IPZS_LIST"

    # segnali di permesso specifico
    if any(x in ql for x in ["rao", "r.a.o", "rol", "r.o.l", "donazione", "cure termali", "permessi elettorali", "testimonianza", "decesso", "104", "perm. esami", "maternit", "padre", "nido"]):
        return "IPZS_SINGLE"

    # se contiene "permessi" ma non chiede elenco, proviamo IPZS prima
    if "permess" in ql or "giustific" in ql:
        return "MIXED"

    return "CCNL"


# ============================================================
# MULTI-QUERY BUILDER
# ============================================================
def build_queries(q: str) -> List[str]:
    q0 = q.strip()
    qs = [q0, f"{q0} CCNL", f"{q0} regole condizioni", f"{q0} definizione procedura"]
    out, seen = [], set()
    for x in qs:
        x = x.strip()
        if x and x not in seen:
            out.append(x)
            seen.add(x)
    return out[:MAX_MULTI_QUERIES]


# ============================================================
# RETRIEVE (generic)
# ============================================================
def retrieve_from_index(
    query: str,
    vec_path: str,
    meta_path: str,
    emb: OpenAIEmbeddings,
    top_k_per_query: int = TOP_K_PER_QUERY,
    top_k_final: int = TOP_K_FINAL,
) -> Tuple[List[Dict[str, Any]], float, List[str], List[Dict[str, Any]]]:
    vectors, meta = load_index(vec_path, meta_path)
    mat_norm = normalize_rows(vectors)

    queries = build_queries(query)

    scores_best: Dict[int, float] = {}
    best_similarity = 0.0

    for q in queries:
        qvec = np.array(emb.embed_query(q), dtype=np.float32)
        sims = cosine_scores(qvec, mat_norm)
        top_idx = np.argsort(-sims)[:top_k_per_query]
        for i in top_idx:
            s = float(sims[int(i)])
            best_similarity = max(best_similarity, s)
            if (int(i) not in scores_best) or (s > scores_best[int(i)]):
                scores_best[int(i)] = s

    final_idx = sorted(scores_best.keys(), key=lambda i: scores_best[i], reverse=True)[:top_k_final]
    selected = [meta[i] for i in final_idx]
    selected = bm25_rerank(query, selected)

    evidence = extract_evidence(selected, max_lines=MAX_EVIDENCE_LINES)
    return selected, best_similarity, queries, selected


def hard_not_found() -> str:
    return "Non ho trovato la risposta nei documenti caricati."


# ============================================================
# CCNL LLM PROMPT
# ============================================================
RULES_CCNL = """
Sei l’assistente UILCOM per lavoratori IPZS.
Devi rispondere in modo chiaro e pratico basandoti SOLO sul contesto (estratti CCNL) fornito.
Non inventare informazioni.

REGOLE:
1) Se non trovi nel contesto, scrivi: "Non ho trovato la risposta nel CCNL caricato."
2) Chiudi SEMPRE con: "Fonte: CCNL (pag. ...)" usando solo le pagine presenti nel contesto.
3) Se l’utente chiede permessi ma nel contesto non ci sono tutti, non inventare.
"""

def split_public(text: str) -> str:
    # compatibile anche se il modello non usa tag
    m = re.search(r"<PUBLIC>(.*?)</PUBLIC>", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return text.strip()


# ============================================================
# CHAT STATE
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render chat history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if st.session_state.is_admin and m["role"] == "assistant":
            dbg = m.get("debug")
            if dbg:
                with st.expander("🧠 Admin: debug", expanded=False):
                    st.write("**Route:**", dbg.get("route"))
                    st.write("**Best similarity:**", dbg.get("best_similarity"))
                    st.write("**Query usate:**")
                    st.code("\n".join(dbg.get("queries", [])))
                    st.write("**Evidenze:**")
                    st.code("\n".join(dbg.get("evidence", [])) or "(nessuna)")
                    if dbg.get("pages_ccnl"):
                        st.write("**Pagine CCNL:**", dbg.get("pages_ccnl"))
                    st.write("**Chunk selezionati (prime righe):**")
                    for c in dbg.get("selected", [])[:4]:
                        st.write(f"**Pagina/Origine: {c.get('page')}**")
                        txt = (c.get("text") or "")
                        st.write(txt[:700] + ("..." if len(txt) > 700 else ""))
                        st.divider()

user_input = st.chat_input("Scrivi una domanda (CCNL / permessi IPZS: RAO, ROL, donazione sangue, ecc.)")

if not user_input:
    st.stop()

# Require indexes
need_ccnl = not (os.path.exists(CCNL_VEC_PATH) and os.path.exists(CCNL_META_PATH))
need_ipzs = not (os.path.exists(IPZS_VEC_PATH) and os.path.exists(IPZS_META_PATH))

if need_ccnl or need_ipzs:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Prima devo indicizzare i documenti: apri la barra laterale e clicca **Indicizza / Reindicizza (CCNL + IPZS)**.",
    })
    st.rerun()

st.session_state.messages.append({"role": "user", "content": user_input})

emb = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
route = detect_route(user_input)

# ============================================================
# IPZS: ELENCO COMPLETO (deterministico)
# ============================================================
if route == "IPZS_LIST":
    # Leggi file intero e costruisci elenco
    try:
        with open(IPZS_PERMESSI_PATH, "r", encoding="utf-8", errors="ignore") as f:
            ipzs_raw = f.read()
        sections = parse_ipzs_permessi_sections(ipzs_raw)

        if not sections:
            public_ans = hard_not_found()
        else:
            # elenco completo
            lines = []
            lines.append("Ecco l’**elenco dei permessi/giustificativi** presenti nel documento IPZS, con una breve spiegazione:")
            lines.append("")
            for s in sections:
                title = s["title"]
                desc = s["desc"]
                # prendi 1-3 righe utili
                desc_short = []
                for ln in desc.splitlines():
                    ln = ln.strip()
                    if not ln:
                        continue
                    # evita solo "Descrizione:"
                    if ln.lower() == "descrizione:":
                        continue
                    desc_short.append(ln)
                    if len(desc_short) >= 3:
                        break
                preview = " ".join(desc_short).strip()
                if not preview:
                    preview = "Vedi descrizione nel documento."
                lines.append(f"- **{title}**: {preview}")

            lines.append("")
            lines.append(ipzs_source_line())
            public_ans = "\n".join(lines).strip()

        assistant_payload = {"role": "assistant", "content": public_ans}

        if st.session_state.is_admin:
            # retrieval non serve qui: è elenco deterministico
            assistant_payload["debug"] = {
                "route": route,
                "best_similarity": None,
                "queries": [],
                "evidence": [],
                "pages_ccnl": [],
                "selected": [],
                "note": "Elenco IPZS generato deterministico dal file completo.",
            }

        st.session_state.messages.append(assistant_payload)
        st.rerun()
    except Exception:
        assistant_payload = {"role": "assistant", "content": hard_not_found()}
        st.session_state.messages.append(assistant_payload)
        st.rerun()


# ============================================================
# IPZS: PERMESSO SPECIFICO (RAO/ROL/altro)
# ============================================================
if route in ["IPZS_SINGLE", "MIXED"]:
    # prima prova IPZS
    selected_ipzs, best_ipzs, queries_ipzs, _ = retrieve_from_index(
        user_input, IPZS_VEC_PATH, IPZS_META_PATH, emb
    )
    retrieval_ok_ipzs = (best_ipzs >= MIN_BEST_SIMILARITY) and (len(selected_ipzs) >= MIN_SELECTED_CHUNKS)

    # carica tutte le sezioni per capire il "titolo" richiesto
    try:
        with open(IPZS_PERMESSI_PATH, "r", encoding="utf-8", errors="ignore") as f:
            ipzs_raw = f.read()
        sections = parse_ipzs_permessi_sections(ipzs_raw)
        titles = [s["title"] for s in sections]
        wanted = extract_specific_permesso_name(user_input, titles)
    except Exception:
        sections, titles, wanted = [], [], None

    if retrieval_ok_ipzs and sections:
        # Se l'utente ha chiesto un permesso specifico, rispondi SOLO quello
        if wanted:
            match = None
            # normalizza RAO/ROL
            if wanted == "RAO":
                for s in sections:
                    if s["title"].strip().lower() in ["rao", "r.a.o."]:
                        match = s
                        break
                if not match:
                    for s in sections:
                        if "rao" in s["title"].lower():
                            match = s
                            break
            elif wanted in ["R.O.L.", "ROL"]:
                for s in sections:
                    if "rol" in s["title"].lower() or "r.o.l" in s["title"].lower():
                        match = s
                        break
            else:
                for s in sections:
                    if s["title"].lower() == wanted.lower():
                        match = s
                        break
                if not match:
                    for s in sections:
                        if wanted.lower() in s["title"].lower():
                            match = s
                            break

            if match:
                public_ans = f"**{match['title']}**\n\n{match['desc'].strip()}\n\n{ipzs_source_line()}"
            else:
                # fallback: usa i chunk recuperati
                context = "\n\n---\n\n".join([c.get("text", "") for c in selected_ipzs[:6]])
                public_ans = f"Ho trovato riferimenti nel documento IPZS, ma non riesco a isolare la sezione esatta.\n\n{context}\n\n{ipzs_source_line()}"
        else:
            # Se NON è chiaro quale permesso, e la route è MIXED (generica),
            # allora non spariamo l'elenco: rispondi chiedendo parola chiave (ma senza bloccare troppo)
            if route == "MIXED" and ("permess" in user_input.lower()):
                public_ans = (
                    "Mi puoi dire **quale permesso** ti interessa (es. **RAO**, **ROL**, **donazione sangue**, **cure termali**, **104**, ecc.)?\n\n"
                    + ipzs_source_line()
                )
            else:
                public_ans = hard_not_found()

        assistant_payload = {"role": "assistant", "content": public_ans}

        if st.session_state.is_admin:
            assistant_payload["debug"] = {
                "route": route,
                "best_similarity": best_ipzs,
                "queries": queries_ipzs,
                "evidence": extract_evidence(selected_ipzs),
                "pages_ccnl": [],
                "selected": selected_ipzs[:6],
                "note": f"Permesso specifico richiesto: {wanted}",
            }

        st.session_state.messages.append(assistant_payload)
        st.rerun()

    # se IPZS debole e route MIXED, proviamo CCNL
    if route == "MIXED":
        # continua su CCNL sotto
        pass
    else:
        assistant_payload = {"role": "assistant", "content": hard_not_found()}
        st.session_state.messages.append(assistant_payload)
        st.rerun()


# ============================================================
# CCNL: Retrieval + LLM con citazioni
# ============================================================
selected_ccnl, best_ccnl, queries_ccnl, _ = retrieve_from_index(
    user_input, CCNL_VEC_PATH, CCNL_META_PATH, emb
)
retrieval_ok_ccnl = (best_ccnl >= MIN_BEST_SIMILARITY) and (len(selected_ccnl) >= MIN_SELECTED_CHUNKS)

if not retrieval_ok_ccnl:
    assistant_payload = {"role": "assistant", "content": "Non ho trovato la risposta nel CCNL caricato."}
    if st.session_state.is_admin:
        assistant_payload["debug"] = {
            "route": "CCNL",
            "best_similarity": best_ccnl,
            "queries": queries_ccnl,
            "evidence": extract_evidence(selected_ccnl),
            "pages_ccnl": unique_pages_ccnl(selected_ccnl),
            "selected": selected_ccnl[:6],
            "note": "retrieval_ok=False -> risposta bloccata",
        }
    st.session_state.messages.append(assistant_payload)
    st.rerun()

pages = unique_pages_ccnl(selected_ccnl, max_pages=8)
cit_line = format_ccnl_citations(pages)

context = "\n\n---\n\n".join([f"[Pagina {c.get('page','?')}] {c.get('text','')}" for c in selected_ccnl])

llm = ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE, api_key=OPENAI_API_KEY)

prompt = f"""
{RULES_CCNL}

DOMANDA (UTENTE):
{user_input}

CONTESTO CCNL (estratti):
{context}

SCRIVI UNA RISPOSTA CHIARA E PRATICA.
CHIUDI SEMPRE CON:
{cit_line}

FORMATO:
<PUBLIC>
...testo...
{cit_line}
</PUBLIC>
"""

try:
    raw = llm.invoke(prompt).content
except Exception as e:
    raw = f"<PUBLIC>Errore nel generare la risposta: {e}\n\n{cit_line}</PUBLIC>"

public_ans = split_public(raw).strip()

# Forza citazione se manca
if cit_line and ("fonte" not in public_ans.lower()):
    public_ans = public_ans.rstrip() + "\n\n" + cit_line

assistant_payload = {"role": "assistant", "content": public_ans}

if st.session_state.is_admin:
    assistant_payload["debug"] = {
        "route": "CCNL",
        "best_similarity": best_ccnl,
        "queries": queries_ccnl,
        "evidence": extract_evidence(selected_ccnl),
        "pages_ccnl": pages,
        "selected": selected_ccnl[:6],
        "bm25_available": BM25_AVAILABLE,
    }

st.session_state.messages.append(assistant_payload)
st.rerun()




