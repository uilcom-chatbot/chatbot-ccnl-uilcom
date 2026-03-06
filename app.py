# app.py — Assistente Contrattuale UILCOM IPZS
# ✅ Rev16 — 7 miglioramenti implementati:
#   1. Routing topic via LLM (classificatore, no keyword fragili)
#   2. Cross-encoder reranker (sentence-transformers, fallback BM25)
#   3. Guardrail estesi: ferie, TFR, maternità, preavviso, congedo matrimoniale
#   4. GPT-4o automatico per domande complesse
#   5. Soglie retrieval calibrate per topic
#   6. Memoria conversazionale migliorata (ultimi 2 turni completi domanda+risposta)
#   7. Risposta LLM strutturata in JSON (confidenza, fonte, avvertenza)

import os
import json
import re
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# Optional: rank-bm25
try:
    from rank_bm25 import BM25Okapi  # type: ignore
    BM25_AVAILABLE = True
except Exception:
    BM25_AVAILABLE = False

# Miglioramento 2: cross-encoder reranker
try:
    from sentence_transformers import CrossEncoder  # type: ignore
    _CE_MODEL = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    CE_AVAILABLE = True
except Exception:
    CE_AVAILABLE = False


# ============================================================
# CONFIG
# ============================================================
APP_TITLE = "🟦 Assistente Contrattuale UILCOM IPZS"

PDF_PATH      = os.path.join("documenti", "ccnl.pdf")
INDEX_DIR     = "index_ccnl"
VEC_PATH      = os.path.join(INDEX_DIR, "vectors.npy")
META_PATH     = os.path.join(INDEX_DIR, "chunks.json")

IPZS_TXT_PATH    = os.path.join("documenti", "PERMESSI_IPZS_COMPLETO_FINALE.txt")
INDEX_DIR_IPZS   = "index_ipzs_permessi"
VEC_PATH_IPZS    = os.path.join(INDEX_DIR_IPZS, "vectors.npy")
META_PATH_IPZS   = os.path.join(INDEX_DIR_IPZS, "chunks.json")

CHUNK_SIZE         = 1200
CHUNK_OVERLAP      = 150
IPZS_CHUNK_SIZE    = 1000
IPZS_CHUNK_OVERLAP = 120

TOP_K_PER_QUERY   = 10
TOP_K_FINAL       = 18
MAX_MULTI_QUERIES = 8
MAX_CHUNKS_PER_PAGE = 3
NEAR_DUP_JACCARD    = 0.92

# Miglioramento 6: memoria completa (domanda + risposta)
MEMORY_FULL_TURNS = 2

MAX_EVIDENCE_LINES = 18

# Miglioramento 4: modelli
LLM_MODEL_FAST   = "gpt-4o-mini"
LLM_MODEL_STRONG = "gpt-4o"
LLM_TEMPERATURE  = 0

# Miglioramento 5: soglie retrieval per topic
MIN_SIMILARITY_BY_TOPIC: Dict[str, float] = {
    "mansioni":             0.28,
    "straordinari":         0.28,
    "notturno_ordinario":   0.25,
    "permessi":             0.22,
    "rol_exfest":           0.22,
    "malattia":             0.24,
    "congedo_matrimoniale": 0.22,
    "ferie":                0.22,
    "tfr":                  0.22,
    "maternita":            0.22,
    "preavviso":            0.22,
    "altro":                0.20,
}
MIN_AVG_TOP3       = 0.210
MIN_SPREAD         = 0.018
MIN_SELECTED_CHUNKS = 2


# ============================================================
# SECRETS / PASSWORDS
# ============================================================
def get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    try:
        if key in st.secrets:  # type: ignore
            return str(st.secrets[key])  # type: ignore
    except Exception:
        pass
    return os.getenv(key, default)


UILCOM_PASSWORD = get_secret("UILCOM_PASSWORD")
ADMIN_PASSWORD  = get_secret("ADMIN_PASSWORD")
OPENAI_API_KEY  = get_secret("OPENAI_API_KEY")


# ============================================================
# PAGE SETUP
# ============================================================
st.set_page_config(page_title="Assistente UILCOM IPZS", page_icon="🟦", layout="centered")
st.title(APP_TITLE)
st.markdown(
    "**Accesso riservato agli iscritti UILCOM**  "
    "Strumento informativo per facilitare la consultazione del **CCNL Grafici Editoria**.  "
    "Le risposte sono basate solo sui documenti caricati e includono, quando disponibili, riferimenti a pagina/scheda.  "
    "Per casi complessi o contestazioni, contatta RSU/UILCOM."
)
st.divider()


# ============================================================
# AUTO-INDICIZZAZIONE (funzioni riutilizzate anche dalla sidebar admin)
# ============================================================
def _run_index_ccnl() -> str:
    """Indicizza il CCNL. Ritorna messaggio di esito."""
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"File non trovato: {PDF_PATH} — metti ccnl.pdf nella cartella /documenti")
    os.makedirs(INDEX_DIR, exist_ok=True)
    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)
    texts = [c.page_content for c in chunks]
    pages = [(int(c.metadata.get("page", 0)) + 1) for c in chunks]
    emb_idx = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    vectors = np.array(emb_idx.embed_documents(texts), dtype=np.float32)
    np.save(VEC_PATH, vectors)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump([{"page": p, "text": t} for p, t in zip(pages, texts)], f, ensure_ascii=False)
    return f"CCNL indicizzato: {len(chunks)} chunk"


# ============================================================
# IPZS TXT SPLIT — definita qui perché usata da _run_index_ipzs
# ============================================================
def split_ipzs_blocks(raw_txt: str) -> List[str]:
    txt = (raw_txt or "").replace("\r\n", "\n")
    lines = txt.split("\n")
    MIN_BLOCK = 30

    sep_idx = [i for i, ln in enumerate(lines) if re.fullmatch(r"[\-\=\*]{5,}", (ln.strip() or "")) is not None]
    if len(sep_idx) >= 1:
        blocks, start = [], 0
        for i in sep_idx:
            block = "\n".join(lines[start:i]).strip()
            if len(block) >= MIN_BLOCK:
                blocks.append(block)
            start = i + 1
        tail = "\n".join(lines[start:]).strip()
        if len(tail) >= MIN_BLOCK:
            blocks.append(tail)
        if blocks:
            return blocks

    starts = []
    for i, ln in enumerate(lines):
        s = (ln or "").strip()
        if not s:
            continue
        if 4 <= len(s) <= 90 and re.fullmatch(r"[A-Z0-9\u00C0-\u00DC\.\-\/\(\)\s]+", s) is not None:
            nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
            nxt2 = lines[i + 2].strip() if i + 2 < len(lines) else ""
            cond = (nxt == "") or (nxt and re.fullmatch(r"[A-Z0-9\u00C0-\u00DC\.\-\/\(\)\s]+", nxt) is None)
            cond = cond or (nxt == "" and nxt2 != "")
            if cond:
                starts.append(i)

    if len(starts) >= 2:
        blocks = []
        for k in range(len(starts)):
            a, b = starts[k], starts[k + 1] if k + 1 < len(starts) else len(lines)
            block = "\n".join(lines[a:b]).strip()
            if len(block) >= MIN_BLOCK:
                blocks.append(block)
        if blocks:
            return blocks

    return [txt.strip()]


def _run_index_ipzs() -> str:
    """Indicizza le schede IPZS. Ritorna messaggio di esito."""
    if not os.path.exists(IPZS_TXT_PATH):
        raise FileNotFoundError(f"File non trovato: {IPZS_TXT_PATH} — metti il TXT nella cartella /documenti")
    os.makedirs(INDEX_DIR_IPZS, exist_ok=True)
    with open(IPZS_TXT_PATH, "r", encoding="utf-8") as f:
        raw_txt = f.read()

    if not raw_txt.strip():
        raise ValueError("Il file IPZS è vuoto.")

    blocks = split_ipzs_blocks(raw_txt)

    # ✅ Fix: se split non trova blocchi validi (file piccolo o formato diverso),
    # usa il testo intero come unico blocco anziché fallire silenziosamente.
    if not blocks:
        blocks = [raw_txt.strip()]

    splitter = RecursiveCharacterTextSplitter(chunk_size=IPZS_CHUNK_SIZE, chunk_overlap=IPZS_CHUNK_OVERLAP)
    chunks_ipzs: List[Dict[str, Any]] = []
    for scheda, b in enumerate(blocks, 1):
        parts = splitter.split_text(b)
        # ✅ Fix: se il blocco è più corto del chunk_size, includilo comunque
        if not parts and b.strip():
            parts = [b.strip()]
        for p in parts:
            if p.strip():
                chunks_ipzs.append({"page": scheda, "text": p})

    if not chunks_ipzs:
        raise ValueError(f"Nessun chunk estratto dal file IPZS. Blocchi trovati: {len(blocks)}. Controlla il formato del file.")

    texts_ipzs = [c["text"] for c in chunks_ipzs]
    emb_idx = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    vectors_ipzs = np.array(emb_idx.embed_documents(texts_ipzs), dtype=np.float32)
    np.save(VEC_PATH_IPZS, vectors_ipzs)
    with open(META_PATH_IPZS, "w", encoding="utf-8") as f:
        json.dump(chunks_ipzs, f, ensure_ascii=False)
    return f"IPZS indicizzato: {len(blocks)} schede, {len(chunks_ipzs)} chunk"


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

if not st.session_state.auth_ok:
    st.stop()


# ============================================================
# AUTO-INDICIZZAZIONE AL PRIMO ACCESSO
# Se i documenti esistono ma l'indice no, indicizza automaticamente.
# L'utente vede una progress bar e non deve fare nulla.
# ============================================================
if "auto_index_ccnl_done" not in st.session_state:
    st.session_state.auto_index_ccnl_done = False
if "auto_index_ipzs_done" not in st.session_state:
    st.session_state.auto_index_ipzs_done = False

_needs_ccnl  = not (os.path.exists(VEC_PATH) and os.path.exists(META_PATH))
_needs_ipzs  = not (os.path.exists(VEC_PATH_IPZS) and os.path.exists(META_PATH_IPZS))
_has_ccnl_src = os.path.exists(PDF_PATH)
_has_ipzs_src = os.path.exists(IPZS_TXT_PATH)

# Indicizza CCNL se necessario e non ancora fatto in questa sessione
if _needs_ccnl and _has_ccnl_src and not st.session_state.auto_index_ccnl_done:
    st.info("⏳ Indicizzazione CCNL in corso. Attendi...")
    progress_ccnl = st.progress(10, text="Indicizzazione CCNL...")
    try:
        msg = _run_index_ccnl()
        progress_ccnl.progress(100, text=f"✅ {msg}")
        st.session_state.auto_index_ccnl_done = True
        st.rerun()
    except Exception as e:
        st.warning(f"⚠️ Indicizzazione CCNL fallita: {e}")
        st.session_state.auto_index_ccnl_done = True  # non ritentare in loop

# Indicizza IPZS se necessario e non ancora fatto in questa sessione
if _needs_ipzs and _has_ipzs_src and not st.session_state.auto_index_ipzs_done:
    st.info("⏳ Indicizzazione IPZS (permessi) in corso. Attendi...")
    progress_ipzs = st.progress(10, text="Indicizzazione IPZS...")
    try:
        msg = _run_index_ipzs()
        progress_ipzs.progress(100, text=f"✅ {msg}")
        st.session_state.auto_index_ipzs_done = True
        st.rerun()
    except Exception as e:
        st.warning(f"⚠️ Indicizzazione IPZS fallita: {e}")
        st.session_state.auto_index_ipzs_done = True  # non ritentare in loop


# ============================================================
# ADMIN MODE
# ============================================================
if "is_admin" not in st.session_state:
    st.session_state.is_admin = False


# ============================================================
# HARD FAIL IF NO OPENAI KEY
# ============================================================
if not OPENAI_API_KEY:
    st.error(
        "Manca la variabile **OPENAI_API_KEY**.\n\n"
        "Streamlit Cloud: **Settings → Secrets → OPENAI_API_KEY**\n"
        "Locale: variabile d'ambiente OPENAI_API_KEY"
    )
    st.stop()


# ============================================================
# TEXT NORMALIZATION
# ============================================================
def normalize_text_for_match(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    s = re.sub(r"\s+", " ", s)
    return s.lower()


# ============================================================
# CCNL: ESCLUSIONE PARTE SESTA
# ============================================================
def is_parte_sesta_chunk(text: str) -> bool:
    t = normalize_text_for_match(text or "")
    if "parte sesta" in t:
        return True
    if "giornali" in t and ("quotidiani" in t or "periodici" in t):
        return True
    if "aziende" in t and "editrici" in t and "stampatrici" in t:
        return True
    return False


# ============================================================
# RETRIEVAL HELPERS
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
    fixed: List[Dict[str, Any]] = []
    for item in meta:
        if isinstance(item, dict) and "text" in item and "page" in item:
            fixed.append({"page": item.get("page", "?"), "text": item.get("text", "")})
        elif isinstance(item, str):
            fixed.append({"page": "?", "text": item})
        else:
            fixed.append({"page": "?", "text": str(item)})
    return vectors, fixed


def unique_pages(chunks: List[Dict[str, Any]], max_pages: int = 8) -> List[int]:
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


def format_public_citations(source: str, pages: List[int]) -> str:
    if not pages:
        return ""
    pages_sorted = sorted(pages)
    if source == "IPZS":
        if len(pages_sorted) == 1:
            return f"**Fonte:** IPZS Permessi (scheda {pages_sorted[0]})"
        return f"**Fonte:** IPZS Permessi (schede {', '.join(map(str, pages_sorted))})"
    if len(pages_sorted) == 1:
        return f"**Fonte:** CCNL (pag. {pages_sorted[0]})"
    return f"**Fonte:** CCNL (pagg. {', '.join(map(str, pages_sorted))})"


def _tokenize_light(s: str) -> List[str]:
    return re.findall(r"[a-z\u00e0\u00e8\u00e9\u00ec\u00f2\u00f90-9]+", (s or "").lower())


def _jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))


def _dedup_near_identical(chunks: List[Dict[str, Any]], thr: float = 0.92) -> List[Dict[str, Any]]:
    kept: List[Dict[str, Any]] = []
    kept_tok: List[List[str]] = []
    for c in chunks:
        tok = _tokenize_light(c.get("text", ""))
        if any(_jaccard(tok, kt) >= thr for kt in kept_tok):
            continue
        kept.append(c)
        kept_tok.append(tok)
    return kept


def _limit_per_page(chunks: List[Dict[str, Any]], max_per_page: int = 3) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    cnt: Dict[Any, int] = {}
    for c in chunks:
        p = c.get("page", "?")
        cnt[p] = cnt.get(p, 0) + 1
        if cnt[p] <= max_per_page:
            out.append(c)
    return out


def compute_retrieval_confidence(sim_values: List[float]) -> Dict[str, float]:
    if not sim_values:
        return {"best": 0.0, "avg_top3": 0.0, "median": 0.0, "spread": 0.0}
    best = float(sim_values[0])
    top3 = sim_values[:3]
    avg_top3 = float(sum(top3) / max(1, len(top3)))
    med = float(np.median(np.array(sim_values, dtype=np.float32)))
    spread = float(best - med)
    return {"best": best, "avg_top3": avg_top3, "median": med, "spread": spread}


# ============================================================
# MIGLIORAMENTO 2: CROSS-ENCODER RERANKER
# ============================================================
def crossencoder_rerank(query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not candidates:
        return candidates
    if CE_AVAILABLE:
        try:
            pairs = [(query, (c.get("text") or "")[:512]) for c in candidates]
            scores = _CE_MODEL.predict(pairs)
            idx = np.argsort(-np.array(scores))
            return [candidates[int(i)] for i in idx]
        except Exception:
            pass
    if BM25_AVAILABLE:
        try:
            corpus = [(c.get("text") or "").lower().split() for c in candidates]
            bm25 = BM25Okapi(corpus)
            scores = bm25.get_scores(query.lower().split())
            idx = np.argsort(-np.array(scores))
            return [candidates[int(i)] for i in idx]
        except Exception:
            pass
    return candidates


# ============================================================
# MIGLIORAMENTO 1: TOPIC CLASSIFIER VIA LLM
# ============================================================
TOPIC_CLASSIFIER_PROMPT = """\
Sei un classificatore di domande per un assistente contrattuale UILCOM IPZS (CCNL Grafici Editoria).
Classifica la domanda in UNO dei seguenti topic:

mansioni           - mansioni superiori, categoria superiore, passaggio livello, inquadramento
straordinari       - lavoro straordinario, maggiorazioni straordinario feriale/notturno/festivo
notturno_ordinario - lavoro notturno ordinario (NON straordinario notturno)
permessi           - permessi retribuiti IPZS, lutto, visite mediche, studio, 104, assemblea, donazione sangue
rol_exfest         - ROL, RAO, ex festività, festività soppresse/abolite, riposi annui
malattia           - malattia, comporto, certificato, visita fiscale, reperibilità, INPS
congedo_matrimoniale - congedo matrimoniale, permesso matrimonio, nozze
ferie              - ferie, ferie annuali, periodo feriale, godimento ferie
tfr                - TFR, trattamento fine rapporto, liquidazione, anticipazione TFR
maternita          - maternità, paternità, congedo parentale, astensione obbligatoria/facoltativa
preavviso          - preavviso, licenziamento, dimissioni, periodo di preavviso
altro              - tutto il resto

Rispondi SOLO con il nome esatto del topic (una parola dalla lista), nient'altro.
Domanda: {q}
"""


@st.cache_data(show_spinner=False, ttl=3600)
def classify_topic_llm(q: str, api_key: str) -> str:
    try:
        llm = ChatOpenAI(model=LLM_MODEL_FAST, temperature=0, api_key=api_key)
        result = llm.invoke(TOPIC_CLASSIFIER_PROMPT.format(q=q.strip())).content.strip().lower()
        valid = {
            "mansioni", "straordinari", "notturno_ordinario", "permessi", "rol_exfest",
            "malattia", "congedo_matrimoniale", "ferie", "tfr", "maternita", "preavviso", "altro"
        }
        for token in re.split(r"[\s\n,\.]+", result):
            if token in valid:
                return token
        return "altro"
    except Exception:
        return "altro"


# ============================================================
# SORGENTE PER TOPIC
# ============================================================
IPZS_TOPICS = {"permessi", "rol_exfest"}


def topic_to_source(topic: str) -> str:
    return "IPZS" if topic in IPZS_TOPICS else "CCNL"


# ============================================================
# MIGLIORAMENTO 4: DOMANDE COMPLESSE → GPT-4o
# ============================================================
def is_complex_question(q: str) -> bool:
    q = q.strip()
    if len(q.split()) > 20:
        return True
    if q.count("?") > 1:
        return True
    keywords = ["e inoltre", "ma anche", "in più", "oltre a", "combinat",
                "differenza tra", "confronto", "rispetto a", "sia ... che"]
    if any(x in q.lower() for x in keywords):
        return True
    return False


def choose_model(q: str) -> str:
    return LLM_MODEL_STRONG if is_complex_question(q) else LLM_MODEL_FAST


# ============================================================
# QUERY BUILDER
# ============================================================
def build_queries(q: str, topic: str) -> List[str]:
    q0 = q.strip()
    qs = [q0, f"{q0} CCNL regola"]

    topic_queries: Dict[str, List[str]] = {
        "rol_exfest": [
            "RAO festività infrasettimanali abolite riposi retribuiti",
            "ROL riduzione orario di lavoro maturazione fruizione monte ore",
        ],
        "permessi": [
            "permessi retribuiti tipologie elenco",
            "permesso studio una settimana l'anno",
            "donazione sangue permesso retribuito",
            "legge 104 art 33 comma 3 permesso",
        ],
        "malattia": [
            "malattia trattamento economico integrazione",
            "malattia periodo di comporto conservazione posto",
            "malattia visite fiscali reperibilità fasce",
        ],
        "straordinari": [
            "lavoro straordinario maggiorazioni percentuale",
            "straordinario diurno maggiorazione 35",
            "straordinario notturno maggiorazione 60",
            "straordinario festivo maggiorazione 60",
        ],
        "notturno_ordinario": [
            "lavoro notturno ordinario maggiorazione percentuale",
            "lavoro notturno turni maggiorazione CCNL grafici",
            "indennità maggiorazione lavoro notturno non straordinario",
        ],
        "mansioni": [
            "mansioni superiori 30 giorni consecutivi 60 giorni non consecutivi",
            "mansioni superiori trattamento corrispondente attività svolta",
            "sostituzione lavoratore assente conservazione del posto",
        ],
        "congedo_matrimoniale": [
            "congedo matrimoniale giorni retribuiti CCNL",
            "matrimonio permesso retribuito giorni lavorativi",
            "congedo matrimoniale grafici editoria",
        ],
        "ferie": [
            "ferie annuali giorni spettanti maturazione CCNL",
            "ferie impiegati giorni lavorativi anzianità",
            "ferie operai giorni lavorativi anzianità",
            "ferie godimento periodo feriale operai impiegati",
            "ferie non godute monetizzazione cessazione",
        ],
        "tfr": [
            "trattamento fine rapporto aliquota calcolo rivalutazione",
            "TFR anticipazione condizioni acquisto casa",
            "liquidazione TFR fondo pensione complementare",
        ],
        "maternita": [
            "maternità astensione obbligatoria mesi retribuzione integrazione",
            "congedo parentale facoltativo mesi indennità",
            "paternità obbligatoria giorni retribuiti",
        ],
        "preavviso": [
            "preavviso licenziamento dimissioni durata livello anzianità",
            "periodo preavviso categoria contratto grafici",
            "indennità sostitutiva preavviso",
        ],
    }

    qs += topic_queries.get(topic, [])

    out, seen = [], set()
    for x in qs:
        x = x.strip()
        if x and x not in seen:
            out.append(x)
            seen.add(x)
    return out[:MAX_MULTI_QUERIES]


# ============================================================
# EVIDENCE EXTRACTION (debug)
# ============================================================
def extract_key_evidence(chunks: List[Dict[str, Any]], source: str) -> List[str]:
    patterns = [
        r"\b30\b", r"\b60\b", r"mansioni?\s+superiori?", r"trattamento\s+corrispondente",
        r"straordin", r"maggiorazion", r"\b35%\b", r"\b60%\b", r"\b40%\b", r"\b80%\b",
        r"parte\s+sesta", r"quotidiani", r"periodici", r"\brao\b", r"\brol\b",
        r"ferie", r"\btfr\b", r"preavviso", r"maternit", r"congedo\s+matrimoniale",
    ]
    evidences: List[str] = []
    for c in chunks:
        page = c.get("page", "?")
        text = c.get("text", "") or ""
        for ln in [x.strip() for x in text.splitlines() if x.strip()]:
            ln_low = normalize_text_for_match(ln)
            if any(re.search(p, ln_low) for p in patterns):
                ln_clean = " ".join(ln.split())
                if 20 <= len(ln_clean) <= 420:
                    tag = "scheda" if source == "IPZS" else "pag."
                    evidences.append(f"({tag} {page}) {ln_clean}")
    out, seen = [], set()
    for e in evidences:
        if e not in seen:
            out.append(e)
            seen.add(e)
    return out[:MAX_EVIDENCE_LINES]


# ============================================================
# GUARDRAIL: MANSIONI SUPERIORI
# ============================================================
def extract_mansioni_rules(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    txt_all = " ".join([normalize_text_for_match(c.get("text") or "") for c in chunks])
    found_30 = bool(re.search(r"\b30\b", txt_all) and re.search(r"(giorn|gg)", txt_all))
    found_60 = bool(re.search(r"\b60\b", txt_all) and re.search(r"(giorn|gg)", txt_all))
    has_trattamento = bool(re.search(r"ha\s+diritto\s+al\s+trattamento\s+corrispondente|trattamento\s+corrispondente", txt_all))
    has_esclusione = bool(re.search(r"non\s+si\s+applica.*sostituzion|sostituzion.*conservazion|diritto.*conservazion.*posto", txt_all))
    has_formazione = bool(re.search(r"formazion|addestramento|affiancamento", txt_all) and re.search(r"non\s+costituisc", txt_all))
    pages = set()
    for c in chunks:
        try:
            pages.add(int(c.get("page", 0)))
        except Exception:
            pass
    return {
        "found_30": found_30, "found_60": found_60,
        "has_trattamento": has_trattamento, "has_esclusione": has_esclusione,
        "has_formazione": has_formazione, "pages": sorted([p for p in pages if p]),
    }


def mansioni_public_answer(user_q: str, rules: Dict[str, Any]) -> str:
    ql = user_q.lower()
    ask_stabilizzazione = any(x in ql for x in [
        "categoria", "livello", "passaggio", "inquadramento", "definitiv",
        "stabilizz", "30", "60", "giorni", "matur", "diviene definitiva"
    ])
    ask_trattamento = any(x in ql for x in ["differenza", "paga", "pagato", "trattamento", "retribuzione"])
    has_30_60 = bool(rules.get("found_30") and rules.get("found_60"))
    parts: List[str] = []

    if ask_trattamento:
        parts.append(
            "Se vieni assegnato a **mansioni superiori**, hai diritto al **trattamento corrispondente all'attività svolta** "
            "(cioè alla **differenza retributiva** per i giorni/periodi in cui svolgi quelle mansioni)."
        )

    if ask_stabilizzazione or not parts:
        note = "" if has_30_60 else "\n\n*(Regola da CCNL — verifica la pagina esatta nella tua copia del contratto.)*"
        parts.append(
            "La categoria/livello superiore **matura** (diventa definitivo) dopo:\n"
            "- **30 giorni consecutivi** di mansioni superiori;\n"
            "- **60 giorni complessivi**, anche non continuativi." + note
        )
        if rules.get("has_esclusione"):
            parts.append(
                "⚠️ **Eccezione:** la maturazione **non opera** se l'assegnazione avviene per **sostituzione** "
                "di un dipendente assente con **diritto alla conservazione del posto**."
            )

    if rules.get("has_formazione"):
        parts.append(
            "ℹ️ La **formazione/addestramento in affiancamento** con personale di categoria superiore "
            "non è considerata assegnazione a mansioni superiori."
        )

    cit = format_public_citations("CCNL", rules.get("pages", []) or [])
    if cit:
        parts.append(cit)
    return "\n\n".join(parts).strip()


# ============================================================
# GUARDRAIL: STRAORDINARI
# ============================================================
def straordinari_public_answer(user_q: str, cc_pages: List[int]) -> str:
    ql = user_q.lower()
    lines = ["Per lo **straordinario** (CCNL Grafici-Editoriali), le maggiorazioni sono:"]
    if "ferial" in ql and "festiv" not in ql and "notturn" not in ql:
        lines.append("- **Straordinario feriale (diurno):** **35%**")
    else:
        lines.append("- **Straordinario diurno:** **35%**")
        lines.append("- **Straordinario notturno:** **60%**")
        lines.append("- **Straordinario festivo:** **60%**")
    lines.append(
        "ℹ️ Le maggiorazioni del **40%** e **80%** riguardano esclusivamente la "
        "**Parte Sesta** (aziende editrici e stampatrici di giornali quotidiani e periodici)."
    )
    cit = format_public_citations("CCNL", cc_pages or [])
    if cit:
        lines.append(cit)
    return "\n".join(lines).strip()


# ============================================================
# MIGLIORAMENTO 3: GUARDRAIL ESTESI
# ============================================================
def ferie_public_answer(cc_pages: List[int]) -> str:
    lines = [
        "Riguardo alle **ferie** (CCNL Grafici-Editoriali):",
        "- **Grafici:** **27 giorni lavorativi** di ferie annuali retribuite.",
        "- **Cartotecnici:** **27 giorni lavorativi** fino a 15 anni di servizio; **30 giorni lavorativi** dopo i 15 anni di servizio.",
        "- Le ferie devono essere godute preferibilmente nell'anno di maturazione; la parte eccedente le 2 settimane può essere rinviata entro 18 mesi.",
        "- Le ferie **non possono essere sostituite da indennità** durante il rapporto di lavoro (salvo alla cessazione).",
        "- In caso di malattia insorta durante le ferie, il periodo feriale si sospende se ricoverato in ospedale (verifica le condizioni esatte nel CCNL).",
        "ℹ️ Per casi specifici (part-time, primo anno, ecc.) contatta RSU/UILCOM.",
    ]
    cit = format_public_citations("CCNL", cc_pages or [])
    if cit:
        lines.append(cit)
    return "\n".join(lines).strip()


def tfr_public_answer(cc_pages: List[int]) -> str:
    lines = [
        "Riguardo al **TFR (Trattamento di Fine Rapporto)**:",
        "- Si calcola dividendo la retribuzione annua utile per **13,5** e rivalutando ogni anno al **75% dell'inflazione ISTAT + 1,5% fisso**.",
        "- Puoi richiedere un'**anticipazione** (fino al 70%) dopo almeno **8 anni di servizio**, per spese sanitarie o acquisto prima casa.",
        "- Il TFR può essere destinato a un **fondo pensione complementare** (scelta entro 6 mesi dall'assunzione o con destinazione tacita).",
        "ℹ️ Per l'importo preciso e le condizioni IPZS, contatta RSU/UILCOM.",
    ]
    cit = format_public_citations("CCNL", cc_pages or [])
    if cit:
        lines.append(cit)
    return "\n".join(lines).strip()


def maternita_public_answer(cc_pages: List[int]) -> str:
    lines = [
        "Riguardo a **maternità e congedo parentale** (D.Lgs. 151/2001 + CCNL):",
        "- **Astensione obbligatoria (maternità):** 5 mesi (2 ante-parto + 3 post-parto), con **indennità INPS all'80%** + eventuale integrazione CCNL al 100%.",
        "- **Congedo parentale:** fino a **6 mesi** per genitore (10 mesi totali coppia) entro i 12 anni del figlio, indennizzato al 30% (o 80% per il primo mese).",
        "- **Paternità obbligatoria:** **10 giorni** di congedo obbligatorio retribuito al 100%.",
        "ℹ️ Verifica eventuali integrazioni previste dal CCNL o da accordi aziendali IPZS.",
    ]
    cit = format_public_citations("CCNL", cc_pages or [])
    if cit:
        lines.append(cit)
    return "\n".join(lines).strip()


def preavviso_public_answer(cc_pages: List[int]) -> str:
    lines = [
        "Riguardo al **preavviso** (CCNL Grafici-Editoriali):",
        "- La durata varia in base alla **categoria/livello** e all'**anzianità di servizio**.",
        "- In mancanza di preavviso, chi recede paga un'**indennità sostitutiva** pari alle retribuzioni del periodo.",
        "- Il **licenziamento per giusta causa** non richiede preavviso.",
        "ℹ️ Per la durata esatta in base al tuo livello, consulta il CCNL o RSU/UILCOM.",
    ]
    cit = format_public_citations("CCNL", cc_pages or [])
    if cit:
        lines.append(cit)
    return "\n".join(lines).strip()


def congedo_matrimoniale_public_answer(cc_pages: List[int]) -> str:
    lines = [
        "Riguardo al **congedo matrimoniale** (CCNL Grafici-Editoriali):",
        "- Il lavoratore ha diritto a un periodo di **congedo retribuito** per matrimonio.",
        "- In genere il CCNL Grafici Editoria prevede **15 giorni** di congedo matrimoniale retribuito.",
        "ℹ️ Verifica la durata esatta nel testo del CCNL o contatta RSU/UILCOM per conferma.",
    ]
    cit = format_public_citations("CCNL", cc_pages or [])
    if cit:
        lines.append(cit)
    return "\n".join(lines).strip()


# ============================================================
# FULLSCAN NOTTURNO ORDINARIO
# ============================================================
def fullscan_notturno_ordinario(meta_all: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for c in meta_all:
        txt = c.get("text", "") or ""
        t = normalize_text_for_match(txt)
        if re.search(r"not\s*-?\s*turn", t) and "%" in txt and "straordin" not in t:
            if is_parte_sesta_chunk(txt):
                continue
            out.append(c)
    return _dedup_near_identical(out)[:40]


# ============================================================
# SYSTEM RULES LLM — risposta JSON strutturata (Miglioramento 7)
# ============================================================
RULES_PUBLIC = """
Sei l'assistente UILCOM per lavoratori IPZS (CCNL Grafici Editoria).
Rispondi in italiano naturale, chiaro e pratico (tono sindacale: "cosa spetta / cosa verificare / cosa fare"),
basandoti SOLO sul contesto fornito. Non inventare informazioni.

REGOLE:
1) Se non trovi risposta nel contesto: risposta = "Non ho trovato la risposta nei documenti caricati."
2) NON confondere lavoro notturno ordinario con straordinario notturno.
3) EX FESTIVITÀ = festività soppresse/abolite/infrasettimanali abolite (sono la stessa cosa).
4) Permessi: elenca SOLO le tipologie presenti nel contesto.
5) confidenza: "alta" = risposta chiara nel contesto | "media" = parziale | "bassa" = inferita/non trovata.
6) avvertenza: aggiungi solo se utile (es. contattare RSU per casi complessi).
7) STRAORDINARI: se vedi 40%/80% o "Parte Sesta/quotidiani/periodici", segnalalo come settore diverso.
8) CATEGORIE: se nel contesto ci sono regole diverse per OPERAI e IMPIEGATI (o altre categorie),
   riportale ENTRAMBE in modo distinto — non rispondere solo per una categoria.

FORMATO — rispondi ESCLUSIVAMENTE con questo JSON valido (nessun testo fuori):
{
  "risposta": "testo risposta in markdown",
  "fonte": "es. CCNL pag. 45 oppure IPZS Permessi scheda 3",
  "confidenza": "alta|media|bassa",
  "avvertenza": "testo avvertenza oppure stringa vuota"
}
"""


# ============================================================
# IPZS: ESTRAZIONE TITOLI PERMESSI
# ============================================================
def extract_ipzs_permessi_titles_from_file(path: str) -> List[str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            txt = f.read().replace("\r\n", "\n")
    except Exception:
        return []
    lines = [ln.rstrip() for ln in txt.split("\n")]
    titles: List[str] = []
    for i in range(len(lines) - 1):
        title = (lines[i] or "").strip()
        underline = (lines[i + 1] or "").strip()
        if not title:
            continue
        if re.fullmatch(r"-{3,}", underline):
            if "IPZS" in title and "PERMESSI" in title:
                continue
            if title.upper() != title:
                continue
            titles.append(title)
    out, seen = [], set()
    for t in titles:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out


def is_lista_permessi_question(q: str) -> bool:
    ql = (q or "").lower()
    triggers = [
        "tutti i permessi", "elenco permessi", "lista permessi", "quali permessi",
        "quali sono i permessi", "permessi disponibili", "tutte le tipologie di permessi",
        "tutte le assenze", "giustificativi disponibili",
    ]
    return any(t in ql for t in triggers)


# ============================================================
# IPZS TXT SPLIT
# ============================================================
# ============================================================
# MIGLIORAMENTO 6: MEMORIA CONVERSAZIONALE MIGLIORATA
# ============================================================
def build_enriched_question(current_q: str, current_topic: str) -> str:
    """
    Includi gli ultimi MEMORY_FULL_TURNS turni completi (domanda + risposta).
    Azzera la memoria solo se il topic cambia (evita contaminazioni).
    """
    if "messages" not in st.session_state:
        return current_q.strip()

    last_topic = st.session_state.get("last_topic", None)
    if last_topic and last_topic != current_topic:
        return current_q.strip()

    messages = st.session_state.messages
    pairs: List[str] = []
    i = len(messages) - 1
    turns = 0
    while i >= 0 and turns < MEMORY_FULL_TURNS:
        msg = messages[i]
        if msg.get("role") == "assistant":
            assistant_content = (msg.get("content") or "").strip()
            if i > 0 and messages[i - 1].get("role") == "user":
                user_content = (messages[i - 1].get("content") or "").strip()
                if user_content != current_q.strip():
                    pairs.insert(0, f"Utente: {user_content}\nAssistente: {assistant_content[:400]}")
                    turns += 1
                i -= 2
            else:
                i -= 1
        else:
            i -= 1

    if not pairs:
        return current_q.strip()

    return (
        "CONTESTO CONVERSAZIONE (ultimi scambi, stesso argomento):\n"
        + "\n---\n".join(pairs)
        + "\n\nDOMANDA ATTUALE:\n"
        + current_q.strip()
    )


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.header("⚙️ Controlli")

    st.subheader("🧠 Admin (debug)")
    if ADMIN_PASSWORD:
        admin_in = st.text_input("Password admin", type="password", placeholder="Solo admin UILCOM", key="admin_pwd")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Login admin", use_container_width=True):
                st.session_state.is_admin = (admin_in == ADMIN_PASSWORD)
                st.success("Admin attivo.") if st.session_state.is_admin else st.error("Password admin errata.")
        with c2:
            if st.button("Logout", use_container_width=True):
                st.session_state.is_admin = False
    else:
        st.caption("ADMIN_PASSWORD non impostata.")

    st.divider()
    st.subheader("📦 Stato indici")
    ok_index = os.path.exists(VEC_PATH) and os.path.exists(META_PATH)
    ok_ipzs  = os.path.exists(VEC_PATH_IPZS) and os.path.exists(META_PATH_IPZS)
    st.write("Indice CCNL:", "✅ pronto" if ok_index else "❌ mancante")
    st.write("Indice IPZS:", "✅ pronto" if ok_ipzs  else "❌ mancante")

    # Pulsanti di reindicizzazione manuale: SOLO ADMIN
    if st.session_state.is_admin:
        st.divider()
        st.caption("🔧 Strumenti admin — reindicizzazione manuale")
        if st.button("🔄 Reindicizza CCNL", use_container_width=True):
            try:
                with st.spinner("Reindicizzazione CCNL in corso..."):
                    msg = _run_index_ccnl()
                st.success(msg)
                st.rerun()
            except Exception as e:
                st.error(str(e))

        if st.button("🔄 Reindicizza IPZS", use_container_width=True):
            try:
                with st.spinner("Reindicizzazione IPZS in corso..."):
                    msg = _run_index_ipzs()
                st.success(msg)
                st.rerun()
            except Exception as e:
                st.error(str(e))

        if st.button("🗑️ Reset auto-index (forza rindexing al prossimo login)", use_container_width=True):
            st.session_state.auto_index_ccnl_done = False
            st.session_state.auto_index_ipzs_done = False
            st.success("Reset eseguito. Al prossimo accesso verrà reindicizzato automaticamente.")

    st.divider()
    if st.button("🧹 Nuova chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_topic = None
        st.rerun()

    st.divider()
    st.caption(f"Veloce: `{LLM_MODEL_FAST}` | Potente: `{LLM_MODEL_STRONG}`")
    st.caption(f"Cross-encoder: {'✅' if CE_AVAILABLE else '❌ (BM25 fallback)'}")


# ============================================================
# CHAT STATE + HISTORY
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_topic" not in st.session_state:
    st.session_state.last_topic = None

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if st.session_state.is_admin and m["role"] == "assistant" and m.get("debug"):
            with st.expander("🧠 Admin debug", expanded=False):
                st.json(m["debug"])


# ============================================================
# CHAT INPUT
# ============================================================
user_input = st.chat_input("Scrivi una domanda (permessi, ROL/RAO, malattia, straordinari, mansioni, ferie, TFR...)")
if not user_input:
    st.stop()

st.session_state.messages.append({"role": "user", "content": user_input})
with st.chat_message("user"):
    st.markdown(user_input)


# ============================================================
# MIGLIORAMENTO 1: CLASSIFICAZIONE TOPIC VIA LLM
# ============================================================
with st.spinner("Analizzo la domanda..."):
    topic = classify_topic_llm(user_input, OPENAI_API_KEY)

# Override sicurezza
if "straordin" in (user_input or "").lower():
    topic = "straordinari"

enriched_q = build_enriched_question(user_input, topic)
source = topic_to_source(topic)


# ============================================================
# CHECK INDICE DISPONIBILE
# ============================================================
if source == "CCNL":
    if not (os.path.exists(VEC_PATH) and os.path.exists(META_PATH)):
        st.session_state.messages.append({"role": "assistant", "content": "⏳ L'indice CCNL non è ancora pronto. Ricarica la pagina — verrà indicizzato automaticamente se il file è presente."})
        st.rerun()
else:
    if not (os.path.exists(VEC_PATH_IPZS) and os.path.exists(META_PATH_IPZS)):
        st.session_state.messages.append({"role": "assistant", "content": "⏳ L'indice IPZS non è ancora pronto. Ricarica la pagina — verrà indicizzato automaticamente se il file è presente."})
        st.rerun()


# ============================================================
# RETRIEVAL
# ============================================================
if source == "CCNL":
    vectors, meta = load_index(VEC_PATH, META_PATH)
else:
    vectors, meta = load_index(VEC_PATH_IPZS, META_PATH_IPZS)

mat_norm = normalize_rows(vectors)
emb = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
queries = build_queries(enriched_q, topic)

scores_best: Dict[int, float] = {}
for q in queries:
    qvec = np.array(emb.embed_query(q), dtype=np.float32)
    sims = cosine_scores(qvec, mat_norm)
    for i in np.argsort(-sims)[:TOP_K_PER_QUERY]:
        idx = int(i)
        if source == "CCNL" and is_parte_sesta_chunk(meta[idx].get("text", "")):
            continue
        s = float(sims[idx])
        if idx not in scores_best or s > scores_best[idx]:
            scores_best[idx] = s

ranked_idx = sorted(scores_best.keys(), key=lambda i: scores_best[i], reverse=True)
candidates: List[Dict[str, Any]] = []
cand_scores: List[float] = []
for i in ranked_idx[:max(TOP_K_FINAL * 3, TOP_K_FINAL)]:
    candidates.append(meta[int(i)])
    cand_scores.append(float(scores_best[int(i)]))

tmp: List[Dict[str, Any]] = []
tmp_scores: List[float] = []
per_page: Dict[Any, int] = {}
for c, s in zip(candidates, cand_scores):
    p = c.get("page", "?")
    per_page[p] = per_page.get(p, 0) + 1
    if per_page[p] <= MAX_CHUNKS_PER_PAGE:
        tmp.append(c)
        tmp_scores.append(s)
    if len(tmp) >= TOP_K_FINAL:
        break

selected = _dedup_near_identical(tmp[:TOP_K_FINAL])
selected_scores = tmp_scores[:len(selected)]

# MIGLIORAMENTO 2: cross-encoder rerank + riallinea scores
_pre_rerank = list(selected)
selected = crossencoder_rerank(enriched_q, selected)
_score_map = {id(c): s for c, s in zip(_pre_rerank, selected_scores)}
selected_scores = [_score_map.get(id(c), 0.0) for c in selected]

# MIGLIORAMENTO 5: soglie per topic
min_best = MIN_SIMILARITY_BY_TOPIC.get(topic, MIN_SIMILARITY_BY_TOPIC["altro"])
confidence = compute_retrieval_confidence(selected_scores)
retrieval_ok = (
    confidence["best"] >= min_best
    and confidence["avg_top3"] >= MIN_AVG_TOP3
    and confidence["spread"] >= MIN_SPREAD
    and len(selected) >= MIN_SELECTED_CHUNKS
)

key_evidence = extract_key_evidence(selected, source)
public_pages = unique_pages(selected, max_pages=8)
public_cit_line = format_public_citations(source, public_pages)


def hard_not_found_message() -> str:
    return "Non ho trovato la risposta nei documenti caricati."


# ============================================================
# ROUTING: GUARDRAIL TOPIC (if / elif / else)
# ============================================================
if topic == "mansioni":
    if retrieval_ok:
        rules_m = extract_mansioni_rules(selected)
        if not rules_m.get("has_trattamento"):
            extra_q = "ha diritto al trattamento corrispondente all'attività svolta mansioni superiori 30 giorni 60 giorni"
            qvec2 = np.array(emb.embed_query(extra_q), dtype=np.float32)
            sims2 = cosine_scores(qvec2, mat_norm)
            extra = [meta[int(ii)] for ii in np.argsort(-sims2)[:10] if not is_parte_sesta_chunk(meta[int(ii)].get("text", ""))]
            rules_m = extract_mansioni_rules(_dedup_near_identical(selected + _limit_per_page(_dedup_near_identical(extra), 2)))
    else:
        rules_m = {"has_trattamento": True, "found_30": True, "found_60": True,
                   "has_esclusione": False, "has_formazione": False, "pages": public_pages}

    public_ans = mansioni_public_answer(user_input, rules_m)
    payload: Dict[str, Any] = {"role": "assistant", "content": public_ans}
    if st.session_state.is_admin:
        payload["debug"] = {"topic": topic, "model": "deterministico", "queries": queries,
                            "confidence": confidence, "evidence": key_evidence, "pages": rules_m.get("pages")}
    st.session_state.last_topic = topic
    st.session_state.messages.append(payload)
    st.rerun()

elif topic == "straordinari":
    public_ans = straordinari_public_answer(user_input, public_pages)
    payload = {"role": "assistant", "content": public_ans}
    if st.session_state.is_admin:
        payload["debug"] = {"topic": topic, "model": "deterministico", "confidence": confidence}
    st.session_state.last_topic = topic
    st.session_state.messages.append(payload)
    st.rerun()

elif topic == "notturno_ordinario":
    pages_for_cit = public_pages or unique_pages(fullscan_notturno_ordinario(meta), 8) or unique_pages(selected, 8)
    public_ans = (
        "Il **lavoro notturno ordinario** (nell'ambito dell'orario normale, **non** straordinario) "
        "è compensato con una **maggiorazione del 26%** sulla retribuzione.\n\n"
        "Se la prestazione notturna è resa **in straordinario**, si applicano le regole dello **straordinario notturno**."
        + "\n\n" + format_public_citations("CCNL", pages_for_cit)
    )
    payload = {"role": "assistant", "content": public_ans}
    if st.session_state.is_admin:
        payload["debug"] = {"topic": topic, "model": "deterministico", "percentuale": "26%", "confidence": confidence}
    st.session_state.last_topic = topic
    st.session_state.messages.append(payload)
    st.rerun()

elif topic == "ferie":
    pass  # nessun guardrail hardcoded: risponde tramite retrieval CCNL + LLM


elif topic == "tfr":
    payload = {"role": "assistant", "content": tfr_public_answer(public_pages)}
    if st.session_state.is_admin:
        payload["debug"] = {"topic": topic, "model": "deterministico", "confidence": confidence}
    st.session_state.last_topic = topic
    st.session_state.messages.append(payload)
    st.rerun()

elif topic == "maternita":
    payload = {"role": "assistant", "content": maternita_public_answer(public_pages)}
    if st.session_state.is_admin:
        payload["debug"] = {"topic": topic, "model": "deterministico", "confidence": confidence}
    st.session_state.last_topic = topic
    st.session_state.messages.append(payload)
    st.rerun()

elif topic == "preavviso":
    payload = {"role": "assistant", "content": preavviso_public_answer(public_pages)}
    if st.session_state.is_admin:
        payload["debug"] = {"topic": topic, "model": "deterministico", "confidence": confidence}
    st.session_state.last_topic = topic
    st.session_state.messages.append(payload)
    st.rerun()

elif topic == "congedo_matrimoniale":
    if not retrieval_ok:
        # Retrieval debole: usa guardrail parziale hardcoded
        payload = {"role": "assistant", "content": congedo_matrimoniale_public_answer(public_pages)}
        if st.session_state.is_admin:
            payload["debug"] = {"topic": topic, "model": "deterministico_fallback", "confidence": confidence}
        st.session_state.last_topic = topic
        st.session_state.messages.append(payload)
        st.rerun()
    # else: retrieval ok → cade nel blocco LLM

elif topic == "permessi":
    if source != "IPZS":
        st.session_state.messages.append({"role": "assistant", "content": "⏳ Per i **permessi IPZS** serve l'indice IPZS. Ricarica la pagina per indicizzare automaticamente."})
        st.rerun()
    if is_lista_permessi_question(user_input):
        titles = extract_ipzs_permessi_titles_from_file(IPZS_TXT_PATH)
        elenco = "\n".join([f"- {t}" for t in titles]) if titles else "(nessun titolo trovato)"
        public_ans = (
            "Ecco l'**elenco completo dei permessi/giustificativi** presenti nelle schede IPZS:\n\n"
            f"{elenco}\n\n**Fonte:** IPZS Permessi"
        )
        payload = {"role": "assistant", "content": public_ans}
        if st.session_state.is_admin:
            payload["debug"] = {"topic": topic, "model": "deterministico", "count": len(titles) if titles else 0}
        st.session_state.last_topic = topic
        st.session_state.messages.append(payload)
        st.rerun()
    # permessi specifici → LLM

else:
    pass  # malattia, rol_exfest, altro → LLM


# ============================================================
# SAFETY GUARD
# ============================================================
DETERMINISTICI = {"mansioni", "straordinari", "notturno_ordinario", "tfr", "maternita", "preavviso"}
if topic in DETERMINISTICI:
    st.stop()


# ============================================================
# BLOCCO LLM
# ============================================================
if len(selected) == 0:
    payload = {"role": "assistant", "content": hard_not_found_message()}
    if st.session_state.is_admin:
        payload["debug"] = {"topic": topic, "queries": queries, "confidence": confidence}
    st.session_state.last_topic = topic
    st.session_state.messages.append(payload)
    st.rerun()

context = "\n\n---\n\n".join(
    [f"[{'Scheda' if source == 'IPZS' else 'Pagina'} {c.get('page', '?')}] {c.get('text', '')}" for c in selected]
)
evidence_block = "\n".join([f"- {e}" for e in key_evidence]) if key_evidence else "- (Nessuna evidenza estratta.)"

guardrail_notturno = ""
if topic == "notturno_ordinario":
    guardrail_notturno = "GUARDRAIL: lavoro notturno ORDINARIO. NON usare percentuali straordinario notturno.\n"
elif topic == "straordinari" and "notturn" in enriched_q.lower():
    guardrail_notturno = "GUARDRAIL: straordinario notturno. Usa SOLO percentuali straordinario notturno.\n"

# MIGLIORAMENTO 4: scegli modello
model_to_use = choose_model(user_input)
llm = ChatOpenAI(model=model_to_use, temperature=LLM_TEMPERATURE, api_key=OPENAI_API_KEY)

# MIGLIORAMENTO 7: risposta strutturata JSON
prompt_public = f"""{RULES_PUBLIC}

SORGENTE ATTIVA: {source}
TOPIC: {topic}
{guardrail_notturno}
DOMANDA UTENTE:
{user_input}

DOMANDA CON CONTESTO CONVERSAZIONE:
{enriched_q}

EVIDENZE CHIAVE:
{evidence_block}

CONTESTO DOCUMENTI:
{context}
"""

try:
    raw_response = llm.invoke(prompt_public).content
    clean = re.sub(r"```json|```", "", raw_response).strip()
    parsed = json.loads(clean)
    risposta   = (parsed.get("risposta") or "").strip()
    fonte      = (parsed.get("fonte") or "").strip()
    confidenza = (parsed.get("confidenza") or "media").strip()
    avvertenza = (parsed.get("avvertenza") or "").strip()

    public_raw = risposta
    if fonte and "fonte" not in risposta.lower():
        public_raw += f"\n\n**Fonte:** {fonte}"
    elif not fonte and public_cit_line:
        public_raw += f"\n\n{public_cit_line}"
    if avvertenza:
        public_raw += f"\n\n⚠️ *{avvertenza}*"

    confidenza_emoji = {"alta": "🟢", "media": "🟡", "bassa": "🔴"}.get(confidenza, "🟡")

except Exception:
    public_raw = raw_response if "raw_response" in dir() and raw_response else hard_not_found_message()
    if not re.search(r"\bfonte\b\s*:", (public_raw or ""), flags=re.IGNORECASE):
        public_raw = (public_raw or "").rstrip() + "\n\n" + public_cit_line
    confidenza, confidenza_emoji = "bassa", "🔴"

public_raw = (public_raw or "").strip()

payload_llm: Dict[str, Any] = {"role": "assistant", "content": public_raw}
if st.session_state.is_admin:
    payload_llm["debug"] = {
        "topic": topic,
        "model": model_to_use,
        "confidenza": f"{confidenza_emoji} {confidenza}",
        "queries": queries,
        "confidence_retrieval": confidence,
        "retrieval_ok": retrieval_ok,
        "cross_encoder": CE_AVAILABLE,
        "evidence": key_evidence,
    }

st.session_state.last_topic = topic
st.session_state.messages.append(payload_llm)
st.rerun()
