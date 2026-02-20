# app.py — Assistente Contrattuale UILCOM IPZS (pubblico con citazioni + guardrail)
# ✅ Risposte SOLO dal CCNL
# ✅ Pubblico: include SEMPRE citazioni (pagine)
# ✅ Admin: debug + evidenze + chunk/pagine usate
# ✅ Topic reset: se cambia argomento, NON usa memoria breve (evita contaminazioni)
# ✅ Guardrail HARD: se retrieval debole -> "Non ho trovato..."
# ✅ Mansioni superiori: risposta deterministica (NO LLM) + pagine

import os
import json
import re
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
PDF_PATH = os.path.join("documenti", "ccnl.pdf")

INDEX_DIR = "index_ccnl"
VEC_PATH = os.path.join(INDEX_DIR, "vectors.npy")
META_PATH = os.path.join(INDEX_DIR, "chunks.json")

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150

TOP_K_PER_QUERY = 12
TOP_K_FINAL = 18
MAX_MULTI_QUERIES = 12

# Memoria: usata SOLO se stesso argomento (topic)
MEMORY_USER_TURNS = 3

# Hard guardrail retrieval
MIN_BEST_SIMILARITY = 0.24          # se max cosine < soglia => "non trovato"
MIN_SELECTED_CHUNKS = 3             # se troppo pochi chunk => "non trovato"

# Admin debug: quante righe evidenza mostrare
MAX_EVIDENCE_LINES = 18

# LLM
LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0


# ============================================================
# SECRETS / PASSWORDS
# ============================================================
def get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    # Streamlit Cloud: st.secrets
    try:
        if key in st.secrets:  # type: ignore
            return str(st.secrets[key])  # type: ignore
    except Exception:
        pass
    # Env
    return os.getenv(key, default)


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


UILCOM_PASSWORD = get_secret("UILCOM_PASSWORD")        # password iscritti
ADMIN_PASSWORD = get_secret("ADMIN_PASSWORD")          # password admin debug
OPENAI_API_KEY = get_secret("OPENAI_API_KEY")          # obbligatoria


# ============================================================
# PAGE SETUP
# ============================================================
st.set_page_config(page_title="Assistente UILCOM IPZS", page_icon="🟦", layout="centered")
st.title(APP_TITLE)
st.markdown(
    "**Accesso riservato agli iscritti UILCOM**  \n"
    "Strumento informativo per facilitare la consultazione del **CCNL Grafici Editoria** e norme applicabili ai lavoratori IPZS.  \n\n"
    "⚠️ Le risposte sono generate **solo** in base al CCNL caricato. "
    "Le citazioni (pagine) sono incluse per permettere la verifica diretta. "
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
    st.warning("Password iscritti non impostata. Imposta UILCOM_PASSWORD in Secrets (Streamlit) o variabile d’ambiente.")
    # Per sviluppo locale puoi forzare qui:
    # st.session_state.auth_ok = True

if not st.session_state.auth_ok:
    st.stop()


# ============================================================
# ADMIN MODE (debug)
# ============================================================
if "is_admin" not in st.session_state:
    st.session_state.is_admin = False

with st.sidebar:
    st.header("⚙️ Controlli")

    # Admin login
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

    # Index management
    st.subheader("📦 Indice CCNL")
    ok_index = os.path.exists(VEC_PATH) and os.path.exists(META_PATH)
    st.write("Indice presente:", "✅" if ok_index else "❌")

    if st.button("Indicizza / Reindicizza CCNL", use_container_width=True):
        # Rebuild
        try:
            with st.spinner("Indicizzazione in corso..."):
                if not os.path.exists(PDF_PATH):
                    raise FileNotFoundError(f"Non trovo il PDF: {PDF_PATH} (metti ccnl.pdf in /documenti)")

                os.makedirs(INDEX_DIR, exist_ok=True)

                loader = PyPDFLoader(PDF_PATH)
                docs = loader.load()

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
                )
                chunks = splitter.split_documents(docs)

                texts = [c.page_content for c in chunks]
                pages = [(int(c.metadata.get("page", 0)) + 1) for c in chunks]  # pagine 1-based

                # Embeddings
                if not OPENAI_API_KEY:
                    raise RuntimeError("Manca OPENAI_API_KEY in Secrets/variabili ambiente.")

                emb = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
                vectors = emb.embed_documents(texts)
                vectors = np.array(vectors, dtype=np.float32)

                np.save(VEC_PATH, vectors)
                with open(META_PATH, "w", encoding="utf-8") as f:
                    json.dump(
                        [{"page": p, "text": t} for p, t in zip(pages, texts)],
                        f,
                        ensure_ascii=False,
                    )

            st.success(f"Indicizzazione completata. Chunk: {len(chunks)}")
            st.rerun()
        except Exception as e:
            st.error(str(e))

    if st.button("🧹 Nuova chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_topic = None
        st.rerun()

    st.divider()
    st.caption("Suggerimento: dopo modifiche a app.py su GitHub, Streamlit Cloud fa auto-redeploy. Se no: **Reboot app**.")


# ============================================================
# HARD FAIL IF NO OPENAI KEY
# ============================================================
if not OPENAI_API_KEY:
    st.error(
        "Manca la variabile **OPENAI_API_KEY**.\n\n"
        "Streamlit Cloud: **Settings → Secrets → OPENAI_API_KEY**\n"
        "Locale: variabile d’ambiente OPENAI_API_KEY"
    )
    st.stop()


# ============================================================
# RETRIEVAL HELPERS
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


def format_public_citations(pages: List[int]) -> str:
    if not pages:
        return ""
    pages_sorted = sorted(pages)
    if len(pages_sorted) == 1:
        return f"**Fonte:** CCNL (pag. {pages_sorted[0]})"
    return f"**Fonte:** CCNL (pagg. {', '.join(map(str, pages_sorted))})"


# ============================================================
# TRIGGERS / CLASSIFIERS
# ============================================================
MANSIONI_TRIGGERS = [
    "mansioni superiori", "mansione superiore", "mansioni più alte", "mansioni piu alte",
    "categoria superiore", "livello superiore", "passaggio di categoria", "cambio categoria",
    "inquadramento superiore", "posto vacante", "sostituzione", "sto sostituendo",
    "differenza paga", "differenza di paga", "pagato di più", "pagato di piu",
    "trattamento corrispondente", "retribuzione corrispondente",
]

CONSERVAZIONE_TRIGGERS = [
    "conservazione del posto",
    "diritto alla conservazione del posto",
    "diritto alla conservazione",
]

PERMESSI_TRIGGERS = [
    "permessi", "permesso", "assenze retribuite", "permessi retribuiti",
    "visite mediche", "lutto", "matrimonio", "nozze", "studio", "esami", "formazione",
    "104", "assemblea", "sindac", "donazione", "rol", "ex festiv", "festività soppresse", "festivita soppresse",
    "festività abolite", "festivita abolite",
]

ROL_EXFEST_TRIGGERS = [
    "rol", "r.o.l", "riduzione orario",
    "ex festiv", "ex-festiv", "exfestiv",
    "festività soppresse", "festivita soppresse",
    "festività abolite", "festivita abolite",
    "festività infrasettimanali", "festivita infrasettimanali",
    "festività infrasettimanali abolite", "festivita infrasettimanali abolite",
]

MALATTIA_TRIGGERS = [
    "malattia", "certificato", "certificat", "inps",
    "comporto", "prognosi", "ricaduta",
    "visita fiscale", "reperibil", "fasce",
    "ricovero", "day hospital",
    "infortunio",
]

STRAORDINARI_TRIGGERS = [
    "straordinario", "straordinari", "maggiorazione", "maggiorazioni",
    "notturno", "festivo", "supplementare"
]

def is_mansioni_question(q: str) -> bool:
    ql = q.lower()
    return any(t in ql for t in MANSIONI_TRIGGERS)

def is_conservazione_context(q: str) -> bool:
    ql = q.lower()
    return any(t in ql for t in CONSERVAZIONE_TRIGGERS)

def is_permessi_question(q: str) -> bool:
    ql = q.lower()
    return any(t in ql for t in PERMESSI_TRIGGERS)

def is_rol_exfest_question(q: str) -> bool:
    ql = q.lower()
    return any(t in ql for t in ROL_EXFEST_TRIGGERS)

def is_malattia_question(q: str) -> bool:
    ql = q.lower()
    return any(t in ql for t in MALATTIA_TRIGGERS)

def is_straordinario_notturno_question(q: str) -> bool:
    ql = q.lower()
    return ("straordin" in ql) and ("notturn" in ql)

def is_lavoro_notturno_question(q: str) -> bool:
    ql = q.lower()
    # lavoro notturno "ordinario": notturno ma NON straordinario
    return ("notturn" in ql) and ("straordin" not in ql)

def detect_topic(q: str) -> str:
    """Topic per evitare che la memoria contamini argomenti diversi."""
    ql = q.lower()

    # PRIORITÀ: domande su malattia devono restare su "malattia"
    if is_malattia_question(ql):
        return "malattia"

    # Mansioni: solo se l’utente parla davvero di mansioni/cambio livello/sostituzione ecc.
    if is_mansioni_question(ql):
        return "mansioni"

    if is_rol_exfest_question(ql):
        return "rol_exfest"

    if is_permessi_question(ql):
        return "permessi"

    if any(t in ql for t in STRAORDINARI_TRIGGERS):
        return "straordinari_notturno_festivo"

    return "altro"


# ============================================================
# MEMORIA BREVE (solo stesso topic)
# ============================================================
def build_enriched_question(current_q: str, current_topic: str) -> str:
    if "messages" not in st.session_state:
        return current_q.strip()

    # Se topic diverso dall'ultimo, non usare memoria
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
# QUERY BUILDER (multi-query)
# ============================================================
def build_queries(q: str) -> List[str]:
    q0 = q.strip()
    qlow = q0.lower()

    qs = [q0, f"{q0} CCNL", f"{q0} regole condizioni", f"{q0} definizione procedura"]

    user_is_rol = is_rol_exfest_question(q0)
    user_is_perm = is_permessi_question(q0)
    user_is_mal = is_malattia_question(q0)
    user_is_mans = is_mansioni_question(q0)
    user_is_conserv = is_conservazione_context(q0)

    # ========== ROL / ex festività ==========
    if user_is_rol:
        qs += [
            "ROL riduzione orario di lavoro monte ore annuo maturazione fruizione",
            "festività soppresse abolite riposi retribuiti quanti giorni",
            "festività infrasettimanali abolite riposi retribuiti",
            "riposi retribuiti in sostituzione delle festività abolite",
        ]

    # ========== Permessi generici ==========
    if user_is_perm and (not user_is_rol):
        qs += [
            "permessi retribuiti tipologie CCNL elenco completo",
            "assenze retribuite tipologie visite mediche lutto matrimonio 104 sindacali studio donazione sangue",
            "permessi sindacali assemblea ore retribuite",
            "diritto allo studio 150 ore triennio permessi retribuiti",
            "ROL riduzione orario di lavoro riposi retribuiti",
            "festività soppresse abolite riposi retribuiti",
        ]

   # ========== Malattia ==========
if user_is_mal:
    qs += [
        "malattia trattamento economico percentuali integrazione INPS CCNL",
        "malattia retribuzione primi giorni carenza integrazione azienda",
        "malattia periodo di comporto durata conservazione posto",
        "malattia visite fiscali reperibilità fasce orarie",
        "malattia ricovero ospedaliero trattamento economico",
    ]

    # ========== Straordinari / notturno ==========
    if any(t in qlow for t in STRAORDINARI_TRIGGERS):
        qs += [
            "lavoro straordinario maggiorazioni limiti",
            "straordinario notturno maggiorazione percentuale",
            "lavoro notturno maggiorazione percentuale",
            "notturno ordinario trattamento economico",
            "lavoro festivo maggiorazioni",
        ]

    # ========== Mansioni superiori / categoria ==========
    if user_is_mans or user_is_conserv:
        qs += [
            "mansioni superiori regole generali posto vacante",
            "mansioni superiori 30 giorni consecutivi 60 giorni non consecutivi",
            "assegnazione a mansioni superiori trattamento corrispondente",
            "non si applica in caso di sostituzione di dipendente assente con diritto alla conservazione del posto",
            "inquadramento superiore effetti",
        ]

    out, seen = [], set()
    for x in qs:
        x = x.strip()
        if x and x not in seen:
            out.append(x)
            seen.add(x)
    return out[:MAX_MULTI_QUERIES]


# ============================================================
# EVIDENCE EXTRACTION (robusta)
# ============================================================
def extract_key_evidence(chunks: List[Dict[str, Any]]) -> List[str]:
    patterns = [
        r"\b30\b", r"\b60\b", r"%", r"tre\s+mesi", r"\b3\s+mesi\b",
        r"posto\s+vacante", r"mansioni?\s+superiori?", r"sostituzion",
        r"conservazion.*posto", r"diritto.*conservazion.*posto",
        r"trattamento\s+corrispondente", r"retribuzion",
    ]

    evidences: List[str] = []
    for c in chunks:
        page = c.get("page", "?")
        text = c.get("text", "") or ""
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        for ln in lines:
            ln_low = ln.lower()
            if any(re.search(p, ln_low) for p in patterns):
                ln_clean = " ".join(ln.split())
                if 20 <= len(ln_clean) <= 420:
                    evidences.append(f"(pag. {page}) {ln_clean}")

    out, seen = [], set()
    for e in evidences:
        if e not in seen:
            out.append(e)
            seen.add(e)
    return out[:MAX_EVIDENCE_LINES]


# ============================================================
# OPTIONAL BM25 RERANK (precision boost)
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
# ⭐ HARD GUARDRAIL: MANSIONI SUPERIORI (NO LLM)
# ============================================================
def _find_snippets(patterns: List[str], chunks: List[Dict[str, Any]], max_hits: int = 6) -> List[Dict[str, Any]]:
    hits = []
    for c in chunks:
        txt = (c.get("text") or "")
        tl = txt.lower()
        if any(re.search(p, tl, flags=re.IGNORECASE) for p in patterns):
            hits.append({"page": c.get("page", "?"), "text": txt})
            if len(hits) >= max_hits:
                break
    return hits

def extract_mansioni_rules(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    patt_30 = [r"\b30\b.*giorn", r"trenta.*giorn"]
    patt_60 = [r"\b60\b.*giorn", r"sessanta.*giorn"]
    patt_consec = [r"consecutiv", r"continuativ"]
    patt_non_consec = [r"non\s+consecutiv", r"discontinu", r"non\s+continuativ"]
    patt_tratt = [r"trattamento\s+corrispondente", r"retribuzion.*corrispond", r"diritto\s+al\s+trattamento"]
    patt_esclus = [r"non\s+.*applica", r"sostituzion", r"conservazion.*posto", r"diritto.*conservazion.*posto"]

    txt_all = " ".join([(c.get("text") or "").lower() for c in chunks])

    found_30 = re.search(r"\b30\b", txt_all) is not None and re.search(r"giorn", txt_all) is not None
    found_60 = re.search(r"\b60\b", txt_all) is not None and re.search(r"giorn", txt_all) is not None

    found_30_consec = (re.search(r"\b30\b", txt_all) is not None) and (re.search(r"consecutiv|continuativ", txt_all) is not None)
    found_60_nonconsec = (re.search(r"\b60\b", txt_all) is not None) and (re.search(r"non\s+consecutiv|discontinu|non\s+continuativ", txt_all) is not None)

    has_trattamento = re.search(r"trattamento\s+corrispondente|diritto\s+al\s+trattamento|retribuzion.*corrispond", txt_all) is not None
    has_esclusione = re.search(r"sostituzion.*conservazion|diritto.*conservazion.*posto|non\s+.*applica", txt_all) is not None
    has_3_mesi = re.search(r"tre\s+mesi|\b3\s+mesi\b", txt_all) is not None

    snip_30 = _find_snippets(patt_30 + patt_consec, chunks)
    snip_60 = _find_snippets(patt_60 + patt_non_consec, chunks)
    snip_tratt = _find_snippets(patt_tratt, chunks)
    snip_escl = _find_snippets(patt_esclus, chunks)

    pages = set()
    for arr in (snip_30, snip_60, snip_tratt, snip_escl):
        for s in arr:
            try:
                pages.add(int(s.get("page", 0)))
            except Exception:
                pass

    return {
        "found_30": found_30,
        "found_60": found_60,
        "found_30_consec": found_30_consec,
        "found_60_nonconsec": found_60_nonconsec,
        "has_trattamento": has_trattamento,
        "has_esclusione": has_esclusione,
        "has_3_mesi": has_3_mesi,
        "snip_30": snip_30,
        "snip_60": snip_60,
        "snip_tratt": snip_tratt,
        "snip_escl": snip_escl,
        "pages": sorted([p for p in pages if p]),
    }

def mansioni_public_answer(user_q: str, rules: Dict[str, Any]) -> str:
    ql = user_q.lower()

    diff_paga = rules.get("has_trattamento", False)
    has_30_60 = bool(rules.get("found_30", False) and rules.get("found_60", False))

    parts = []

    if any(x in ql for x in ["pagato", "pagata", "differenza", "trattamento", "retribuzione", "piu", "più"]):
        if diff_paga:
            parts.append("Se vieni adibito a **mansioni superiori**, hai diritto al **trattamento economico corrispondente all’attività svolta** per i giorni/periodi in cui le svolgi.")
        else:
            parts.append("Nel CCNL caricato **non trovo** una previsione esplicita sulla **differenza retributiva** per mansioni superiori (nel materiale recuperato).")

    if any(x in ql for x in ["categoria", "livello", "passaggio", "inquadramento", "definitiv"]):
        if has_30_60:
            parts.append("Per il **passaggio (stabilizzazione) alla categoria/livello superiore**: dal CCNL risultano le soglie di **30 giorni consecutivi** oppure **60 giorni non consecutivi** di adibizione a mansioni superiori.")
        else:
            parts.append("Nel CCNL caricato **non trovo** nel materiale recuperato la regola specifica sui giorni (es. 30/60) per la stabilizzazione del livello.")

    if rules.get("has_esclusione", False):
        parts.append("⚠️ Attenzione: se l’assegnazione avviene per **sostituzione di un lavoratore assente con diritto alla conservazione del posto**, possono applicarsi **limitazioni/esclusioni** previste dal CCNL.")

    m = re.search(r"\b(\d{1,3})\s+giorn", ql)
    if m:
        gg = int(m.group(1))
        if gg <= 10 and diff_paga:
            parts.append(
                f"Quindi, se parli di **{gg} giorni**, in base al CCNL hai diritto alla **retribuzione corrispondente** per quei giorni; "
                "la **stabilizzazione** (passaggio definitivo) richiede invece le soglie indicate (30/60) se previste nel CCNL."
            )

    if not parts:
        if has_30_60:
            parts.append("Per mansioni superiori: dal CCNL risultano le soglie di **30 giorni consecutivi** o **60 giorni non consecutivi** per la stabilizzazione del livello, e le eventuali eccezioni in caso di sostituzione con conservazione del posto.")
        elif diff_paga:
            parts.append("Per mansioni superiori: dal CCNL risulta il diritto al **trattamento corrispondente** all’attività svolta.")
        else:
            parts.append("Non ho trovato la risposta nel CCNL caricato (nel materiale recuperato).")

    # Citazioni pubbliche
    pages = rules.get("pages", []) or []
    cit = format_public_citations([p for p in pages if isinstance(p, int)])
    if cit:
        parts.append(cit)

    return "\n\n".join(parts).strip()

def mansioni_admin_debug(rules: Dict[str, Any]) -> str:
    lines = []
    lines.append(f"- found_30: {rules.get('found_30')}, found_60: {rules.get('found_60')}")
    lines.append(f"- has_trattamento: {rules.get('has_trattamento')}")
    lines.append(f"- has_esclusione: {rules.get('has_esclusione')}")
    lines.append(f"- has_3_mesi: {rules.get('has_3_mesi')} (IGNORATO se 30/60 presenti)")
    lines.append(f"- pages: {rules.get('pages')}")
    def fmt(snips: List[Dict[str, Any]], title: str):
        if not snips:
            lines.append(f"- {title}: (nessuno)")
            return
        lines.append(f"- {title}:")
        for s in snips[:4]:
            p = s.get("page", "?")
            t = " ".join((s.get("text","") or "").split())
            lines.append(f"  • (pag. {p}) {t[:240]}{'...' if len(t)>240 else ''}")
    fmt(rules.get("snip_tratt", []), "Evidenze trattamento corrispondente")
    fmt(rules.get("snip_30", []), "Evidenze 30 giorni")
    fmt(rules.get("snip_60", []), "Evidenze 60 giorni")
    fmt(rules.get("snip_escl", []), "Evidenze esclusioni (sostituzione/conservazione)")
    return "\n".join(lines)


# ============================================================
# SYSTEM RULES (core) — PUBBLICO CON CITAZIONI
# ============================================================
RULES = """
Sei l’assistente UILCOM per lavoratori IPZS.
Devi rispondere in modo chiaro e pratico basandoti SOLO sul contesto (estratti CCNL) fornito.
Non inventare informazioni.

REGOLE IMPORTANTI:
1) Se non trovi nel contesto, scrivi: "Non ho trovato la risposta nel CCNL caricato."
2) NON confondere lavoro notturno con straordinario notturno:
   - Se la domanda è "lavoro notturno" (ordinario), usa solo regole/percentuali del notturno ordinario.
   - Se nel contesto trovi solo "straordinario notturno", devi dirlo e NON applicarlo al notturno ordinario.
3) TERMINOLOGIA EX FESTIVITÀ:
   - Se l’utente dice "ex festività" ma nel CCNL trovi "festività soppresse/abolite/infrasettimanali abolite",
     spiega che nel CCNL la dicitura è quella (equivalente all’uso comune).
4) Permessi:
   - Elenca SOLO le tipologie che trovi nel contesto.
5) PUBBLICO: devi SEMPRE includere una riga finale "Fonte: CCNL (pag. X...)" con le pagine usate.
6) MALATTIA:
   - Se la domanda riguarda la malattia, includi se presenti nel contesto:
     • trattamento economico (percentuali o integrazione)
     • periodo di comporto
     • eventuali regole di reperibilità/visite fiscali
   - Se alcune informazioni non sono nel contesto recuperato, non inventarle.

FORMATO OUTPUT OBBLIGATORIO:

<PUBLIC>
...testo per l’utente...
(Fonte: CCNL (pag. ...))
</PUBLIC>

<ADMIN>
- Evidenze: ...
- Pagine/chunk usati: ...
</ADMIN>
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
        if st.session_state.is_admin and m["role"] == "assistant":
            dbg = m.get("debug", None)
            if dbg:
                with st.expander("🧠 Admin: Query / Evidenze / Chunk (debug)", expanded=False):
                    st.write("**Topic:**", dbg.get("topic", ""))
                    st.write("**Domanda arricchita (memoria breve):**")
                    st.code(dbg.get("enriched_q", ""))
                    st.write("**Query usate:**")
                    st.code("\n".join(dbg.get("queries", [])))
                    st.write("**Best similarity:**", dbg.get("best_similarity", None))
                    st.write("**Evidenze estratte:**")
                    st.code("\n".join(dbg.get("evidence", [])) or "(nessuna)")
                    if dbg.get("mansioni_guardrail"):
                        st.write("**Guardrail mansioni (deterministico):**")
                        st.code(dbg.get("mansioni_guardrail"))
                    st.write("**Chunk selezionati (prime righe):**")
                    for c in dbg.get("selected", []):
                        st.write(f"**Pagina {c.get('page')}**")
                        txt = (c.get("text") or "")
                        st.write(txt[:800] + ("..." if len(txt) > 800 else ""))
                        st.divider()


user_input = st.chat_input("Scrivi una domanda sul CCNL (permessi, ROL/festività soppresse, malattia, straordinari, categorie...)")

if not user_input:
    st.stop()

# Require index
if not (os.path.exists(VEC_PATH) and os.path.exists(META_PATH)):
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Prima devo indicizzare il CCNL: apri la barra laterale e clicca **Indicizza / Reindicizza CCNL**.",
    })
    st.rerun()

# Append user msg
st.session_state.messages.append({"role": "user", "content": user_input})


# ============================================================
# RETRIEVAL PIPELINE (con topic reset)
# ============================================================
topic = detect_topic(user_input)
enriched_q = build_enriched_question(user_input, topic)

vectors, meta = load_index()
mat_norm = normalize_rows(vectors)
emb = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

queries = build_queries(enriched_q)

# Multi-query semantic retrieval
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

# Optional BM25 rerank for final precision
selected = bm25_rerank(enriched_q, selected)

# Evidence
key_evidence = extract_key_evidence(selected)

# Paginate citations (public)
public_pages = unique_pages(selected, max_pages=8)
public_cit_line = format_public_citations(public_pages)

# HARD guardrail retrieval: se debole -> non rispondere
retrieval_ok = (best_similarity >= MIN_BEST_SIMILARITY) and (len(selected) >= MIN_SELECTED_CHUNKS)

def hard_not_found_message() -> str:
    msg = "Non ho trovato la risposta nel CCNL caricato."
    # in modalità pubblica, se non troviamo bene, NON inventiamo pagine
    return msg

# ============================================================
# ⭐ HARD GUARDRAIL MANSIONI: risposta deterministica (stop LLM)
# ============================================================
user_is_mans = (topic == "mansioni")
if user_is_mans:
    if not retrieval_ok:
        public_ans = hard_not_found_message()
    else:
        rules_m = extract_mansioni_rules(selected)
        public_ans = mansioni_public_answer(user_input, rules_m)

    assistant_payload: Dict[str, Any] = {"role": "assistant", "content": public_ans}

    if st.session_state.is_admin:
        assistant_payload["debug"] = {
            "topic": topic,
            "enriched_q": enriched_q,
            "queries": queries,
            "evidence": key_evidence,
            "selected": selected,
            "best_similarity": best_similarity,
            "mansioni_guardrail": mansioni_admin_debug(extract_mansioni_rules(selected)) if retrieval_ok else "(retrieval debole: guardrail non applicato)",
            "bm25_available": BM25_AVAILABLE,
        }

    st.session_state.last_topic = topic
    st.session_state.messages.append(assistant_payload)
    st.rerun()


# ============================================================
# LLM CALL (tutto il resto)
# ============================================================
if not retrieval_ok:
    assistant_payload: Dict[str, Any] = {"role": "assistant", "content": hard_not_found_message()}
    if st.session_state.is_admin:
        assistant_payload["debug"] = {
            "topic": topic,
            "enriched_q": enriched_q,
            "queries": queries,
            "evidence": key_evidence,
            "selected": selected,
            "best_similarity": best_similarity,
            "bm25_available": BM25_AVAILABLE,
            "note": "retrieval_ok=False -> risposta bloccata",
        }
    st.session_state.last_topic = topic
    st.session_state.messages.append(assistant_payload)
    st.rerun()

context = "\n\n---\n\n".join([f"[Pagina {c.get('page','?')}] {c.get('text','')}" for c in selected])
evidence_block = "\n".join([f"- {e}" for e in key_evidence]) if key_evidence else "- (Nessuna evidenza estratta automaticamente.)"

guardrail_notturno = ""
if is_lavoro_notturno_question(enriched_q):
    guardrail_notturno = (
        "GUARDRAIL NOTTURNO: la domanda è su lavoro notturno (ordinario). NON usare percentuali di straordinario notturno.\n"
    )
elif is_straordinario_notturno_question(enriched_q):
    guardrail_notturno = (
        "GUARDRAIL STRAORD. NOTTURNO: la domanda è su straordinario notturno. Usa SOLO le percentuali relative allo straordinario notturno.\n"
    )

llm = ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE, api_key=OPENAI_API_KEY)

prompt = f"""
{RULES}

{guardrail_notturno}

DOMANDA (UTENTE):
{user_input}

DOMANDA ARRICCHITA (MEMORIA BREVE - solo stesso topic):
{enriched_q}

EVIDENZE (se presenti, sono operative):
{evidence_block}

CONTESTO CCNL (estratti):
{context}

RICORDA:
- Nel PUBLIC: risposta pulita MA con citazione finale "Fonte: CCNL (pag. ...)".
- Nel ADMIN: inserisci elenco pagine trovate e 5-10 righe evidenza importanti con (pag. X).
"""

def split_public_admin(text: str) -> Tuple[str, str]:
    pub = ""
    adm = ""
    m_pub = re.search(r"<PUBLIC>(.*?)</PUBLIC>", text, flags=re.DOTALL | re.IGNORECASE)
    m_adm = re.search(r"<ADMIN>(.*?)</ADMIN>", text, flags=re.DOTALL | re.IGNORECASE)
    if m_pub:
        pub = m_pub.group(1).strip()
    else:
        pub = text.strip()
    if m_adm:
        adm = m_adm.group(1).strip()
    return pub, adm

try:
    raw = llm.invoke(prompt).content
except Exception as e:
    raw = f"<PUBLIC>Errore nel generare la risposta: {e}</PUBLIC><ADMIN></ADMIN>"

public_ans, admin_ans = split_public_admin(raw)
public_ans = (public_ans or "").strip()

# Post-guardrail: se modello “dimentica” la fonte, la aggiungiamo noi (solo se abbiamo pagine)
if public_ans:
    has_source = bool(re.search(r"\bfonte\b\s*:", public_ans, flags=re.IGNORECASE))
    if (not has_source) and public_cit_line:
        public_ans = public_ans.rstrip() + "\n\n" + public_cit_line
else:
    public_ans = hard_not_found_message()

assistant_payload: Dict[str, Any] = {
    "role": "assistant",
    "content": public_ans,
}

if st.session_state.is_admin:
    assistant_payload["debug"] = {
        "topic": topic,
        "enriched_q": enriched_q,
        "queries": queries,
        "evidence": key_evidence,
        "selected": selected,
        "best_similarity": best_similarity,
        "admin_llm_section": admin_ans,
        "bm25_available": BM25_AVAILABLE,
    }

st.session_state.last_topic = topic
st.session_state.messages.append(assistant_payload)
st.rerun()

