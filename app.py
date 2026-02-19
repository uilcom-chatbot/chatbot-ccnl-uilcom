# app.py — Assistente Contrattuale UILCOM IPZS (versione definitiva + Upgrade 3)
# ✅ Risposte SOLO dal CCNL (PUBLIC pulito)
# ✅ Admin: debug + evidenze + chunk/pagine usate (solo admin)
# ✅ Upgrade A: Dataset UILCOM (Q/A reali) come “guardrail di precisione”
# ✅ Upgrade B: Auto-correzione Admin → aggiunge domande/risposte al dataset_uilcom.json
# ✅ Fix: ex festività = festività soppresse/abolite/infrasettimanali abolite
# ✅ Fix: mansioni superiori (30/60 + posto vacante + esclusione conservazione posto)
# ✅ Fix: lavoro notturno vs straordinario notturno (non confondere %)
# ✅ Indice vettoriale persistente (vectors.npy + chunks.json)

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

MEMORY_USER_TURNS = 3

# Permessi: quante categorie diverse provare a coprire
PERMESSI_MIN_CATEGORY_COVERAGE = 3

# Admin debug: quante righe evidenza mostrare
MAX_EVIDENCE_LINES = 18

# LLM
LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0

# Dataset UILCOM (Q/A)
DATASET_PATH = "dataset_uilcom.json"
DATASET_INDEX_DIR = "index_dataset"
DATASET_VEC_PATH = os.path.join(DATASET_INDEX_DIR, "dataset_vectors.npy")
DATASET_META_PATH = os.path.join(DATASET_INDEX_DIR, "dataset_meta.json")
DATASET_TOP_K = 4  # quante Q/A mostrare al modello come guardrail


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
    "⚠️ Le risposte sono generate **solo** in base al CCNL caricato. Per casi specifici o interpretazioni, rivolgersi a RSU/UILCOM o HR."
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
        try:
            with st.spinner("Indicizzazione CCNL in corso..."):
                if not os.path.exists(PDF_PATH):
                    raise FileNotFoundError(f"Non trovo il PDF: {PDF_PATH} (metti ccnl.pdf in /documenti)")

                os.makedirs(INDEX_DIR, exist_ok=True)

                loader = PyPDFLoader(PDF_PATH)
                docs = loader.load()

                splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
                chunks = splitter.split_documents(docs)

                texts = [c.page_content for c in chunks]
                pages = [(int(c.metadata.get("page", 0)) + 1) for c in chunks]  # 1-based

                if not OPENAI_API_KEY:
                    raise RuntimeError("Manca OPENAI_API_KEY in Secrets/variabili ambiente.")

                emb = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
                vectors = np.array(emb.embed_documents(texts), dtype=np.float32)

                np.save(VEC_PATH, vectors)
                with open(META_PATH, "w", encoding="utf-8") as f:
                    json.dump([{"page": p, "text": t} for p, t in zip(pages, texts)], f, ensure_ascii=False)

            st.success(f"Indicizzazione CCNL completata. Chunk: {len(texts)}")
            st.rerun()
        except Exception as e:
            st.error(str(e))

    if st.button("🧹 Nuova chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_user_question = ""
        st.session_state.last_public_answer = ""
        st.rerun()

    st.divider()
    st.caption("Dopo modifiche a app.py su GitHub, Streamlit Cloud di solito fa auto-redeploy. Se no: **Reboot app**.")


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
# IO HELPERS (dataset)
# ============================================================
def safe_read_json(path: str, default):
    try:
        if not os.path.exists(path):
            return default
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def safe_write_json(path: str, data) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def normalize_rows(mat: np.ndarray) -> np.ndarray:
    return mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)

def cosine_scores(query_vec: np.ndarray, mat_norm: np.ndarray) -> np.ndarray:
    q = query_vec / (np.linalg.norm(query_vec) + 1e-12)
    return mat_norm @ q


# ============================================================
# LOAD INDEX CCNL
# ============================================================
def load_index_ccnl() -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    vectors = np.load(VEC_PATH)
    meta = safe_read_json(META_PATH, [])
    fixed = []
    for item in meta:
        if isinstance(item, dict) and "text" in item and "page" in item:
            fixed.append({"page": item.get("page", "?"), "text": item.get("text", "")})
        elif isinstance(item, str):
            fixed.append({"page": "?", "text": item})
        else:
            fixed.append({"page": "?", "text": str(item)})
    return vectors, fixed


# ============================================================
# DATASET UILCOM: BUILD / LOAD INDEX
# ============================================================
def dataset_exists() -> bool:
    return os.path.exists(DATASET_PATH)

def load_dataset_entries() -> List[Dict[str, Any]]:
    data = safe_read_json(DATASET_PATH, [])
    out = []
    if isinstance(data, list):
        for x in data:
            if isinstance(x, dict):
                domanda = str(x.get("domanda", "")).strip()
                risposta = str(x.get("risposta_corretta", "")).strip()
                categoria = str(x.get("categoria", "")).strip()
                if domanda and risposta:
                    out.append({"domanda": domanda, "risposta_corretta": risposta, "categoria": categoria})
    return out

def build_dataset_index(entries: List[Dict[str, Any]], emb: OpenAIEmbeddings) -> None:
    os.makedirs(DATASET_INDEX_DIR, exist_ok=True)
    # Embedding on "domanda" (search by user question)
    qs = [e["domanda"] for e in entries]
    if not qs:
        # remove any old index
        try:
            if os.path.exists(DATASET_VEC_PATH):
                os.remove(DATASET_VEC_PATH)
            if os.path.exists(DATASET_META_PATH):
                os.remove(DATASET_META_PATH)
        except Exception:
            pass
        return
    vecs = np.array(emb.embed_documents(qs), dtype=np.float32)
    np.save(DATASET_VEC_PATH, vecs)
    safe_write_json(DATASET_META_PATH, entries)

def load_dataset_index() -> Tuple[Optional[np.ndarray], List[Dict[str, Any]]]:
    if not (os.path.exists(DATASET_VEC_PATH) and os.path.exists(DATASET_META_PATH)):
        return None, []
    vecs = np.load(DATASET_VEC_PATH)
    meta = safe_read_json(DATASET_META_PATH, [])
    if not isinstance(meta, list):
        meta = []
    fixed = []
    for e in meta:
        if isinstance(e, dict):
            fixed.append({
                "domanda": str(e.get("domanda", "")).strip(),
                "risposta_corretta": str(e.get("risposta_corretta", "")).strip(),
                "categoria": str(e.get("categoria", "")).strip(),
            })
    return vecs, fixed

def get_dataset_hits(query: str, emb: OpenAIEmbeddings) -> List[Dict[str, Any]]:
    vecs, meta = load_dataset_index()
    if vecs is None or not meta:
        return []
    mat_norm = normalize_rows(vecs)
    qvec = np.array(emb.embed_query(query), dtype=np.float32)
    sims = cosine_scores(qvec, mat_norm)
    top_idx = np.argsort(-sims)[:DATASET_TOP_K]
    out = []
    for i in top_idx:
        i = int(i)
        if i < 0 or i >= len(meta):
            continue
        item = meta[i]
        score = float(sims[i])
        out.append({**item, "score": score})
    return out


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
# TRIGGERS / CLASSIFIERS
# ============================================================
MANSIONI_TRIGGERS = [
    "mansioni superiori", "mansione superiore", "mansioni più alte", "mansioni piu alte",
    "categoria superiore", "livello superiore", "passaggio di categoria", "cambio categoria",
    "inquadramento superiore", "posto vacante", "sostituzione", "sto sostituendo",
]

CONSERVAZIONE_TRIGGERS = [
    "maternità", "maternita", "congedo maternità", "congedo maternita",
    "congedo parentale", "parentale",
    "malattia", "infortunio", "aspettativa",
    "conservazione del posto", "diritto alla conservazione del posto",
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
    return ("notturn" in ql) and ("straordin" not in ql)


# ============================================================
# PERMESSI: CATEGORIE (coverage)
# ============================================================
PERMESSI_CATEGORIES = {
    "rol_exfest": [r"\brol\b", r"riduzione\s+orario", r"festivit", r"soppresse", r"abolite"],
    "visite_mediche": [r"visite?\s+med", r"specialist", r"accertament", r"struttur[ae]\s+sanitar"],
    "lutto": [r"\blutto\b", r"decesso", r"familiare", r"grave\s+lutto"],
    "matrimonio": [r"matrimon", r"nozz"],
    "studio_formazione": [r"diritto\s+allo\s+studio", r"\b150\s+ore\b", r"\bstudio\b", r"\besami\b", r"formazion"],
    "legge_104": [r"\b104\b", r"legge\s*104", r"handicap"],
    "sindacali": [r"sindacal", r"\brsu\b", r"assemblea", r"permessi?\s+sindacal"],
    "donazione_sangue": [r"donazion", r"sangue", r"emocomponent"],
    "altri_permessi": [r"permessi?\s+retribuit", r"assenze?\s+retribuit", r"conged"],
}

def permessi_category_coverage(chunks: List[Dict[str, Any]]) -> Tuple[int, List[str]]:
    joined = " ".join([(c.get("text") or "") for c in chunks]).lower()
    found = set()
    for cat, pats in PERMESSI_CATEGORIES.items():
        for p in pats:
            if re.search(p, joined, flags=re.IGNORECASE):
                found.add(cat)
                break
    return len(found), sorted(found)

def build_permessi_expansion_queries(user_q: str) -> List[str]:
    base = user_q.strip()
    qs = [
        f"{base} ROL festività soppresse abolite riposi retribuiti",
        f"{base} permessi visite mediche retribuiti",
        f"{base} permessi lutto retribuiti",
        f"{base} permessi matrimonio retribuiti",
        f"{base} permessi legge 104 retribuiti",
        f"{base} permessi sindacali assemblea RSU",
        f"{base} permessi diritto allo studio 150 ore",
        f"{base} permessi donazione sangue",
        f"{base} assenze retribuite elenco tipologie",
    ]
    out, seen = [], set()
    for x in qs:
        x = x.strip()
        if x and x not in seen:
            out.append(x)
            seen.add(x)
    return out[:MAX_MULTI_QUERIES]


# ============================================================
# MEMORIA BREVE
# ============================================================
def build_enriched_question(current_q: str) -> str:
    if "messages" not in st.session_state:
        return current_q.strip()

    user_msgs = [m["content"] for m in st.session_state.messages if m.get("role") == "user" and m.get("content")]
    prev = user_msgs[:-1] if (user_msgs and user_msgs[-1].strip() == current_q.strip()) else user_msgs
    last = prev[-MEMORY_USER_TURNS:] if prev else []
    last = [x.strip() for x in last if x.strip()]
    if not last:
        return current_q.strip()

    return (
        "CONTESTO CONVERSAZIONE (ultime richieste utente):\n"
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

    if user_is_rol:
        qs += [
            "ROL riduzione orario di lavoro monte ore annuo maturazione fruizione",
            "festività soppresse abolite riposi retribuiti quanti giorni",
            "festività infrasettimanali abolite riposi retribuiti",
            "riposi retribuiti in sostituzione delle festività abolite",
            "modalità richiesta fruizione ROL e riposi festività abolite preavviso programmazione",
            "residui ROL scadenze eventuale monetizzazione (se prevista)",
        ]

    if user_is_perm and (not user_is_rol):
        qs += [
            "permessi retribuiti tipologie CCNL elenco completo",
            "assenze retribuite tipologie visite mediche lutto matrimonio 104 sindacali studio donazione sangue",
            "permessi sindacali assemblea ore retribuite",
            "diritto allo studio 150 ore triennio permessi retribuiti",
            "permessi per esami lavoratori studenti una settimana di calendario all'anno",
            "ROL riduzione orario di lavoro riposi retribuiti",
            "festività soppresse abolite riposi retribuiti",
        ]

    if user_is_mal:
        qs += [
            "malattia trattamento economico percentuali integrazione",
            "malattia periodo di comporto regole conteggio",
            "malattia obblighi comunicazione certificazione",
            "controlli visite fiscali reperibilità fasce",
            "malattia durante ferie sospensione ferie",
            "ricovero ospedaliero day hospital ricaduta",
        ]

    if any(t in qlow for t in STRAORDINARI_TRIGGERS):
        qs += [
            "lavoro straordinario maggiorazioni limiti",
            "straordinario notturno maggiorazione percentuale",
            "lavoro notturno maggiorazione percentuale",
            "notturno ordinario trattamento economico",
            "lavoro festivo maggiorazioni",
        ]

    if user_is_mans or user_is_conserv:
        qs += [
            "mansioni superiori regole generali posto vacante",
            "mansioni superiori 30 giorni consecutivi 60 giorni discontinui",
            "assegnazione a mansioni superiori effetti inquadramento",
            "non si applica in caso di sostituzione di dipendente assente con diritto alla conservazione del posto",
            "trattamento economico durante mansioni superiori",
            "formazione addestramento affiancamento non costituisce mansioni superiori",
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
        r"\b30\b", r"\b60\b", r"\b\d{1,3}\b", r"%",
        r"posto\s+vacante", r"mansioni?\s+superiori?", r"sostituzion",
        r"conservazion.*posto", r"diritto.*conservazion.*posto",
        r"non\s+si\s+applica", r"non\s+costituisc",
        r"affiancament", r"addestrament", r"formazion",
        r"\brol\b", r"riduzione\s+orario",
        r"festivit", r"soppresse", r"abolite", r"infrasettimanali",
        r"notturn", r"straordin", r"maggior",
        r"permess", r"assenze?\s+retribuit",
        r"assemblea", r"sindacal", r"\b104\b",
        r"diritto\s+allo\s+studio", r"\b150\s+ore\b",
        r"malatt", r"comporto", r"certificat", r"reperibil", r"visita\s+fiscale",
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

def evidence_has_30_60(evidence_lines: List[str]) -> bool:
    joined = " ".join(evidence_lines).lower()
    return (re.search(r"\b30\b", joined) is not None) and (re.search(r"\b60\b", joined) is not None)


# ============================================================
# SYSTEM RULES (core) + DATASET GUARDRAIL
# ============================================================
RULES = """
Sei l’assistente UILCOM per lavoratori IPZS.
Devi rispondere in modo chiaro e pratico basandoti SOLO sul contesto (estratti CCNL) fornito.
Non inventare informazioni.

IMPORTANTE (dataset UILCOM):
- Potresti ricevere anche un elenco di Q/A "Dataset UILCOM" (basato su casi reali).
- Usa il dataset SOLO come "guardrail" per interpretare correttamente il CCNL, MA NON deve contraddire il contesto CCNL.
- Se dataset e CCNL sono in conflitto o il CCNL non conferma, devi seguire il CCNL e dichiarare che nel CCNL recuperato non emerge.

REGOLE IMPORTANTI:
1) Se non trovi nel contesto, scrivi: "Non ho trovato la risposta nel CCNL caricato."
2) NON confondere lavoro notturno con straordinario notturno:
   - Se la domanda è "lavoro notturno" (ordinario), usa solo regole/percentuali del notturno ordinario.
   - Se nel contesto trovi solo "straordinario notturno", devi dirlo e NON applicarlo al notturno ordinario.
3) MANSIONI SUPERIORI / CATEGORIA:
   - Se nel contesto emergono 30 giorni consecutivi / 60 giorni non consecutivi, questi vanno riportati.
   - Se nel contesto emerge la distinzione posto vacante vs sostituzione con conservazione del posto, riportala chiaramente.
4) TERMINOLOGIA EX FESTIVITÀ:
   - Se l’utente dice "ex festività" ma nel CCNL trovi "festività soppresse/abolite/infrasettimanali abolite",
     spiega che nel CCNL la dicitura è quella (equivalente all’uso comune).
5) Permessi:
   - Elenca SOLO le tipologie che trovi nel contesto.
   - Se l’utente chiede "tutti i permessi", includi anche ROL/festività abolite solo se compaiono nel contesto recuperato.
6) Output: prepara una risposta pulita per l’utente (senza citare pagine).
   Le pagine/estratti vanno messi SOLO nella sezione ADMIN.

FORMATO OUTPUT OBBLIGATORIO:

<PUBLIC>
...testo per l’utente...
</PUBLIC>

<ADMIN>
- Evidenze: ...
- Dataset hits: ...
- Pagine/chunk usati: ...
</ADMIN>
"""


# ============================================================
# ADMIN: DATASET UI (auto-correzione)
# ============================================================
def admin_dataset_panel(emb: OpenAIEmbeddings) -> None:
    if not st.session_state.is_admin:
        return

    with st.sidebar:
        st.divider()
        st.subheader("📘 Dataset UILCOM (Auto-correzione)")

        exists = dataset_exists()
        st.write("dataset_uilcom.json:", "✅" if exists else "❌ (crealo nella repo/progetto)")

        entries = load_dataset_entries()
        st.caption(f"Voci dataset attuali: {len(entries)}")

        # Build/Rebuild dataset index
        if st.button("Indicizza dataset UILCOM", use_container_width=True):
            try:
                with st.spinner("Indicizzazione dataset in corso..."):
                    entries = load_dataset_entries()
                    build_dataset_index(entries, emb)
                st.success("Dataset indicizzato.")
            except Exception as e:
                st.error(f"Errore indicizzazione dataset: {e}")

        # Quick add from last Q/A
        st.caption("Suggerimento: dopo una risposta sbagliata, incolla qui la correzione e salva.")

        default_q = st.session_state.get("last_user_question", "")
        default_a = st.session_state.get("last_public_answer", "")

        with st.form("add_dataset_form", clear_on_submit=False):
            domanda = st.text_area("Domanda (utente)", value=default_q, height=80)
            risposta = st.text_area("Risposta corretta (UILCOM)", value="", height=110)
            categoria = st.text_input("Categoria (es. permessi, mansioni_superiori, malattia, straordinari...)", value="")
            salva = st.form_submit_button("➕ Aggiungi al dataset", use_container_width=True)

        if salva:
            try:
                domanda = (domanda or "").strip()
                risposta = (risposta or "").strip()
                categoria = (categoria or "").strip()
                if not domanda or not risposta:
                    st.error("Compila almeno Domanda e Risposta corretta.")
                else:
                    data = safe_read_json(DATASET_PATH, [])
                    if not isinstance(data, list):
                        data = []
                    data.append({
                        "domanda": domanda,
                        "risposta_corretta": risposta,
                        "categoria": categoria
                    })
                    safe_write_json(DATASET_PATH, data)

                    # rebuild dataset index immediately (best UX)
                    with st.spinner("Aggiorno indice dataset..."):
                        build_dataset_index(load_dataset_entries(), emb)

                    st.success("Aggiunto al dataset + indicizzato.")
            except Exception as e:
                st.error(f"Errore salvataggio dataset: {e}")


# ============================================================
# CHAT STATE
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_user_question" not in st.session_state:
    st.session_state.last_user_question = ""
if "last_public_answer" not in st.session_state:
    st.session_state.last_public_answer = ""

# Prepare embeddings instance (used for CCNL + dataset)
emb = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

# Admin dataset panel
admin_dataset_panel(emb)

# Render chat (pulita)
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if st.session_state.is_admin and m["role"] == "assistant":
            dbg = m.get("debug", None)
            if dbg:
                with st.expander("🧠 Admin: Query / Evidenze / Dataset / Chunk (debug)", expanded=False):
                    st.write("**Domanda arricchita (memoria breve):**")
                    st.code(dbg.get("enriched_q", ""))

                    st.write("**Query usate:**")
                    st.code("\n".join(dbg.get("queries", [])))

                    st.write("**Dataset hits (guardrail):**")
                    ds = dbg.get("dataset_hits", [])
                    if ds:
                        for h in ds:
                            st.write(f"- score={h.get('score'):.3f} | cat={h.get('categoria','')} | domanda: {h.get('domanda','')}")
                            st.write(f"  risposta: {h.get('risposta_corretta','')}")
                    else:
                        st.write("(nessuno)")

                    st.write("**Evidenze estratte (CCNL):**")
                    st.code("\n".join(dbg.get("evidence", [])) or "(nessuna)")

                    st.write("**Chunk selezionati (prime righe):**")
                    for c in dbg.get("selected", []):
                        st.write(f"**Pagina {c.get('page')}**")
                        txt = (c.get("text") or "")
                        st.write(txt[:800] + ("..." if len(txt) > 800 else ""))
                        st.divider()


user_input = st.chat_input("Scrivi una domanda sul CCNL (permessi, ROL/festività soppresse, malattia, straordinari, categorie...)")

if not user_input:
    st.stop()

# Require index CCNL
if not (os.path.exists(VEC_PATH) and os.path.exists(META_PATH)):
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Prima devo indicizzare il CCNL: apri la barra laterale e clicca **Indicizza / Reindicizza CCNL**.",
    })
    st.rerun()

# Append user msg
st.session_state.messages.append({"role": "user", "content": user_input})
st.session_state.last_user_question = user_input


# ============================================================
# RETRIEVAL PIPELINE (CCNL + Dataset guardrail)
# ============================================================
enriched_q = build_enriched_question(user_input)

vectors, meta = load_index_ccnl()
mat_norm = normalize_rows(vectors)

queries = build_queries(enriched_q)

# Multi-query semantic retrieval (CCNL)
scores_best: Dict[int, float] = {}
for q in queries:
    qvec = np.array(emb.embed_query(q), dtype=np.float32)
    sims = cosine_scores(qvec, mat_norm)
    top_idx = np.argsort(-sims)[:TOP_K_PER_QUERY]
    for i in top_idx:
        i = int(i)
        s = float(sims[i])
        if (i not in scores_best) or (s > scores_best[i]):
            scores_best[i] = s

provisional_idx = sorted(scores_best.keys(), key=lambda i: scores_best[i], reverse=True)[:TOP_K_FINAL]
provisional_selected = [meta[i] for i in provisional_idx]
provisional_evidence = extract_key_evidence(provisional_selected)

# Guardrail: permessi coverage pass 2
user_is_perm = is_permessi_question(enriched_q)
user_is_rol = is_rol_exfest_question(enriched_q)

if user_is_perm and (not user_is_rol):
    cov_n, _ = permessi_category_coverage(provisional_selected)
    if cov_n < PERMESSI_MIN_CATEGORY_COVERAGE:
        extra_queries = build_permessi_expansion_queries(enriched_q)
        for q in extra_queries:
            qvec = np.array(emb.embed_query(q), dtype=np.float32)
            sims = cosine_scores(qvec, mat_norm)
            top_idx = np.argsort(-sims)[:TOP_K_PER_QUERY]
            for i in top_idx:
                i = int(i)
                s = float(sims[i])
                if (i not in scores_best) or (s > scores_best[i]):
                    scores_best[i] = s

# Re-ranking with domain boosts (mansioni + notturno)
user_is_mans = is_mansioni_question(enriched_q) or ("categoria" in enriched_q.lower())
user_is_conserv = is_conservazione_context(enriched_q)
user_is_notturno = is_lavoro_notturno_question(enriched_q)
user_is_straord_notturno = is_straordinario_notturno_question(enriched_q)

force_30_60 = evidence_has_30_60(provisional_evidence) and user_is_mans

for i in list(scores_best.keys()):
    txt = (meta[i].get("text") or "").lower()
    boost = 0.0

    if user_is_mans or user_is_conserv:
        if re.search(r"\b30\b", txt) and re.search(r"\b60\b", txt):
            boost += 0.18
        if "posto vacante" in txt:
            boost += 0.08
        if "conservazione del posto" in txt or "diritto alla conservazione" in txt:
            boost += 0.08
        if "non si applica" in txt or "non si applicano" in txt:
            boost += 0.07
        if "affianc" in txt or "formaz" in txt or "addestr" in txt:
            boost += 0.03

    if user_is_rol:
        if re.search(r"\brol\b", txt) or "riduzione orario" in txt:
            boost += 0.16
        if "festività soppresse" in txt or "festivita soppresse" in txt or "abolite" in txt or "infrasettimanali" in txt:
            boost += 0.16
        if "diritto allo studio" in txt or "150 ore" in txt:
            boost -= 0.10

    if user_is_notturno:
        if "notturn" in txt and "straordin" not in txt:
            boost += 0.10
        if "straordin" in txt and "notturn" in txt:
            boost -= 0.08

    if user_is_straord_notturno:
        if "straordin" in txt and "notturn" in txt:
            boost += 0.12

    if user_is_perm and (not user_is_rol):
        if "permess" in txt or "assenze retribuite" in txt:
            boost += 0.07

    scores_best[i] = scores_best[i] + boost

final_idx = sorted(scores_best.keys(), key=lambda i: scores_best[i], reverse=True)[:TOP_K_FINAL]
selected = [meta[i] for i in final_idx]

# Optional BM25 rerank for CCNL precision
selected = bm25_rerank(enriched_q, selected)

# Dataset hits (guardrail)
dataset_hits = get_dataset_hits(enriched_q, emb)

# Build contexts
ccnl_context = "\n\n---\n\n".join([f"[Pagina {c.get('page','?')}] {c.get('text','')}" for c in selected])

dataset_context = ""
if dataset_hits:
    lines = []
    for h in dataset_hits:
        cat = h.get("categoria", "")
        lines.append(f"- [cat={cat}] DOMANDA: {h.get('domanda','')}\n  RISPOSTA UILCOM: {h.get('risposta_corretta','')}")
    dataset_context = "\n".join(lines)

key_evidence = extract_key_evidence(selected)
evidence_block = "\n".join([f"- {e}" for e in key_evidence]) if key_evidence else "- (Nessuna evidenza estratta automaticamente.)"

guardrail_mansioni = ""
if force_30_60:
    guardrail_mansioni = "GUARDRAIL MANSIONI: nel contesto emergono 30/60. Devi riportare questi valori se parli di passaggio categoria/mansioni superiori.\n"

guardrail_notturno = ""
if user_is_notturno:
    guardrail_notturno = "GUARDRAIL NOTTURNO: la domanda è su lavoro notturno (ordinario). NON usare percentuali di straordinario notturno.\n"
elif user_is_straord_notturno:
    guardrail_notturno = "GUARDRAIL STRAORD. NOTTURNO: la domanda è su straordinario notturno. Usa SOLO le percentuali relative allo straordinario notturno.\n"


# ============================================================
# LLM CALL
# ============================================================
llm = ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE, api_key=OPENAI_API_KEY)

prompt = f"""
{RULES}

{guardrail_mansioni}
{guardrail_notturno}

DOMANDA (UTENTE):
{user_input}

DOMANDA ARRICCHITA (MEMORIA BREVE):
{enriched_q}

DATASET UILCOM (guardrail, se presente — NON deve contraddire il CCNL):
{dataset_context if dataset_context else "(nessun match dataset)"}

EVIDENZE CCNL (se presenti, sono operative):
{evidence_block}

CONTESTO CCNL (estratti):
{ccnl_context}

RICORDA:
- Nel PUBLIC: risposta pulita, senza pagine e senza citazioni.
- Nel ADMIN: inserisci elenco pagine trovate + evidenze con (pag. X) + elenco dataset hits usati (se presenti).
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

if not public_ans:
    public_ans = "Non ho trovato la risposta nel CCNL caricato."

st.session_state.last_public_answer = public_ans


# ============================================================
# OUTPUT (utente pulito + admin debug)
# ============================================================
assistant_payload: Dict[str, Any] = {
    "role": "assistant",
    "content": public_ans,
}

if st.session_state.is_admin:
    assistant_payload["debug"] = {
        "enriched_q": enriched_q,
        "queries": queries,
        "evidence": key_evidence,
        "selected": selected,
        "dataset_hits": dataset_hits,
        "admin_llm_section": admin_ans,
        "bm25_available": BM25_AVAILABLE,
        "dataset_exists": dataset_exists(),
    }

st.session_state.messages.append(assistant_payload)
st.rerun()
