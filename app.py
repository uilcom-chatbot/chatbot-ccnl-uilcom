# app.py — Assistente Contrattuale UILCOM IPZS (base tua + fix robusti)
# ✅ Risposte SOLO dal CCNL
# ✅ Utenti: risposta pulita (senza fonti)
# ✅ Admin: debug + evidenze + chunk/pagine usate
# ✅ Fix: ex festività = festività soppresse/abolite/infrasettimanali abolite
# ✅ Fix robusto: mansioni superiori (NO 30/60 se non presenti; usa "<= 3 mesi" se presente)
# ✅ Fix robusto: differenza paga (trattamento economico) anche per pochi giorni
# ✅ Fix robusto: lavoro notturno (non scambiare un 60% non-notturno come notturno)
# ✅ Indice vettoriale persistente (vectors.npy + chunks.json)

import os
import json
import re
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
                n = None

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

                n = len(chunks)

            st.success(f"Indicizzazione completata. Chunk: {n}")
            st.rerun()
        except Exception as e:
            st.error(str(e))

    if st.button("🧹 Nuova chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("Suggerimento: dopo modifiche a app.py su GitHub, Streamlit Cloud in genere fa auto-redeploy. Se no: **Reboot app**.")


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

DIFFERENZA_PAGA_TRIGGERS = [
    "differenza paga", "pagamento maggiore", "paga maggiore", "più paga", "piu paga",
    "trattamento economico", "trattamento corrispondente", "retribuzione", "retribuito",
    "quanto mi spetta in più", "quanto mi spetta in piu", "maggior compenso", "compenso maggiore",
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

def is_differenza_paga_question(q: str) -> bool:
    ql = q.lower()
    return any(t in ql for t in DIFFERENZA_PAGA_TRIGGERS)


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

    # ROL/ex festività
    if user_is_rol:
        qs += [
            "ROL riduzione orario di lavoro monte ore annuo maturazione fruizione",
            "festività soppresse abolite riposi retribuiti quanti giorni",
            "festività infrasettimanali abolite riposi retribuiti",
            "riposi retribuiti in sostituzione delle festività abolite",
            "modalità richiesta fruizione ROL e riposi festività abolite preavviso programmazione",
        ]

    # Permessi generici
    if user_is_perm and (not user_is_rol):
        qs += [
            "permessi retribuiti tipologie CCNL elenco completo",
            "assenze retribuite tipologie visite mediche lutto matrimonio 104 sindacali studio donazione sangue",
            "permessi sindacali assemblea ore retribuite",
            "diritto allo studio 150 ore triennio permessi retribuiti",
        ]
        qs += [
            "ROL riduzione orario di lavoro riposi retribuiti",
            "festività soppresse abolite riposi retribuiti",
        ]

    # Malattia
    if user_is_mal:
        qs += [
            "malattia trattamento economico percentuali integrazione",
            "malattia periodo di comporto regole conteggio",
            "malattia obblighi comunicazione certificazione",
            "controlli visite fiscali reperibilità fasce",
        ]

    # Straordinari / notturno
    if any(t in qlow for t in STRAORDINARI_TRIGGERS):
        qs += [
            "lavoro straordinario maggiorazioni limiti",
            "straordinario notturno maggiorazione percentuale",
            "lavoro notturno maggiorazione percentuale",
            "notturno ordinario trattamento economico",
            "lavoro festivo maggiorazioni",
        ]

    # Mansioni superiori / categoria / differenza paga
    if user_is_mans or user_is_conserv or is_differenza_paga_question(q0):
        qs += [
            "mansioni superiori regole generali posto vacante",
            "mansioni superiori trattamento corrispondente all'attività svolta",
            "assegnazione a mansioni superiori effetti inquadramento assegnazione definitiva",
            "dopo un periodo fissato dai contratti collettivi e comunque non superiore a tre mesi",
            "sostituzione di lavoratore assente con diritto alla conservazione del posto",
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
        r"\b30\b", r"\b60\b", r"\b3\s+mesi\b", r"\btre\s+mesi\b",
        r"posto\s+vacante", r"mansioni?\s+superiori?", r"sostituzion",
        r"conservazion.*posto", r"diritto.*conservazion.*posto",
        r"trattament\w*\s+corrispondente\s+all[’']attivit[àa]\s+svolta",
        r"divien\w*\s+definitiv\w*",
        r"\brol\b", r"riduzione\s+orario",
        r"festivit", r"soppresse", r"abolite", r"infrasettimanali",
        r"notturn", r"straordin", r"maggior",
        r"permess", r"assenze?\s+retribuit",
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

def evidence_has_3_mesi(evidence_lines: List[str]) -> bool:
    joined = " ".join(evidence_lines).lower()
    return (re.search(r"\b3\s+mesi\b", joined) is not None) or ("tre mesi" in joined)


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
# MANSIONI: estrazione clausole chiave per risposte determinate
# ============================================================
def find_mansioni_clause_info(selected_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    joined = " ".join([(c.get("text") or "") for c in selected_chunks]).lower()

    found_trattamento = re.search(
        r"trattament\w*\s+corrispondente\s+all[’']attivit[àa]\s+svolta",
        joined, flags=re.IGNORECASE
    ) is not None

    found_3_mesi = re.search(
        r"non\s+superiore\s+a\s+tre\s+mesi|non\s+superiore\s+a\s+3\s+mesi",
        joined, flags=re.IGNORECASE
    ) is not None

    found_conservazione = re.search(
        r"conservazion\w*\s+del\s+posto|diritto\s+alla\s+conservazione\s+del\s+posto",
        joined, flags=re.IGNORECASE
    ) is not None

    snippet = None
    page = None
    for c in selected_chunks:
        txt = (c.get("text") or "")
        if re.search(r"trattament\w*\s+corrispondente\s+all[’']attivit[àa]\s+svolta", txt, flags=re.IGNORECASE):
            page = c.get("page", "?")
            lines = [" ".join(ln.split()) for ln in txt.splitlines() if ln.strip()]
            for ln in lines:
                if "tratt" in ln.lower() and "attivit" in ln.lower():
                    snippet = ln[:350]
                    break
            if snippet is None:
                snippet = txt[:350]
            break

    return {
        "found_trattamento": found_trattamento,
        "found_3_mesi": found_3_mesi,
        "found_conservazione": found_conservazione,
        "page": page,
        "snippet": snippet
    }


# ============================================================
# SYSTEM RULES (core)
# ============================================================
RULES = """
Sei l’assistente UILCOM per lavoratori IPZS.
Devi rispondere in modo chiaro e pratico basandoti SOLO sul contesto (estratti CCNL) fornito.
Non inventare informazioni.

REGOLE IMPORTANTI:
1) Se non trovi nel contesto, scrivi: "Non ho trovato la risposta nel CCNL caricato."
2) NON confondere lavoro notturno con straordinario notturno.
3) MANSIONI SUPERIORI:
   - Se nel contesto c'è "trattamento corrispondente all'attività svolta", va detto (differenza paga).
   - Non inventare 30/60. Se non compaiono, NON citarli.
   - Se nel contesto c'è "periodo ... non superiore a tre mesi", va usato.
4) TERMINOLOGIA EX FESTIVITÀ:
   - Se l’utente dice "ex festività" ma nel CCNL trovi "festività soppresse/abolite/infrasettimanali abolite",
     spiega che nel CCNL la dicitura è quella (equivalente all’uso comune).
5) Output:
   - <PUBLIC> risposta pulita, senza pagine
   - <ADMIN> solo per debug admin
"""

# ============================================================
# CHAT STATE
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if st.session_state.is_admin and m["role"] == "assistant":
            dbg = m.get("debug", None)
            if dbg:
                with st.expander("🧠 Admin: Query / Evidenze / Chunk (debug)", expanded=False):
                    st.write("**Domanda arricchita (memoria breve):**")
                    st.code(dbg.get("enriched_q", ""))
                    st.write("**Query usate:**")
                    st.code("\n".join(dbg.get("queries", [])))
                    st.write("**Evidenze estratte:**")
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

# Require index
if not (os.path.exists(VEC_PATH) and os.path.exists(META_PATH)):
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Prima devo indicizzare il CCNL: apri la barra laterale e clicca **Indicizza / Reindicizza CCNL**.",
    })
    st.rerun()

st.session_state.messages.append({"role": "user", "content": user_input})


# ============================================================
# RETRIEVAL PIPELINE
# ============================================================
enriched_q = build_enriched_question(user_input)

vectors, meta = load_index()
mat_norm = normalize_rows(vectors)
emb = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

queries = build_queries(enriched_q)

scores_best: Dict[int, float] = {}

for q in queries:
    qvec = np.array(emb.embed_query(q), dtype=np.float32)
    sims = cosine_scores(qvec, mat_norm)
    top_idx = np.argsort(-sims)[:TOP_K_PER_QUERY]
    for i in top_idx:
        s = float(sims[int(i)])
        if (int(i) not in scores_best) or (s > scores_best[int(i)]):
            scores_best[int(i)] = s

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
                s = float(sims[int(i)])
                if (int(i) not in scores_best) or (s > scores_best[int(i)]):
                    scores_best[int(i)] = s

# Flags
user_is_mans = is_mansioni_question(enriched_q)
user_is_conserv = is_conservazione_context(enriched_q)
user_is_notturno = is_lavoro_notturno_question(enriched_q)
user_is_straord_notturno = is_straordinario_notturno_question(enriched_q)
user_is_diffpaga = is_differenza_paga_question(enriched_q)

# Re-ranking with domain boosts + anti-error
for i in list(scores_best.keys()):
    txt = (meta[i].get("text") or "").lower()
    boost = 0.0

    # Mansioni superiori boosts
    if user_is_mans or user_is_conserv or user_is_diffpaga:
        if "tratt" in txt and "attivit" in txt and "svolta" in txt:
            boost += 0.16
        if "non superiore a tre mesi" in txt or "non superiore a 3 mesi" in txt:
            boost += 0.12
        if "conservazione del posto" in txt or "diritto alla conservazione" in txt:
            boost += 0.10
        if "posto vacante" in txt:
            boost += 0.06

        # IMPORTANTISSIMO: non inventare 30/60, quindi boost solo se compaiono davvero
        if re.search(r"\b30\b", txt) and re.search(r"\b60\b", txt):
            boost += 0.10

    # Notturno
    if user_is_notturno:
        if "notturn" in txt and "straordin" not in txt:
            boost += 0.10
        if "60%" in txt and "notturn" not in txt:
            boost -= 0.18  # evita errori: 60% non-notturno
        if "straordin" in txt and "notturn" in txt:
            boost -= 0.08

    if user_is_straord_notturno:
        if "straordin" in txt and "notturn" in txt:
            boost += 0.12

    scores_best[i] = scores_best[i] + boost

final_idx = sorted(scores_best.keys(), key=lambda i: scores_best[i], reverse=True)[:TOP_K_FINAL]
selected = [meta[i] for i in final_idx]
selected = bm25_rerank(enriched_q, selected)

context = "\n\n---\n\n".join([f"[Pagina {c.get('page','?')}] {c.get('text','')}" for c in selected])
key_evidence = extract_key_evidence(selected)

# ============================================================
# HARD OVERRIDE 1: DIFFERENZA PAGA (mansioni superiori)
# ============================================================
if user_is_diffpaga and user_is_mans:
    clause = find_mansioni_clause_info(selected)

    if clause["found_trattamento"]:
        # Risposta deterministica corretta (niente 30/60)
        public_ans = (
            "Sì. Se ti vengono assegnate **mansioni superiori**, hai diritto al **trattamento economico "
            "corrispondente all’attività svolta** (quindi alla differenza paga) anche se l’assegnazione dura pochi giorni.\n\n"
            "Le soglie temporali (quando presenti) riguardano gli **effetti sull’inquadramento / assegnazione definitiva**, "
            "non il diritto alla retribuzione della mansione effettivamente svolta.\n\n"
            "Se l’assegnazione avviene per **sostituzione** di un lavoratore assente con **diritto alla conservazione del posto**, "
            "gli effetti di stabilizzazione/definitività non si applicano (resta fermo quanto previsto dal CCNL per il trattamento economico)."
        )

        assistant_payload: Dict[str, Any] = {"role": "assistant", "content": public_ans}
        if st.session_state.is_admin:
            ev = []
            if clause["page"] is not None and clause["snippet"]:
                ev.append(f"(pag. {clause['page']}) {clause['snippet']}")
            assistant_payload["debug"] = {
                "enriched_q": enriched_q,
                "queries": queries,
                "evidence": ev or key_evidence,
                "selected": selected,
                "admin_llm_section": "(override: differenza paga mansioni superiori)",
                "bm25_available": BM25_AVAILABLE,
            }

        st.session_state.messages.append(assistant_payload)
        st.rerun()

# ============================================================
# HARD OVERRIDE 2: NOTTURNO – evita 60% se non è notturno
# ============================================================
if user_is_notturno:
    joined = " ".join([(c.get("text") or "") for c in selected]).lower()
    has_notturno = ("notturn" in joined)
    has_60 = ("60%" in joined or "maggiorata del 60" in joined)

    # Se nel contesto NON c'è "notturno" ma appare 60%, non deve essere usato per il notturno.
    if (not has_notturno) and has_60:
        public_ans = (
            "Nel contesto recuperato **non trovo una percentuale specifica** riferita al **lavoro notturno ordinario**.\n\n"
            "Vedo invece un riferimento al **60%**, ma **non risulta collegato al lavoro notturno** negli estratti selezionati: "
            "quindi non posso usarlo per rispondere correttamente.\n\n"
            "Prova a chiedere: **“Nel CCNL qual è la maggiorazione per lavoro notturno ordinario?”** oppure reindicizza e riprova."
        )
        assistant_payload: Dict[str, Any] = {"role": "assistant", "content": public_ans}
        if st.session_state.is_admin:
            assistant_payload["debug"] = {
                "enriched_q": enriched_q,
                "queries": queries,
                "evidence": key_evidence,
                "selected": selected,
                "admin_llm_section": "(override: protezione anti-60%-non-notturno)",
                "bm25_available": BM25_AVAILABLE,
            }
        st.session_state.messages.append(assistant_payload)
        st.rerun()


# ============================================================
# LLM CALL
# ============================================================
llm = ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE, api_key=OPENAI_API_KEY)

evidence_block = "\n".join([f"- {e}" for e in key_evidence]) if key_evidence else "- (Nessuna evidenza estratta automaticamente.)"

# Guardrail mansioni: NON dire 30/60 se non compaiono; usa <=3 mesi se compare
guardrail_mansioni = ""
if user_is_mans:
    has30_60 = evidence_has_30_60(key_evidence)
    has3 = evidence_has_3_mesi(key_evidence)
    if has30_60:
        guardrail_mansioni += "GUARDRAIL: nel contesto compaiono 30 e 60: puoi riportarli SOLO se sono chiaramente riferiti alle mansioni superiori.\n"
    else:
        guardrail_mansioni += "GUARDRAIL: NON compaiono 30/60 nel contesto mansioni: NON citarli.\n"
    if has3:
        guardrail_mansioni += "GUARDRAIL: nel contesto c'è il limite 'non superiore a tre mesi': usalo per i tempi di assegnazione definitiva.\n"

guardrail_notturno = ""
if user_is_notturno:
    guardrail_notturno = "GUARDRAIL: domanda su lavoro notturno ordinario: NON usare percentuali di straordinario notturno.\n"
elif user_is_straord_notturno:
    guardrail_notturno = "GUARDRAIL: domanda su straordinario notturno: usa SOLO percentuali di straordinario notturno.\n"

prompt = f"""
{RULES}

{guardrail_mansioni}
{guardrail_notturno}

DOMANDA (UTENTE):
{user_input}

DOMANDA ARRICCHITA (MEMORIA BREVE):
{enriched_q}

EVIDENZE (se presenti, sono operative):
{evidence_block}

CONTESTO CCNL (estratti):
{context}

RICORDA:
- Nel PUBLIC: risposta pulita, senza pagine.
- Nel ADMIN: inserisci elenco pagine trovate e 5-10 righe evidenza più importanti con (pag. X).
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
        "admin_llm_section": admin_ans,
        "bm25_available": BM25_AVAILABLE,
    }

st.session_state.messages.append(assistant_payload)
st.rerun()
