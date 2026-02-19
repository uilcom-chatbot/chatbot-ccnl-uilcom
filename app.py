import os
import json
import re
import time
import hashlib
from typing import List, Dict, Any, Tuple

import numpy as np
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# Optional: BM25 rerank (precision boost)
try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except Exception:
    HAS_BM25 = False


# ============================================================
# CONFIG
# ============================================================
APP_TITLE = "🟦 Assistente Contrattuale UILCOM IPZS"
APP_ICON = "🟦"

# Percorso PDF: preferisci /documenti/ccnl.pdf (come da tuo progetto)
PDF_PATH_PRIMARY = os.path.join("documenti", "ccnl.pdf")
PDF_PATH_FALLBACK = "ccnl.pdf"

INDEX_DIR = "index_ccnl"
VEC_PATH = os.path.join(INDEX_DIR, "vectors.npy")
META_PATH = os.path.join(INDEX_DIR, "chunks.json")

# Retrieval
TOP_K_PER_QUERY = 12
TOP_K_FINAL = 18
MAX_MULTI_QUERIES = 12

# Chunking
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150

# Memoria breve
MEMORY_USER_TURNS = 3

# Debug (admin)
ADMIN_SESSION_KEY = "admin_ok"
DEBUG_LOG_KEY = "debug_log"

# Performance/cache
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

# Se vuoi ridurre costo/latency: abbassa TOP_K_FINAL / MAX_MULTI_QUERIES


# ============================================================
# SECRETS / ENV
# ============================================================
def get_secret(name: str, default=None):
    try:
        v = st.secrets.get(name, default)  # type: ignore
    except Exception:
        v = default
    if v is None:
        v = os.getenv(name, default)
    return v


UILCOM_PASSWORD = get_secret("UILCOM_PASSWORD", None)
ADMIN_PASSWORD = get_secret("ADMIN_PASSWORD", None)
OPENAI_API_KEY = get_secret("OPENAI_API_KEY", None)


# ============================================================
# STREAMLIT UI
# ============================================================
st.set_page_config(page_title="Assistente Contrattuale UILCOM IPZS", page_icon=APP_ICON)

st.title(APP_TITLE)
st.markdown(
    "**Accesso riservato agli iscritti UILCOM**  \n"
    "Strumento informativo per facilitare la consultazione del CCNL Grafici Editoria e norme applicabili ai lavoratori IPZS.  \n\n"
    "Le risposte sono generate **solo** sulla base del CCNL caricato.  \n"
    "Per casi specifici e interpretazioni: RSU/UILCOM o HR e testo ufficiale."
)
st.divider()


# ============================================================
# AUTH (ISCRITTI)
# ============================================================
if "auth_ok" not in st.session_state:
    st.session_state.auth_ok = False

if UILCOM_PASSWORD:
    with st.expander("🔒 Accesso iscritti UILCOM", expanded=not st.session_state.auth_ok):
        pwd_in = st.text_input("Password", type="password", placeholder="Inserisci la password iscritti")
        if st.button("Entra"):
            if pwd_in == UILCOM_PASSWORD:
                st.session_state.auth_ok = True
                st.success("Accesso consentito.")
            else:
                st.session_state.auth_ok = False
                st.error("Password non corretta.")
else:
    st.info("🔐 Password iscritti non impostata: imposta UILCOM_PASSWORD nei Secrets (online) o variabili d’ambiente (locale).")
    st.session_state.auth_ok = True  # per test locale

if not st.session_state.auth_ok:
    st.stop()


# ============================================================
# ADMIN DEBUG (invisibile agli utenti se non attivo)
# ============================================================
def debug_log(event: str, payload: Any = None):
    if DEBUG_LOG_KEY not in st.session_state:
        st.session_state[DEBUG_LOG_KEY] = []
    st.session_state[DEBUG_LOG_KEY].append(
        {
            "t": time.strftime("%H:%M:%S"),
            "event": event,
            "payload": payload,
        }
    )

if ADMIN_SESSION_KEY not in st.session_state:
    st.session_state[ADMIN_SESSION_KEY] = False


# ============================================================
# PDF PATH RESOLVE
# ============================================================
def resolve_pdf_path() -> str:
    if os.path.exists(PDF_PATH_PRIMARY):
        return PDF_PATH_PRIMARY
    if os.path.exists(PDF_PATH_FALLBACK):
        return PDF_PATH_FALLBACK
    return PDF_PATH_PRIMARY  # default, trigger error message later


# ============================================================
# EMBEDDINGS / LLM (OpenAI)
# ============================================================
def ensure_openai_key_or_stop():
    if not OPENAI_API_KEY:
        st.error(
            "Manca la variabile **OPENAI_API_KEY**.\n\n"
            "• In locale: imposta OPENAI_API_KEY nelle variabili d'ambiente.\n"
            "• Online (Streamlit Cloud): Settings → Secrets → OPENAI_API_KEY"
        )
        st.stop()


@st.cache_resource(show_spinner=False)
def get_embeddings():
    ensure_openai_key_or_stop()
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)


def get_llm():
    ensure_openai_key_or_stop()
    return ChatOpenAI(model=CHAT_MODEL, temperature=0)


# ============================================================
# INDEX BUILD/LOAD
# ============================================================
def build_index(pdf_path: str) -> int:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Non trovo il PDF: {pdf_path}. Metti il file in /documenti/ccnl.pdf")

    os.makedirs(INDEX_DIR, exist_ok=True)

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)

    texts = [c.page_content for c in chunks]
    # PyPDFLoader metadata page is 0-based: qui salviamo 1-based del FOGLIO PDF (non “pagina del CCNL”)
    pages_pdf = [(c.metadata.get("page", 0) + 1) for c in chunks]

    emb = get_embeddings()
    vectors = emb.embed_documents(texts)
    vectors = np.array(vectors, dtype=np.float32)

    np.save(VEC_PATH, vectors)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump([{"page_pdf": p, "text": t} for p, t in zip(pages_pdf, texts)], f, ensure_ascii=False)

    return len(chunks)


def load_index() -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    vectors = np.load(VEC_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # Hardening: meta deve essere lista di dict
    if not isinstance(meta, list):
        raise ValueError("chunks.json non valido: atteso una lista.")
    fixed = []
    for item in meta:
        if isinstance(item, dict):
            fixed.append(item)
        elif isinstance(item, str):
            fixed.append({"page_pdf": "?", "text": item})
        else:
            fixed.append({"page_pdf": "?", "text": str(item)})

    return vectors, fixed


def index_exists() -> bool:
    return os.path.exists(VEC_PATH) and os.path.exists(META_PATH)


# ============================================================
# RETRIEVAL HELPERS
# ============================================================
def normalize_rows(mat: np.ndarray) -> np.ndarray:
    return mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)


def cosine_scores(query_vec: np.ndarray, mat_norm: np.ndarray) -> np.ndarray:
    q = query_vec / (np.linalg.norm(query_vec) + 1e-12)
    return mat_norm @ q


def tokenize(text: str) -> List[str]:
    # tokenizer semplice e stabile per BM25
    return re.findall(r"[a-zàèéìòù0-9]+", (text or "").lower())


def bm25_rerank(query: str, candidates: List[Dict[str, Any]], top_n: int) -> List[Dict[str, Any]]:
    if not HAS_BM25 or not candidates:
        return candidates[:top_n]
    docs_tokens = [tokenize(c.get("text", "")) for c in candidates]
    bm25 = BM25Okapi(docs_tokens)
    scores = bm25.get_scores(tokenize(query))
    order = np.argsort(-np.array(scores))[:top_n]
    return [candidates[i] for i in order]


# ============================================================
# INTENTS / TRIGGERS
# ============================================================
CONSERVAZIONE_TRIGGERS = [
    "maternità", "maternita", "congedo maternità", "congedo maternita",
    "congedo parentale", "parentale",
    "malattia", "infortunio", "aspettativa",
    "assente", "assenza", "sostituzione", "sostituendo", "sto sostituendo",
]

MANSIONI_ALTE_TRIGGERS = [
    "mansioni più alte", "mansioni piu alte",
    "mansioni più elevate", "mansioni piu elevate",
    "mansioni superiori", "mansione superiore",
    "livello superiore", "categoria superiore", "inquadramento superiore",
    "passaggio di livello", "passaggio categoria",
    "posto vacante",
]

STRAORDINARI_TRIGGERS = [
    "straordin", "lavoro straordinario", "maggiorazione", "maggiorazioni",
    "notturno", "lavoro notturno", "straordinario notturno",
    "festivo", "straordinario festivo", "domenica", "turno di notte",
]

MALATTIA_TRIGGERS = [
    "malattia", "ammal", "certificat", "inps", "comporto",
    "visita fiscale", "reperibil", "fasce", "prognosi", "ricaduta",
]

PERMESSI_TRIGGERS = [
    "permess", "retribuit", "assenze retribuite",
    "visita medica", "lutto", "matrimonio", "studio", "esami", "150 ore",
    "104", "sindacal", "assemblea", "rsu", "donazione sangue",
    "rol", "ex festiv", "festività soppresse",
]

ROL_TRIGGERS = [
    "rol", "r.o.l", "riduzione orario", "ex festiv", "festività soppresse",
]


def has_any(q: str, triggers: List[str]) -> bool:
    ql = (q or "").lower()
    return any(t in ql for t in triggers)


def is_mansioni_superiori_question(q: str) -> bool:
    return has_any(q, MANSIONI_ALTE_TRIGGERS) or has_any(q, CONSERVAZIONE_TRIGGERS)


def is_malattia_question(q: str) -> bool:
    return has_any(q, MALATTIA_TRIGGERS)


def is_permessi_question(q: str) -> bool:
    return has_any(q, PERMESSI_TRIGGERS)


def is_rol_question(q: str) -> bool:
    return has_any(q, ROL_TRIGGERS)


def is_straordinari_question(q: str) -> bool:
    return has_any(q, STRAORDINARI_TRIGGERS)


# ============================================================
# MEMORY (ENRICH QUESTION)
# ============================================================
def build_enriched_question(current_q: str) -> str:
    if "messages" not in st.session_state:
        return current_q.strip()

    user_msgs = [m["content"] for m in st.session_state.messages if m.get("role") == "user" and m.get("content")]
    prev = user_msgs[:-1] if (user_msgs and user_msgs[-1].strip() == current_q.strip()) else user_msgs

    last = [x.strip() for x in (prev[-MEMORY_USER_TURNS:] if prev else []) if x.strip()]
    if not last:
        return current_q.strip()

    return (
        "CONTESTO CONVERSAZIONE (ultime richieste utente):\n"
        + "\n".join([f"- {x}" for x in last])
        + "\n\nDOMANDA ATTUALE:\n"
        + current_q.strip()
    )


# ============================================================
# MULTI-QUERY BUILDER
# ============================================================
def build_queries(user_q: str) -> List[str]:
    q0 = (user_q or "").strip()
    ql = q0.lower()

    qs = [q0, f"{q0} CCNL", f"{q0} regole", f"{q0} condizioni"]

    user_is_rol = is_rol_question(q0)
    user_is_perm = is_permessi_question(q0)
    user_is_mal = is_malattia_question(q0)
    user_is_mans = is_mansioni_superiori_question(q0)
    user_is_strao = is_straordinari_question(q0)

    if user_is_rol:
        qs += [
            "ROL riduzione orario lavoro monte ore maturazione fruizione",
            "ex festività festività soppresse permessi ore giorni spettanti",
            "ROL ex festività richiesta fruizione preavviso",
        ]
    elif user_is_perm:
        qs += [
            "permessi retribuiti tipologie elenco",
            "assenze retribuite visite mediche lutto matrimonio studio esami 150 ore 104 sindacali assemblea donazione sangue",
            "diritto allo studio 150 ore triennio",
            "permessi visite mediche giustificativo",
            "permessi lutto giorni familiari",
            "permessi matrimonio congedo matrimoniale",
        ]

    if user_is_mal:
        qs += [
            "malattia trattamento economico integrazione percentuali",
            "malattia comporto periodo conservazione posto conteggio",
            "malattia obblighi comunicazione certificato",
            "visite fiscali reperibilità fasce",
        ]

    if user_is_mans:
        qs += [
            "mansioni superiori 30 giorni continuativi 60 giorni non continuativi",
            "mansioni superiori non si applica sostituzione diritto conservazione del posto",
            "affiancamento formazione addestramento non costituisce mansioni superiori",
            "posto vacante mansioni superiori",
        ]

    # FIX straordinari/notturno: query mirate per evitare confusione "notturno" vs "straordinario notturno"
    if user_is_strao:
        qs += [
            "lavoro notturno maggiorazione",
            "straordinario notturno maggiorazione",
            "lavoro straordinario maggiorazioni",
            "lavoro festivo maggiorazioni",
        ]

    # de-dup
    out, seen = [], set()
    for x in qs:
        x = (x or "").strip()
        if x and x not in seen:
            out.append(x)
            seen.add(x)
    return out[:MAX_MULTI_QUERIES]


# ============================================================
# EVIDENCE EXTRACTOR (per guardrail interni)
# (non mostrato agli utenti, solo admin)
# ============================================================
def extract_key_evidence(chunks: List[Dict[str, Any]]) -> List[str]:
    evidences = []
    patterns = [
        r"\b30\b", r"\b60\b", r"\b%\b",
        r"mansioni?\s+superiori?", r"sostituzion", r"conservazion.*posto",
        r"non\s+si\s+applica", r"non\s+costituisc",
        r"straordin", r"notturn", r"festiv",
        r"malatt", r"comporto", r"certificat", r"reperibil",
        r"permess", r"\brol\b", r"ex\s*fest", r"lutto", r"matrimon", r"\b104\b",
    ]

    def interesting(ln: str) -> bool:
        ln_low = ln.lower()
        return any(re.search(p, ln_low) for p in patterns)

    for c in chunks:
        text = (c.get("text", "") or "")
        page_pdf = c.get("page_pdf", "?")
        for ln in [x.strip() for x in text.splitlines() if x.strip()]:
            if interesting(ln):
                ln_clean = " ".join(ln.split())
                if 20 <= len(ln_clean) <= 420:
                    evidences.append(f"(foglio PDF {page_pdf}) {ln_clean}")

    # unique keep order
    out, seen = [], set()
    for e in evidences:
        if e not in seen:
            out.append(e)
            seen.add(e)
    return out[:20]


def evidence_has_30_60(evidence_lines: List[str]) -> bool:
    joined = " ".join(evidence_lines).lower()
    return (re.search(r"\b30\b", joined) is not None) and (re.search(r"\b60\b", joined) is not None)


# ============================================================
# RULES / GUARDRAILS (massima precisione)
# ============================================================
SYSTEM_RULES = (
    "Sei l’assistente UILCOM per lavoratori IPZS. "
    "Rispondi in modo chiaro, professionale e pratico basandoti SOLO sul contesto del CCNL fornito. "
    "Non inventare informazioni e non usare conoscenze esterne.\n\n"

    "REGOLA PROVA: puoi affermare un numero/percentuale/durata SOLO se nel contesto recuperato "
    "c'è una frase che lo supporta chiaramente. Se non c’è, dì che non emerge dal CCNL recuperato.\n\n"

    "REGOLA CHIAVE MANSIONI SUPERIORI: se nel contesto sono presenti soglie 30 giorni continuativi / 60 non continuativi, "
    "questi valori hanno priorità e vanno riportati. "
    "Se la situazione è una sostituzione di assente con diritto alla conservazione del posto (es. maternità/malattia), "
    "specifica che la regola di consolidamento non si applica ai fini dell’inquadramento definitivo, se risulta dal testo.\n\n"

    "FIX NOTTURNO vs STRAORDINARIO NOTTURNO: non confondere 'lavoro notturno' con 'straordinario notturno'. "
    "Se la percentuale 60% è citata nel contesto per lo STRAORDINARIO notturno, non usarla per il semplice lavoro notturno. "
    "Se il CCNL recuperato non indica chiaramente la maggiorazione del lavoro notturno ordinario, dillo esplicitamente.\n\n"

    "FORMATO RISPOSTA: scrivi una risposta breve e chiara. "
    "NON mostrare pagine o riferimenti all’utente finale. "
    "Chiudi sempre con: 'Nota UILCOM: Questa risposta è informativa; per casi specifici verificare con RSU/UILCOM o HR e con il testo ufficiale.'\n\n"

    "CONSIGLIO PRATICO: aggiungi 1–2 bullet operativi coerenti con la domanda (es. ordine di servizio, verifica sostituzione/posto vacante, RSU/HR). "
)


# ============================================================
# SIDEBAR (CONTROLLI + ADMIN)
# ============================================================
with st.sidebar:
    st.header("⚙️ Controlli")

    # Admin login (solo se ADMIN_PASSWORD impostata)
    st.subheader("👤 Admin (debug)")
    if ADMIN_PASSWORD:
        admin_in = st.text_input("Password admin", type="password", placeholder="Solo admin")
        colA, colB = st.columns(2)
        with colA:
            if st.button("Login admin"):
                st.session_state[ADMIN_SESSION_KEY] = (admin_in == ADMIN_PASSWORD)
                if st.session_state[ADMIN_SESSION_KEY]:
                    st.success("Admin attivo.")
                else:
                    st.error("Password admin errata.")
        with colB:
            if st.button("Logout admin"):
                st.session_state[ADMIN_SESSION_KEY] = False
                st.info("Admin disattivato.")
    else:
        st.caption("ADMIN_PASSWORD non impostata (consigliato in Secrets).")

    st.divider()

    st.caption("Indice CCNL (serve solo la prima volta o se cambi PDF).")
    if st.button("Indicizza CCNL (prima volta / aggiorna)"):
        try:
            pdf_path = resolve_pdf_path()
            with st.spinner("Indicizzazione in corso..."):
                n = build_index(pdf_path)
            st.success(f"Indicizzazione completata! Chunk creati: {n}")
        except Exception as e:
            st.error(str(e))

    st.write("📦 Indice presente:", "✅" if index_exists() else "❌")

    if st.button("🧹 Nuova chat"):
        st.session_state.messages = []
        st.rerun()

    if st.session_state.get(ADMIN_SESSION_KEY):
        st.divider()
        st.subheader("🧪 Debug console (admin)")
        if st.button("Pulisci debug log"):
            st.session_state[DEBUG_LOG_KEY] = []
        if st.session_state.get(DEBUG_LOG_KEY):
            st.json(st.session_state[DEBUG_LOG_KEY][-20:])


# ============================================================
# CHAT UI
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostra storico chat (pulito stile ChatGPT: solo testo)
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_input = st.chat_input("Scrivi una domanda sul CCNL (ferie, malattia, permessi, straordinari, livelli...)")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    if not index_exists():
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Prima devo indicizzare il CCNL: apri il menu a sinistra e clicca **Indicizza CCNL**."
        })
        st.rerun()

    # Load
    vectors, meta = load_index()
    mat_norm = normalize_rows(vectors)
    emb = get_embeddings()
    llm = get_llm()

    enriched_q = build_enriched_question(user_input)
    queries = build_queries(enriched_q)

    user_is_mans = is_mansioni_superiori_question(enriched_q)
    user_is_strao = is_straordinari_question(enriched_q)

    debug_log("user_input", user_input)
    debug_log("enriched_q", enriched_q)
    debug_log("queries", queries)

    # --- Pass 1: embedding retrieval multi-query
    scores_best: Dict[int, float] = {}
    for q in queries:
        qvec = np.array(emb.embed_query(q), dtype=np.float32)
        sims = cosine_scores(qvec, mat_norm)
        top_idx = np.argsort(-sims)[:TOP_K_PER_QUERY]
        for i in top_idx:
            s = float(sims[i])
            if (i not in scores_best) or (s > scores_best[i]):
                scores_best[i] = s

    provisional_idx = sorted(scores_best.keys(), key=lambda i: scores_best[i], reverse=True)[:TOP_K_FINAL * 2]
    provisional_selected = [meta[i] for i in provisional_idx]

    # --- Optional: BM25 rerank su candidati (migliora precisione su numeri/percentuali)
    reranked = bm25_rerank(enriched_q, provisional_selected, TOP_K_FINAL)

    # --- Guardrail interno: se mansioni, cerca 30/60
    evidence_lines = extract_key_evidence(reranked)
    has_30_60 = evidence_has_30_60(evidence_lines) if user_is_mans else False

    debug_log("has_bm25", HAS_BM25)
    debug_log("has_30_60", has_30_60)
    if st.session_state.get(ADMIN_SESSION_KEY):
        debug_log("evidence", evidence_lines)

    # Context string
    context_blocks = []
    for c in reranked:
        # NON mostrare pagine all’utente; le mettiamo solo internamente nel contesto al modello
        context_blocks.append(f"[Foglio PDF {c.get('page_pdf','?')}] {c.get('text','')}")
    context = "\n\n---\n\n".join(context_blocks)

    # --- Prompt con fix notturno/straordinario
    # Se l’utente chiede "lavoro notturno", forziamo il modello a NON usare 60% se non è esplicito
    extra_fix = ""
    if user_is_strao:
        extra_fix = (
            "\nATTENZIONE STRAORDINARI/NOTTURNO: "
            "se nel contesto è presente la maggiorazione del 60%, verifica che sia riferita allo STRAORDINARIO notturno. "
            "Se la domanda è sul lavoro notturno ordinario e il contesto non lo specifica, NON usare 60% e dichiara che non emerge.\n"
        )

    # Se mansioni e trovi 30/60, imponi priorità
    mans_fix = ""
    if user_is_mans and has_30_60:
        mans_fix = (
            "\nMANSIONI SUPERIORI: nel contesto risultano 30 giorni continuativi e 60 non continuativi. "
            "Devi riportare questi valori in modo esplicito (posto vacante vs sostituzione con conservazione del posto).\n"
        )

    prompt = f"""
{SYSTEM_RULES}
{extra_fix}
{mans_fix}

DOMANDA UTENTE:
{user_input}

CONTESTO CCNL (estratti):
{context}

ISTRUZIONI RISPOSTA:
- Risposta UILCOM: 2-5 righe
- Dettagli: 4-10 punti (solo ciò che è supportato nel contesto)
- Consiglio pratico UILCOM: 1-2 bullet
- Nota UILCOM: (frase standard)

IMPORTANTE:
- Non mostrare pagine o numeri di foglio all’utente.
- Non inventare percentuali o durate: se non emergono dal contesto, dillo.
RISPOSTA:
"""

    try:
        answer = llm.invoke(prompt).content
    except Exception as e:
        answer = f"Errore nel generare la risposta: {e}"

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()
