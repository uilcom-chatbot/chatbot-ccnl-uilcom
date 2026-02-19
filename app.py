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

# Optional: BM25 rerank (precision boost)
try:
    from rank_bm25 import BM25Okapi  # type: ignore
    HAS_BM25 = True
except Exception:
    HAS_BM25 = False


# ============================================================
# UI / CONFIG
# ============================================================
st.set_page_config(page_title="Assistente Contrattuale UILCOM IPZS", page_icon="🟦", layout="centered")

APP_TITLE = "🟦 Assistente Contrattuale UILCOM IPZS"
st.title(APP_TITLE)
st.markdown(
    "**Accesso riservato agli iscritti UILCOM**  \n"
    "Strumento informativo per facilitare la consultazione del CCNL Grafici Editoria e norme applicabili ai lavoratori IPZS.  \n"
    "Le risposte sono generate **solo** sulla base del CCNL caricato. Per casi specifici rivolgersi a RSU/UILCOM o HR."
)
st.divider()

# Paths (repository)
PDF_PATH = os.path.join("documenti", "ccnl.pdf")  # assicurati che il PDF sia in /documenti/ccnl.pdf
INDEX_DIR = "index_ccnl"
VEC_PATH = os.path.join(INDEX_DIR, "vectors.npy")
META_PATH = os.path.join(INDEX_DIR, "chunks.json")

# Retrieval tuning
TOP_K_FINAL = 18
TOP_K_PER_QUERY = 12
MAX_MULTI_QUERIES = 14

# Short memory
MEMORY_USER_TURNS = 3

# Permessi
PERMESSI_MIN_CATEGORY_COVERAGE = 3

# Night work guardrail
NOTTURNO_PENALTY_STRAO = 0.18
NOTTURNO_BOOST_TURNI = 0.10

# Debug
DEBUG_DEFAULT = False


# ============================================================
# SECRETS / ENV HELPERS
# ============================================================
def get_secret(name: str) -> Optional[str]:
    try:
        v = st.secrets.get(name, None)  # type: ignore
    except Exception:
        v = None
    if not v:
        v = os.getenv(name)
    return v


UILCOM_PASSWORD = get_secret("UILCOM_PASSWORD")  # password iscritti
ADMIN_PASSWORD = get_secret("ADMIN_PASSWORD")    # password admin debug (facoltativa)
OPENAI_API_KEY = get_secret("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error(
        "Manca la variabile **OPENAI_API_KEY**.\n\n"
        "- In locale: imposta OPENAI_API_KEY nelle variabili d'ambiente.\n"
        "- Online (Streamlit Cloud): **Settings → Secrets** → aggiungi OPENAI_API_KEY."
    )
    st.stop()

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


# ============================================================
# AUTH (Iscritti)
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
    st.info("🔐 Password iscritti non impostata. (Per produzione, imposta UILCOM_PASSWORD nei Secrets.)")
    st.session_state.auth_ok = True

if not st.session_state.auth_ok:
    st.stop()


# ============================================================
# ADMIN DEBUG (facoltativo e invisibile ai non-admin)
# ============================================================
if "admin_ok" not in st.session_state:
    st.session_state.admin_ok = False

def _admin_login_ui():
    if not ADMIN_PASSWORD:
        return
    with st.expander("🛠️ Admin (debug)", expanded=False):
        p = st.text_input("Password admin", type="password", placeholder="Solo amministratori")
        if st.button("Sblocca debug"):
            st.session_state.admin_ok = (p == ADMIN_PASSWORD)
            if st.session_state.admin_ok:
                st.success("Debug attivo (solo per te).")
            else:
                st.error("Password admin errata.")

_admin_login_ui()


# ============================================================
# RETRIEVAL HELPERS
# ============================================================
def normalize_rows(mat: np.ndarray) -> np.ndarray:
    return mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)

def cosine_scores(query_vec: np.ndarray, mat_norm: np.ndarray) -> np.ndarray:
    q = query_vec / (np.linalg.norm(query_vec) + 1e-12)
    return mat_norm @ q

def ensure_meta_dicts(meta: Any) -> List[Dict[str, Any]]:
    """Hardening: evita errori 'str has no attribute get'."""
    if isinstance(meta, list) and meta and isinstance(meta[0], dict):
        return meta
    # Prova a convertire (caso raro)
    fixed: List[Dict[str, Any]] = []
    if isinstance(meta, list):
        for item in meta:
            if isinstance(item, dict):
                fixed.append(item)
            elif isinstance(item, str):
                fixed.append({"page": "?", "text": item})
            else:
                fixed.append({"page": "?", "text": str(item)})
    else:
        fixed = [{"page": "?", "text": str(meta)}]
    return fixed


# ============================================================
# TRIGGERS / CLASSIFIERS
# ============================================================
CONSERVAZIONE_TRIGGERS = [
    "maternità", "maternita", "congedo maternità", "congedo maternita",
    "congedo parentale", "parentale",
    "malattia", "infortunio",
    "aspettativa",
    "assente", "assenza", "sostituzione", "sostituendo", "sto sostituendo",
]

MANSIONI_ALTE_TRIGGERS = [
    "mansioni più alte", "mansioni piu alte",
    "mansioni più elevate", "mansioni piu elevate",
    "mansioni superiori", "mansioni superiore", "mansione superiore",
    "sto facendo il lavoro", "mi fanno fare il lavoro", "mi stanno facendo fare",
    "faccio mansioni", "faccio il lavoro di",
    "capoturno", "capo turno", "caporeparto", "capo reparto",
    "sostituisco", "sto sostituendo",
    "livello superiore", "categoria superiore", "inquadramento superiore",
    "passaggio di livello", "passaggio categoria", "avanzamento", "promozione",
    "posto vacante",
]

MALATTIA_TRIGGERS = [
    "malattia", "ammal", "certificat", "certificato", "inps",
    "comporto", "prognosi", "ricaduta",
    "visita fiscale", "controllo", "reperibil", "fasce",
    "assenza per malattia", "indennità", "indennita", "trattamento economico",
    "ospedal", "ricovero", "day hospital",
    "infortunio", "infortun",
    "malattia durante ferie", "mi ammalo in ferie",
]

PERMESSI_TRIGGERS = [
    "permess", "permesso", "permessi", "retribuit", "assenze retribuite",
    "visita medica", "visite mediche", "medico", "specialista",
    "lutto", "matrimonio", "nozze",
    "studio", "formazione", "esami",
    "104", "legge 104", "handicap",
    "donazione sangue", "donazione",
    "sindacal", "assemblea", "rsu",
    "rol", "ex festiv", "ex-festiv", "exfestiv", "festivit", "festività",
]

ROL_TRIGGERS = [
    "rol", "r.o.l", "riduzione orario", "riduzione dell'orario", "riduzione orario di lavoro",
    "ex festiv", "ex-festiv", "exfestiv", "ex festività", "ex-festività", "ex festivita",
    "festività soppresse", "festivita soppresse", "festività abolite", "festivita abolite",
    "permessi rol", "ore rol", "giorni rol",
    "quanti rol", "quante ore rol", "quanto rol",
    "quante ex festività", "quante ex festivita", "quante festività soppresse",
]

IPZS_TRIGGERS = [
    "ipzs", "poligrafico", "zecca", "azienda", "accordo aziendale", "accordi aziendali",
    "ordine di servizio", "ods", "turni", "reparto", "linea", "impianto",
]

# ✅ NOTTURNO vs STRAORDINARIO (fix 60%)
NOTTURNO_TRIGGERS = [
    "lavoro notturno", "turno notturno", "notturno", "notte",
    "maggiorazione notturna", "indennità notturna", "indennita notturna",
]
STRAORDINARIO_TRIGGERS = [
    "straordinario", "straordinari", "lavoro straordinario",
    "straordinario notturno", "straordinario di notte", "ore extra",
]

def is_mansioni_superiori_question(user_q: str) -> bool:
    ql = user_q.lower()
    return any(t in ql for t in MANSIONI_ALTE_TRIGGERS)

def is_malattia_question(user_q: str) -> bool:
    ql = user_q.lower()
    return any(t in ql for t in MALATTIA_TRIGGERS)

def is_permessi_question(user_q: str) -> bool:
    ql = user_q.lower()
    return any(t in ql for t in PERMESSI_TRIGGERS)

def is_rol_question(user_q: str) -> bool:
    ql = user_q.lower()
    return any(t in ql for t in ROL_TRIGGERS)

def is_ipzs_context(user_q: str) -> bool:
    ql = user_q.lower()
    return any(t in ql for t in IPZS_TRIGGERS)

def is_notturno_question(user_q: str) -> bool:
    ql = user_q.lower()
    return any(t in ql for t in NOTTURNO_TRIGGERS)

def is_straordinario_question(user_q: str) -> bool:
    ql = user_q.lower()
    return any(t in ql for t in STRAORDINARIO_TRIGGERS)

def is_notturno_ordinario(user_q: str) -> bool:
    return is_notturno_question(user_q) and (not is_straordinario_question(user_q))


# ============================================================
# PERMESSI CATEGORIES
# ============================================================
PERMESSI_CATEGORIES = {
    "visite_mediche": [r"visite?\s+med", r"visita\s+med", r"accertament", r"specialist", r"struttur[ae]\s+sanitar"],
    "lutto": [r"\blutto\b", r"decesso", r"grave\s+lutto", r"familiare"],
    "matrimonio": [r"matrimon", r"nozz"],
    "studio_formazione": [r"diritto\s+allo\s+studio", r"\b150\s+ore\b", r"\bstudio\b", r"\besami\b", r"formazion"],
    "legge_104": [r"\b104\b", r"legge\s*104", r"handicap"],
    "sindacali": [r"sindacal", r"\brsu\b", r"assemblea", r"permessi?\s+sindacal"],
    "donazione_sangue": [r"donazion", r"sangue", r"emocomponent"],
    "altri_permessi": [r"permessi?\s+retribuit", r"assenze?\s+retribuit", r"conged"],
}

def permessi_category_coverage(selected_chunks: List[Dict[str, Any]]) -> Tuple[int, List[str]]:
    found = set()
    joined = " ".join([(c.get("text", "") or "") for c in selected_chunks]).lower()
    for cat, pats in PERMESSI_CATEGORIES.items():
        for p in pats:
            if re.search(p, joined, flags=re.IGNORECASE):
                found.add(cat)
                break
    return len(found), sorted(found)

def build_permessi_expansion_queries(user_q: str) -> List[str]:
    base = user_q.strip()
    return [
        f"{base} permessi visite mediche retribuiti",
        f"{base} permessi lutto retribuiti",
        f"{base} permessi matrimonio retribuiti",
        f"{base} permessi legge 104 retribuiti",
        f"{base} permessi sindacali assemblea RSU",
        f"{base} permessi diritto allo studio 150 ore",
        f"{base} permessi donazione sangue",
        f"{base} assenze retribuite elenco tipologie",
    ][:MAX_MULTI_QUERIES]


# ============================================================
# MEMORY (short)
# ============================================================
def build_enriched_question(current_q: str) -> str:
    if "messages" not in st.session_state:
        return current_q.strip()

    user_msgs = [m["content"] for m in st.session_state.messages if m.get("role") == "user" and m.get("content")]
    if user_msgs and user_msgs[-1].strip() == current_q.strip():
        prev = user_msgs[:-1]
    else:
        prev = user_msgs

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

    user_is_rol = is_rol_question(q0)
    user_is_perm = is_permessi_question(q0)
    user_is_mal = is_malattia_question(q0)
    user_is_mans = is_mansioni_superiori_question(q0)
    user_is_conserv = any(t in qlow for t in CONSERVAZIONE_TRIGGERS)

    user_is_notturno_ord = is_notturno_ordinario(q0)
    user_is_strao = is_straordinario_question(q0)

    # ROL / ex festività
    if user_is_rol:
        qs += [
            "ROL riduzione orario di lavoro monte ore annuo maturazione fruizione",
            "ex festività festività soppresse permessi ore giorni spettanti",
            "permessi ROL ed ex festività: quanti, come maturano e come si usano",
            "ROL ex festività richiesta fruizione preavviso eventuale programmazione",
            "ROL ex festività residui scadenze eventuale monetizzazione (se prevista)",
        ]

    # Permessi generici (non ROL)
    elif user_is_perm:
        qs += [
            "permessi retribuiti tipologie CCNL elenco completo",
            "assenze retribuite tipologie (visite mediche, lutto, matrimonio, 104, sindacali, studio, donazione sangue)",
            "permessi per visite mediche come funzionano",
            "permessi per lutto matrimonio",
            "permessi legge 104",
            "permessi sindacali assemblea ore retribuite",
            "permessi per studio esami 150 ore triennio",
            "permessi donazione sangue",
        ]

    # Malattia
    if user_is_mal:
        qs += [
            "malattia trattamento economico percentuali integrazione",
            "malattia periodo di comporto regole conteggio",
            "malattia obblighi comunicazione certificazione",
            "controlli visite fiscali reperibilità fasce",
            "malattia durante ferie sospensione ferie se certificata",
            "ricovero ospedaliero day hospital ricaduta",
        ]

    # Ferie
    if any(k in qlow for k in ["ferie", "residu", "matur", "programmaz", "chiusura"]):
        qs += [
            "ferie giorni spettanti maturazione fruizione frazionamento",
            "residui ferie termini regole",
            "malattia durante ferie cosa succede",
        ]

    # Straordinari (generico)
    if any(k in qlow for k in ["straordin", "maggior", "banca ore"]):
        qs += [
            "lavoro straordinario maggiorazioni limiti autorizzazione",
            "straordinario notturno festivo maggiorazioni",
            "banca ore recuperi straordinario se previsti",
        ]

    # ✅ NOTTURNO ORDINARIO (fix 60%)
    if user_is_notturno_ord:
        qs += [
            "lavoro notturno ordinario maggiorazione",
            "turno notturno indennità notturna",
            "lavoro a turni notturni maggiorazioni",
            "maggiorazione per lavoro notturno (non straordinario)",
        ]

    # ✅ STRAORDINARIO NOTTURNO (se richiesto)
    if user_is_strao and ("notturn" in qlow):
        qs += [
            "straordinario notturno maggiorazione percentuale",
            "lavoro straordinario di notte percentuali",
        ]

    # Trasferimenti / trasferte
    if any(k in qlow for k in ["trasfer", "trasferta", "spostamento", "mobilità", "sede"]):
        qs += [
            "trasferimento regole preavviso tutele",
            "trasferta indennità rimborsi spese",
        ]

    # Livelli / inquadramento
    if any(k in qlow for k in ["livell", "inquadr", "qualifica", "categoria", "passaggio"]):
        qs += [
            "classificazione livelli inquadramento criteri",
            "passaggi di livello avanzamenti regole",
        ]

    # Mansioni superiori / sostituzioni
    if user_is_mans or user_is_conserv:
        qs += [
            "mansioni superiori posto vacante",
            "assegnazione a mansioni superiori regole generali",
            "mansioni superiori 30 giorni consecutivi 60 discontinui",
            "non si applica in caso di sostituzione di dipendente assente con diritto alla conservazione del posto",
            "formazione addestramento affiancamento non costituisce mansioni superiori",
            "trattamento economico durante mansioni superiori",
        ]

    # Dedup
    out, seen = [], set()
    for x in qs:
        x = x.strip()
        if x and x not in seen:
            out.append(x)
            seen.add(x)
    return out[:MAX_MULTI_QUERIES]


# ============================================================
# EVIDENCE EXTRACTION (used only for internal guardrails / admin)
# ============================================================
def extract_key_evidence(chunks: List[Dict[str, Any]]) -> List[str]:
    evidences: List[str] = []
    patterns = [
        r"\b30\b", r"\b60\b", r"\b\d{1,3}\b", r"\b%\b",
        r"posto\s+vacante", r"mansioni?\s+superiori?", r"sostituzion",
        r"conservazion.*posto", r"diritto.*conservazion.*posto",
        r"non\s+si\s+applica", r"non\s+costituisc",
        r"formazion", r"addestrament", r"affiancament",
        r"malatt", r"comporto", r"certificat", r"reperibil", r"visita\s+fiscale",
        r"permess", r"\brol\b", r"riduzione\s+orario", r"ex\s*fest", r"festivit",
        r"lutto", r"matrimon", r"nozz", r"\b104\b", r"sindacal", r"assemblea",
        r"\b3\s*mesi\b", r"tre\s+mesi",
        r"notturn", r"turno", r"straordin",
    ]

    def line_is_interesting(ln_low: str) -> bool:
        return any(re.search(p, ln_low) for p in patterns)

    for c in chunks:
        page = c.get("page", "?")
        text = c.get("text", "") or ""
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        for ln in lines:
            ln_low = ln.lower()
            if line_is_interesting(ln_low):
                ln_clean = " ".join(ln.split())
                if 20 <= len(ln_clean) <= 420:
                    evidences.append(f"(pag. {page}) {ln_clean}")

    out, seen = [], set()
    for e in evidences:
        if e not in seen:
            out.append(e)
            seen.add(e)
    return out[:18]

def evidence_has_30_60(evidence_lines: List[str]) -> bool:
    joined = " ".join(evidence_lines).lower()
    return (re.search(r"\b30\b", joined) is not None) and (re.search(r"\b60\b", joined) is not None)

def text_mentions_three_months(txt: str) -> bool:
    t = (txt or "").lower()
    return ("tre mesi" in t) or (re.search(r"\b3\s*mesi\b", t) is not None)


# ============================================================
# INDEX BUILD / LOAD
# ============================================================
def build_index() -> int:
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"Non trovo il PDF: {PDF_PATH} (metti 'ccnl.pdf' in /documenti)")

    os.makedirs(INDEX_DIR, exist_ok=True)

    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    texts = [c.page_content for c in chunks]
    pages = [(c.metadata.get("page", 0) + 1) for c in chunks]  # numero foglio PDF (interno)

    emb = OpenAIEmbeddings()
    vectors = np.array(emb.embed_documents(texts), dtype=np.float32)

    np.save(VEC_PATH, vectors)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump([{"page": p, "text": t} for p, t in zip(pages, texts)], f, ensure_ascii=False)

    return len(chunks)

def load_index() -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    vectors = np.load(VEC_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    meta = ensure_meta_dicts(meta)
    return vectors, meta


# ============================================================
# RULES / PROMPT (anti-invenzioni + guardrails)
# ============================================================
rules = (
    "Sei l’assistente UILCOM per lavoratori IPZS. "
    "Rispondi in modo chiaro e professionale basandoti SOLO sul contesto del CCNL fornito. "
    "Non inventare informazioni.\n\n"

    "REGOLA PROVA: puoi elencare una regola/voce SOLO se nel contesto recuperato c'è un riferimento testuale che la supporta. "
    "Se non emerge dal CCNL recuperato, devi dirlo esplicitamente.\n\n"

    "REGOLA CITAZIONE MIRATA (INTERNA): se affermi numeri/percentuali/durate o esclusioni (es. 'non si applica'), "
    "devi averli trovati nel contesto. Se non trovi la frase che lo dimostra, NON affermarlo.\n\n"

    "OBBLIGO LIMITAZIONI: se nel contesto compaiono frasi di esclusione/limitazione "
    "(es. 'non si applica', 'non costituisce', 'sostituzione', 'diritto alla conservazione del posto', "
    "'affiancamento/formazione'), riportale.\n\n"

    "MODULO MALATTIA: quando la domanda riguarda la malattia, se presenti nel contesto includi: "
    "(1) trattamento economico/percentuali, (2) comporto, (3) obblighi certificazione/comunicazione, "
    "(4) controlli/reperibilità, (5) casi particolari (ricovero/ricaduta/malattia durante ferie). "
    "Se una voce non è nel contesto, scrivi che non emerge.\n\n"

    "SUPER MODULO PERMESSI: quando la domanda riguarda 'permessi retribuiti' in generale (oltre a ROL/ex festività), "
    "cerca e riassumi più categorie. Inserisci SOLO categorie supportate dal contesto recuperato. "
    "Se trovi una sola categoria, dillo.\n\n"

    "FIX ROL/EX FESTIVITÀ: se l’utente chiede ROL o ex festività, rispondi SU ROL/ex festività e non confondere con studio.\n\n"

    "REGOLA CHIAVE (SOSTITUZIONI): se la domanda riguarda sostituzione di lavoratore assente con diritto alla conservazione del posto "
    "(es. maternità, malattia, infortunio), specifica che la regola delle mansioni superiori può non applicarsi ai fini "
    "dell’inquadramento definitivo se così risulta dal contesto.\n\n"

    "PRIORITÀ MANSIONI SUPERIORI: se nel contesto sono presenti 30 giorni consecutivi / 60 discontinui, "
    "questi valori hanno priorità.\n\n"

    # ✅ GUARDRAIL NOTTURNO (fix 60%)
    "GUARDRAIL NOTTURNO: non confondere 'lavoro notturno ordinario' (turno notturno) con 'straordinario notturno'. "
    "Se nel contesto trovi una percentuale (es. 60%) associata a 'straordinario notturno' o 'lavoro straordinario', "
    "NON attribuirla al lavoro notturno ordinario. "
    "Se il CCNL caricato non riporta chiaramente la maggiorazione del notturno ordinario, devi dirlo: "
    "'Non emerge dal CCNL recuperato la percentuale del notturno ordinario'.\n\n"

    "CONSIGLIO PRATICO: chiudi sempre con 1–2 bullet operativi coerenti con la domanda "
    "(ordine di servizio, posto vacante vs sostituzione, RSU/UILCOM, HR, accordi IPZS se rilevanti).\n"
)


# ============================================================
# SIDEBAR (controlli + indicizzazione)
# ============================================================
with st.sidebar:
    st.header("⚙️ Controlli")

    ok_index = os.path.exists(VEC_PATH) and os.path.exists(META_PATH)
    st.write("📦 Indice presente:", "✅" if ok_index else "❌")

    if st.button("Indicizza CCNL (prima volta / dopo cambio PDF)"):
        try:
            with st.spinner("Indicizzazione in corso..."):
                n = build_index()
            st.success(f"Indicizzazione completata. Chunk creati: {n}")
        except Exception as e:
            st.error(str(e))

    if st.button("🧹 Nuova chat"):
        st.session_state.messages = []
        st.rerun()

    # Debug toggle only if admin_ok
    if st.session_state.admin_ok:
        st.divider()
        st.subheader("🛠️ Debug (admin)")
        st.session_state.debug_on = st.toggle("Mostra fonti/estratti", value=st.session_state.get("debug_on", DEBUG_DEFAULT))
    else:
        st.session_state.debug_on = False


# ============================================================
# CHAT STATE
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render chat
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        # fonti SOLO per admin con debug attivo
        if m["role"] == "assistant" and st.session_state.debug_on and m.get("sources"):
            with st.expander("📚 Debug: estratti CCNL usati"):
                for s in m["sources"]:
                    st.write(f"**Foglio PDF {s.get('page','?')}**")
                    t = s.get("text", "") or ""
                    st.write(t[:900] + ("..." if len(t) > 900 else ""))
                    st.divider()


user_input = st.chat_input("Scrivi una domanda sul CCNL (mansioni, permessi, malattia, ferie, notturno, straordinari...)")

if not user_input:
    st.stop()

st.session_state.messages.append({"role": "user", "content": user_input})

# Ensure index
if not (os.path.exists(VEC_PATH) and os.path.exists(META_PATH)):
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Prima devo indicizzare il CCNL: apri il menu a sinistra e clicca **Indicizza CCNL**.",
        "sources": []
    })
    st.rerun()


# ============================================================
# RETRIEVAL PIPELINE
# ============================================================
enriched_q = build_enriched_question(user_input)

vectors, meta = load_index()
mat_norm = normalize_rows(vectors)
emb = OpenAIEmbeddings()

user_is_mans = is_mansioni_superiori_question(enriched_q)
user_is_mal = is_malattia_question(enriched_q)
user_is_perm = is_permessi_question(enriched_q)
user_is_rol = is_rol_question(enriched_q)
user_mentions_ipzs = is_ipzs_context(enriched_q)
user_is_notturno_ord = is_notturno_ordinario(enriched_q)
user_is_strao = is_straordinario_question(enriched_q)

queries = build_queries(enriched_q)

scores_best: Dict[int, float] = {}
for q in queries:
    qvec = np.array(emb.embed_query(q), dtype=np.float32)
    sims = cosine_scores(qvec, mat_norm)
    top_idx = np.argsort(-sims)[:TOP_K_PER_QUERY]
    for i in top_idx:
        s = float(sims[i])
        if (i not in scores_best) or (s > scores_best[i]):
            scores_best[i] = s

provisional_idx = sorted(scores_best.keys(), key=lambda i: scores_best[i], reverse=True)[:TOP_K_FINAL]
provisional_selected = [meta[i] for i in provisional_idx]
provisional_evidence = extract_key_evidence(provisional_selected)

# Super pass 2 for permessi generic
perm_guardrail_note = ""
if user_is_perm and (not user_is_rol):
    cov_n, _ = permessi_category_coverage(provisional_selected)
    if cov_n < PERMESSI_MIN_CATEGORY_COVERAGE:
        extra_queries = build_permessi_expansion_queries(enriched_q)
        for q in extra_queries:
            qvec = np.array(emb.embed_query(q), dtype=np.float32)
            sims = cosine_scores(qvec, mat_norm)
            top_idx = np.argsort(-sims)[:TOP_K_PER_QUERY]
            for i in top_idx:
                s = float(sims[i])
                if (i not in scores_best) or (s > scores_best[i]):
                    scores_best[i] = s
        perm_guardrail_note = (
            "GUARDRAIL PERMESSI: includi SOLO categorie supportate dal contesto recuperato. "
            "Se una categoria non emerge dal CCNL caricato, dichiaralo."
        )

# Re-ranking with boosts + NOTTURNO penalty/boost (fix 60%)
guardrail_30_60 = evidence_has_30_60(provisional_evidence) and user_is_mans

for i in list(scores_best.keys()):
    txt = (meta[i].get("text", "") or "").lower()
    boost = 0.0

    # Mansioni
    if user_is_mans:
        if re.search(r"\b30\b", txt) and re.search(r"\b60\b", txt):
            boost += 0.14
        if "non si applica" in txt or "non si applicano" in txt:
            boost += 0.07
        if "conservazione del posto" in txt or "diritto alla conservazione" in txt:
            boost += 0.06
        if "affianc" in txt or "formaz" in txt or "addestr" in txt:
            boost += 0.03
        if "posto vacante" in txt:
            boost += 0.04
        if guardrail_30_60 and text_mentions_three_months(txt):
            boost -= 0.12

    # Malattia
    if user_is_mal:
        if "comporto" in txt:
            boost += 0.08
        if "malatt" in txt:
            boost += 0.05
        if "%" in txt or "percent" in txt or "trattamento econom" in txt:
            boost += 0.05
        if "certificat" in txt or "comunicaz" in txt:
            boost += 0.04
        if "reperibil" in txt or "visita fiscale" in txt:
            boost += 0.04

    # Permessi
    if user_is_perm and (not user_is_rol):
        if "permess" in txt or "assenze retribuite" in txt:
            boost += 0.07
        for pats in PERMESSI_CATEGORIES.values():
            if any(re.search(p, txt, flags=re.IGNORECASE) for p in pats):
                boost += 0.03
                break

    # ROL / ex festività
    if user_is_rol:
        if re.search(r"\brol\b", txt) or "riduzione orario" in txt:
            boost += 0.14
        if "ex festiv" in txt or "festività soppresse" in txt or "festivita soppresse" in txt:
            boost += 0.14
        if "diritto allo studio" in txt or "150 ore" in txt:
            boost -= 0.10

    # ✅ NOTTURNO ORDINARIO vs STRAORDINARIO (fix 60%)
    if user_is_notturno_ord:
        if "straordin" in txt:
            boost -= NOTTURNO_PENALTY_STRAO
        if "notturn" in txt and ("turn" in txt or "ordinario" in txt):
            boost += NOTTURNO_BOOST_TURNI

    scores_best[i] = scores_best[i] + boost

final_idx = sorted(scores_best.keys(), key=lambda i: scores_best[i], reverse=True)[:TOP_K_FINAL]
selected = [meta[i] for i in final_idx]

# Optional BM25 rerank on selected (additional precision)
if HAS_BM25 and len(selected) >= 6:
    docs_texts = [(c.get("text", "") or "") for c in selected]
    tokenized = [re.findall(r"\w+", t.lower()) for t in docs_texts]
    bm25 = BM25Okapi(tokenized)
    q_tokens = re.findall(r"\w+", enriched_q.lower())
    bm_scores = bm25.get_scores(q_tokens)
    order = np.argsort(-bm_scores)
    selected = [selected[int(j)] for j in order[:TOP_K_FINAL]]

context = "\n\n---\n\n".join([f"[Foglio PDF {c.get('page','?')}] {c.get('text','')}" for c in selected])

# Evidence (admin/debug/internal)
key_evidence = extract_key_evidence(selected)
evidence_block = "\n".join([f"- {e}" for e in key_evidence]) if key_evidence else "- (Nessuna evidenza estratta automaticamente.)"

# Notes for prompt steering
guardrail_note = ""
if user_is_mans and evidence_has_30_60(key_evidence):
    guardrail_note = (
        "GUARDRAIL MANSIONI SUPERIORI: nel contesto compaiono 30 giorni continuativi / 60 non continuativi. "
        "Usa questi valori come riferimento principale se applicabili."
    )

rol_note = ""
if user_is_rol:
    rol_note = "NOTA ROL/EX FESTIVITÀ: concentrati su ROL/ex festività; non mischiare con studio se non richiesto."

ipzs_note = ""
if user_mentions_ipzs:
    ipzs_note = "NOTA IPZS: può dipendere da prassi/accordi aziendali; segnala verifica con RSU/UILCOM e/o HR."

notturno_note = ""
if user_is_notturno_ord:
    notturno_note = (
        "NOTA NOTTURNO: domanda su NOTTURNO ORDINARIO. "
        "Non usare percentuali di straordinario notturno (es. 60%) se non esplicitamente riferite al notturno ordinario nel contesto."
    )

perm_format_note = ""
if user_is_perm and (not user_is_rol):
    perm_format_note = (
        "FORMATO PERMESSI: elenca solo categorie supportate dal contesto recuperato. "
        "Se l'utente si aspetta categorie non trovate, dichiarale come 'Non emerge dal CCNL recuperato'."
    )

# Model
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Prompt
prompt = f"""
{rules}

{guardrail_note}
{perm_guardrail_note}
{perm_format_note}
{rol_note}
{ipzs_note}
{notturno_note}

DOMANDA (UTENTE):
{user_input}

DOMANDA ARRICCHITA (MEMORIA BREVE):
{enriched_q}

EVIDENZE (INTERNE - usa come guida, non citarle all'utente):
{evidence_block}

CONTESTO (estratti CCNL):
{context}

SCRIVI LA RISPOSTA CON QUESTA STRUTTURA (SENZA MOSTRARE FONTI ALL'UTENTE):

Risposta UILCOM:
(2–6 righe, chiare)

Dettagli:
(4–10 punti; se c'è una esclusione/limitazione che cambia l'esito, mettila come primo punto)

Consiglio pratico UILCOM:
(1–2 bullet operativi, brevi)

Nota UILCOM:
Questa risposta è informativa; per casi specifici verificare con RSU/UILCOM o HR e con il testo ufficiale.

RISPOSTA:
"""

try:
    answer = llm.invoke(prompt).content
except Exception as e:
    answer = f"Errore nel generare la risposta: {e}"

st.session_state.messages.append({
    "role": "assistant",
    "content": answer,
    # Salviamo le fonti SOLO per debug/admin (non mostrate ai non-admin)
    "sources": selected if st.session_state.debug_on else []
})
st.rerun()
