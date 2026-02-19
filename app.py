import os
import json
import re
import hashlib
from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI


# =========================================================
# CONFIG UI (pulita tipo ChatGPT)
# =========================================================
st.set_page_config(page_title="Assistente Contrattuale UILCOM IPZS", page_icon="🟦")

st.title("🟦 Assistente Contrattuale UILCOM IPZS")
st.caption("Accesso riservato agli iscritti UILCOM — strumento informativo basato sul CCNL caricato.")
st.divider()


# =========================================================
# PATHS / PARAMETRI
# =========================================================
PDF_PATH = os.path.join("documenti", "ccnl.pdf")
INDEX_DIR = "index_ccnl"

VEC_PATH = os.path.join(INDEX_DIR, "vectors.npy")
META_PATH = os.path.join(INDEX_DIR, "chunks.json")

# Retrieval
TOP_K_FINAL = 18
TOP_K_PER_QUERY = 14
MAX_MULTI_QUERIES = 14

# Memoria breve (solo per migliorare retrieval)
MEMORY_USER_TURNS = 3

# Permessi: se domanda è ampia, vogliamo trovare più categorie
PERMESSI_MIN_CATEGORY_COVERAGE = 3

# Mansioni/categoria: se non troviamo 30/60, facciamo Super Pass 2 mirato
MANSIONI_MIN_SIGNAL = 1


# =========================================================
# PASSWORDS (USER + ADMIN DEBUG)
# =========================================================
def _get_secret(name: str) -> Optional[str]:
    val = None
    try:
        val = st.secrets.get(name, None)  # type: ignore
    except Exception:
        val = None
    if not val:
        val = os.getenv(name)
    return val

UILCOM_PASSWORD = _get_secret("UILCOM_PASSWORD")
ADMIN_PASSWORD = _get_secret("ADMIN_PASSWORD")  # opzionale


def _hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


if "auth_ok" not in st.session_state:
    st.session_state.auth_ok = False
if "is_admin" not in st.session_state:
    st.session_state.is_admin = False


with st.expander("🔒 Accesso iscritti UILCOM", expanded=not st.session_state.auth_ok):
    if UILCOM_PASSWORD:
        pwd_in = st.text_input("Password iscritti", type="password", placeholder="Inserisci password")
        if st.button("Entra", use_container_width=True):
            if pwd_in == UILCOM_PASSWORD:
                st.session_state.auth_ok = True
                st.success("Accesso consentito.")
            else:
                st.session_state.auth_ok = False
                st.error("Password non corretta.")
    else:
        st.info("Password iscritti non impostata (UILCOM_PASSWORD). In locale si prosegue per test.")
        st.session_state.auth_ok = True

if not st.session_state.auth_ok:
    st.stop()

# Login admin (solo se ADMIN_PASSWORD presente)
if ADMIN_PASSWORD:
    with st.expander("🛠️ Admin (Debug)", expanded=False):
        ap = st.text_input("Admin password", type="password", placeholder="Solo per debug")
        if st.button("Accedi come Admin", use_container_width=True):
            if ap == ADMIN_PASSWORD:
                st.session_state.is_admin = True
                st.success("Admin attivo: debug visibile solo a te.")
            else:
                st.session_state.is_admin = False
                st.error("Admin password errata.")
else:
    st.session_state.is_admin = False


# =========================================================
# UTILS: NORMALIZE / COSINE
# =========================================================
def normalize_rows(mat: np.ndarray) -> np.ndarray:
    return mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)

def cosine_scores(query_vec: np.ndarray, mat_norm: np.ndarray) -> np.ndarray:
    q = query_vec / (np.linalg.norm(query_vec) + 1e-12)
    return mat_norm @ q


# =========================================================
# TRIGGERS / INTENT
# =========================================================
CONSERVAZIONE_TRIGGERS = [
    "maternità", "maternita",
    "congedo maternità", "congedo maternita",
    "congedo parentale", "parentale",
    "malattia", "infortunio", "aspettativa",
    "assente", "assenza", "sostituzione", "sostituendo", "sto sostituendo",
    "conservazione del posto", "diritto alla conservazione",
]

# RAO: voce spesso aziendale
RAO_TRIGGERS = [
    "rao", "riposi annui orari", "riposo annuo orario", "permessi rao", "ore rao"
]

# Notturno: distinguere ordinario vs straordinario
NOTTURNO_TRIGGERS = ["notturno", "lavoro notturno", "turno notturno", "ore notturne", "notte"]
STRAORDINARIO_TRIGGERS = ["straordin", "straordinario", "extra", "oltre orario"]
FESTIVO_TRIGGERS = ["festiv", "domenic", "festivo", "festività", "festivita"]

# Mansioni/categoria
MANSIONI_ALTE_TRIGGERS = [
    "mansioni più alte", "mansioni piu alte",
    "mansioni più elevate", "mansioni piu elevate",
    "mansioni superiori", "mansioni superiore", "mansione superiore",
    "livello superiore", "categoria superiore", "inquadramento superiore",
    "passaggio di livello", "passaggio livello",
    "passaggio di categoria", "passaggio categoria",
    "come si passa di categoria", "come si passa ad una categoria superiore",
    "quando si cambia categoria",
    "posto vacante",
    "avanzamento", "promozione",
]

PASSAGGIO_CATEGORIA_TRIGGERS = [
    "passaggio categoria", "passare di categoria", "categoria superiore",
    "inquadramento superiore", "livello superiore", "promozione",
    "mansioni superiori", "mansioni superior", "posto vacante",
]

MALATTIA_TRIGGERS = [
    "malattia", "ammal", "certificat", "certificato", "inps",
    "comporto", "prognosi", "ricaduta",
    "visita fiscale", "controllo", "reperibil", "fasce",
    "assenza per malattia", "indennità", "indennita", "trattamento economico",
    "ospedal", "ricovero", "day hospital",
    "malattia durante ferie", "mi ammalo in ferie",
    "infortunio", "infortun",
]

# Permessi (generici) + ROL/ex-fest separati
PERMESSI_TRIGGERS = [
    "permess", "permesso", "permessi", "retribuit", "assenze retribuite",
    "visita medica", "visite mediche", "medico", "specialista",
    "lutto", "matrimonio", "nozze",
    "studio", "formazione", "esami",
    "104", "legge 104", "handicap",
    "donazione sangue", "donazione",
    "sindacal", "assemblea", "rsu",
    "rol", "ex festiv", "ex-festiv", "exfestiv", "festivit", "festività", "festivita",
    "rao", "riposi annui orari"
]

ROL_TRIGGERS = [
    "rol", "r.o.l", "riduzione orario", "riduzione dell'orario", "riduzione orario di lavoro",
    "ex festiv", "ex-festiv", "exfestiv", "ex festività", "ex-festività", "ex festivita",
    "festività soppresse", "festivita soppresse", "festività abolite", "festivita abolite",
    "permessi rol", "ore rol", "giorni rol",
    "quanti rol", "quante ore rol", "quanto rol",
    "quante ex festività", "quante ex festivita", "quante festività soppresse",
    "rao", "riposi annui orari"
]

IPZS_TRIGGERS = [
    "ipzs", "poligrafico", "zecca",
    "accordo aziendale", "accordi aziendali",
    "ordine di servizio", "ods",
    "turni", "reparto", "linea", "impianto",
]


def is_mansioni_superiori_question(q: str) -> bool:
    ql = q.lower()
    return any(t in ql for t in MANSIONI_ALTE_TRIGGERS)

def is_passaggio_categoria_question(q: str) -> bool:
    ql = q.lower()
    return any(t in ql for t in PASSAGGIO_CATEGORIA_TRIGGERS)

def is_malattia_question(q: str) -> bool:
    ql = q.lower()
    return any(t in ql for t in MALATTIA_TRIGGERS)

def is_permessi_question(q: str) -> bool:
    ql = q.lower()
    return any(t in ql for t in PERMESSI_TRIGGERS)

def is_rol_question(q: str) -> bool:
    ql = q.lower()
    return any(t in ql for t in ROL_TRIGGERS)

def is_ipzs_context(q: str) -> bool:
    ql = q.lower()
    return any(t in ql for t in IPZS_TRIGGERS)

def mentions_rao(q: str) -> bool:
    ql = q.lower()
    return any(t in ql for t in RAO_TRIGGERS)

def is_notturno_question(q: str) -> bool:
    ql = q.lower()
    return any(t in ql for t in NOTTURNO_TRIGGERS)

def is_straordinario_question(q: str) -> bool:
    ql = q.lower()
    return any(t in ql for t in STRAORDINARIO_TRIGGERS)

def is_notturno_ordinario(q: str) -> bool:
    # notturno ma NON straordinario
    return is_notturno_question(q) and (not is_straordinario_question(q))

def is_straordinario_notturno(q: str) -> bool:
    ql = q.lower()
    return is_notturno_question(q) and any(t in ql for t in STRAORDINARIO_TRIGGERS)


# =========================================================
# PERMESSI: CATEGORIE (per coverage)
# =========================================================
PERMESSI_CATEGORIES = {
    "visite_mediche": [r"visite?\s+med", r"accertament", r"specialist", r"sanitar"],
    "lutto": [r"\blutto\b", r"decesso", r"grave\s+lutto", r"familiare"],
    "matrimonio": [r"matrimon", r"nozz", r"congedo\s+matrimon"],
    "studio_formazione": [r"diritto\s+allo\s+studio", r"\b150\s+ore\b", r"\bstudio\b", r"\besami\b", r"formazion"],
    "legge_104": [r"\b104\b", r"legge\s*104", r"handicap"],
    "sindacali": [r"sindacal", r"\brsu\b", r"assemblea", r"permessi?\s+sindacal"],
    "donazione_sangue": [r"donazion", r"sangue", r"emocomponent"],
    "rol_exfest": [r"\brol\b", r"riduzione\s+orario", r"ex\s*fest", r"festivit", r"festività\s+soppresse", r"festivita\s+soppresse"],
}

def permessi_category_coverage(chunks: List[Dict[str, Any]]) -> Tuple[int, List[str]]:
    joined = " ".join([(c.get("text", "") or "") for c in chunks]).lower()
    found = set()
    for cat, pats in PERMESSI_CATEGORIES.items():
        for p in pats:
            if re.search(p, joined, flags=re.IGNORECASE):
                found.add(cat)
                break
    return len(found), sorted(found)

def build_permessi_expansion_queries(q: str) -> List[str]:
    base = q.strip()
    return [
        f"{base} permessi visite mediche retribuiti",
        f"{base} permessi lutto retribuiti giorni familiari",
        f"{base} congedo matrimoniale retribuito",
        f"{base} permessi sindacali assemblea ore",
        f"{base} diritto allo studio 150 ore triennio",
        f"{base} donazione sangue permesso retribuito",
        f"{base} ROL riduzione orario monte ore",
        f"{base} ex festività festività soppresse ore",
    ][:MAX_MULTI_QUERIES]


# =========================================================
# MANSIONI: SUPER PASS 2 (se manca 30/60)
# =========================================================
def mansioni_signal(chunks: List[Dict[str, Any]]) -> int:
    joined = " ".join([(c.get("text", "") or "") for c in chunks]).lower()
    signals = ["mansioni superiori", "posto vacante", "conservazione del posto", "30", "60", "trenta", "sessanta"]
    return sum(1 for s in signals if s in joined)

def build_mansioni_expansion_queries(q: str) -> List[str]:
    base = q.strip()
    return [
        "mansioni superiori 30 giorni continuativi 60 giorni discontinui posto vacante",
        "assegnazione mansioni superiori 30 60 conservazione del posto non si applica sostituzione",
        "passaggio di categoria mansioni superiori 30 60 posto vacante",
        "non si applica sostituzione dipendente assente diritto alla conservazione del posto mansioni superiori",
        "formazione addestramento affiancamento non costituisce mansioni superiori",
        f"{base} mansioni superiori 30 60",
    ][:MAX_MULTI_QUERIES]


# =========================================================
# MEMORIA BREVE (solo per retrieval, non mostrata all'utente)
# =========================================================
def build_enriched_question(current_q: str) -> str:
    if "messages" not in st.session_state:
        return current_q.strip()

    user_msgs = [m["content"] for m in st.session_state.messages if m.get("role") == "user" and m.get("content")]
    prev = user_msgs[:-1] if (user_msgs and user_msgs[-1].strip() == current_q.strip()) else user_msgs
    last = [x.strip() for x in (prev[-MEMORY_USER_TURNS:] if prev else []) if x.strip()]

    enriched = current_q.strip()
    if last:
        enriched = (
            "CONTESTO CONVERSAZIONE (ultime richieste utente):\n"
            + "\n".join([f"- {x}" for x in last])
            + "\n\nDOMANDA ATTUALE:\n"
            + current_q.strip()
        )

    # Nota RAO: possibile voce aziendale/busta paga
    if mentions_rao(current_q):
        enriched += (
            "\n\nNOTA: 'RAO' potrebbe essere una dicitura aziendale/busta paga. "
            "Nel CCNL cercare ROL/ex festività/riduzione orario come riferimenti contrattuali."
        )
    return enriched


# =========================================================
# QUERY BUILDER (multi-query)
# =========================================================
def build_queries(q: str) -> List[str]:
    q0 = q.strip()
    ql = q0.lower()

    qs = [q0, f"{q0} CCNL", f"{q0} regole condizioni", f"{q0} procedura definizione"]

    user_is_rol = is_rol_question(q0)
    user_is_perm = is_permessi_question(q0)
    user_is_mal = is_malattia_question(q0)
    user_is_passcat = is_passaggio_categoria_question(q0)
    user_is_mans = is_mansioni_superiori_question(q0) or user_is_passcat
    user_is_conserv = any(t in ql for t in CONSERVAZIONE_TRIGGERS)

    # ROL / ex festività
    if user_is_rol:
        qs += [
            "ROL riduzione orario di lavoro monte ore annuo maturazione fruizione",
            "ex festività festività soppresse permessi ore giorni spettanti",
            "permessi ROL ed ex festività: quanti, come maturano e come si usano",
            "ROL ex festività richiesta fruizione preavviso programmazione residui",
        ]

    # Permessi generici: forza anche ROL/exfest tra le query (così non resta solo studio)
    if user_is_perm and (not user_is_rol):
        qs += [
            "permessi retribuiti tipologie CCNL elenco completo",
            "assenze retribuite tipologie (visite mediche, lutto, matrimonio, sindacali, studio, donazione sangue)",
            "congedo matrimoniale retribuito",
            "permessi per lutto giorni familiari",
            "permessi per visite mediche giustificativo",
            "assemblea sindacale ore retribuite",
            "diritto allo studio 150 ore triennio",
            "donazione sangue permesso retribuito",
            # ANCORA ROL/ex-fest (serve per completezza, anche se l'utente non li nomina)
            "ROL riduzione orario di lavoro CCNL grafici editoria monte ore annuo",
            "ex festività festività soppresse permessi ore giorni CCNL grafici editoria",
        ]

        # RAO: non inventare — cercare riferimenti contrattuali equivalenti
        if mentions_rao(q0):
            qs += [
                "RAO riposi annui orari busta paga corrispondenza con ROL riduzione orario",
                "riduzione orario di lavoro ROL permessi annui",
            ]

    # Malattia
    if user_is_mal:
        qs += [
            "malattia trattamento economico percentuali integrazione",
            "malattia periodo di comporto regole conteggio",
            "malattia obblighi comunicazione certificazione",
            "visita fiscale reperibilità fasce controllo",
            "ricovero ospedaliero day hospital ricaduta",
            "malattia durante ferie sospensione ferie",
        ]

    # Ferie (generico)
    if any(k in ql for k in ["ferie", "residu", "matur", "programmaz", "chiusura"]):
        qs += [
            "ferie giorni spettanti maturazione fruizione frazionamento",
            "residui ferie termini regole",
            "malattia durante ferie cosa succede",
        ]

    # Straordinari
    if any(k in ql for k in ["straordin", "maggior", "notturn", "festiv"]):
        qs += [
            "lavoro straordinario maggiorazioni",
            "straordinario notturno maggiorazione",
            "straordinario festivo maggiorazione",
            "lavoro notturno ordinario maggiorazione",
        ]

    # Mansioni superiori / passaggio categoria
    if user_is_mans or user_is_conserv:
        qs += [
            "mansioni superiori posto vacante",
            "assegnazione a mansioni superiori regole generali",
            "mansioni superiori 30 giorni consecutivi 60 discontinui",
            "non si applica in caso di sostituzione di dipendente assente con diritto alla conservazione del posto",
            "formazione addestramento affiancamento non costituisce mansioni superiori",
            "trattamento economico durante mansioni superiori",
        ]

    # Dedup + limit
    out, seen = [], set()
    for x in qs:
        x = x.strip()
        if x and x not in seen:
            out.append(x)
            seen.add(x)
    return out[:MAX_MULTI_QUERIES]


# =========================================================
# EVIDENCE EXTRACTION (ROBUSTA)
# =========================================================
def _as_chunk_dict(c: Any) -> Optional[Dict[str, Any]]:
    if isinstance(c, dict):
        return c
    return None

def extract_key_evidence(chunks: List[Dict[str, Any]]) -> List[str]:
    """
    Estrae righe operative: numeri, percentuali, esclusioni ("non si applica"), 30/60, tre mesi.
    Supporta anche 'trenta'/'sessanta'.
    """
    evidences: List[str] = []

    patterns = [
        r"\b30\b", r"\b60\b", r"\btrenta\b", r"\bsessanta\b",
        r"\b30\s+giorni\b", r"\b60\s+giorni\b",
        r"\b\d{1,3}\b", r"\b%\b",
        r"posto\s+vacante",
        r"mansioni?\s+superiori?",
        r"sostituzion",
        r"conservazion.*posto",
        r"diritto.*conservazion.*posto",
        r"non\s+si\s+applica",
        r"non\s+costituisc",
        r"affiancament",
        r"formazion",
        r"addestrament",
        r"straordin",
        r"lavoro\s+notturno",
        r"turno\s+notturno",
        r"festiv",
        r"rol\b",
        r"riduzione\s+orario",
        r"ex\s*fest",
        r"lutto",
        r"matrimon",
        r"assemblea",
        r"\b3\s*mesi\b", r"tre\s+mesi",
    ]

    def interesting(ln_low: str) -> bool:
        return any(re.search(p, ln_low, flags=re.IGNORECASE) for p in patterns)

    for raw in chunks:
        c = _as_chunk_dict(raw)
        if not c:
            continue
        page = c.get("page", "?")
        text = c.get("text", "") or ""

        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        for ln in lines:
            ln_low = ln.lower()
            if interesting(ln_low):
                ln_clean = " ".join(ln.split())
                if 18 <= len(ln_clean) <= 420:
                    evidences.append(f"(pdfpag. {page}) {ln_clean}")

    # Dedup
    out, seen = [], set()
    for e in evidences:
        if e not in seen:
            out.append(e)
            seen.add(e)
    return out[:20]

def evidence_has_30_60(evidence_lines: List[str]) -> bool:
    joined = " ".join(evidence_lines).lower()
    has_30 = (re.search(r"\b30\b", joined) is not None) or ("trenta" in joined)
    has_60 = (re.search(r"\b60\b", joined) is not None) or ("sessanta" in joined)
    return has_30 and has_60

def evidence_mentions_three_months(evidence_lines: List[str]) -> bool:
    joined = " ".join(evidence_lines).lower()
    return ("tre mesi" in joined) or (re.search(r"\b3\s*mesi\b", joined) is not None)

def text_mentions_three_months(txt: str) -> bool:
    t = (txt or "").lower()
    return ("tre mesi" in t) or (re.search(r"\b3\s*mesi\b", t) is not None)


# =========================================================
# INDEX BUILD / LOAD
# =========================================================
def build_index() -> int:
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"Non trovo il PDF: {PDF_PATH} (metti 'ccnl.pdf' in /documenti)")

    os.makedirs(INDEX_DIR, exist_ok=True)

    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=160)
    chunks = splitter.split_documents(docs)

    texts = [c.page_content for c in chunks]
    pages = [(c.metadata.get("page", 0) + 1) for c in chunks]  # pagina PDF (foglio), NON numero CCNL

    emb = OpenAIEmbeddings()
    vectors = emb.embed_documents(texts)
    vectors = np.array(vectors, dtype=np.float32)

    np.save(VEC_PATH, vectors)

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump([{"page": p, "text": t} for p, t in zip(pages, texts)], f, ensure_ascii=False)

    return len(chunks)

def load_index() -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    vectors = np.load(VEC_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    # normalizza: deve essere lista di dict
    meta2: List[Dict[str, Any]] = []
    for x in meta:
        if isinstance(x, dict) and "text" in x:
            meta2.append({"page": x.get("page", "?"), "text": x.get("text", "")})
    return vectors, meta2


# =========================================================
# RULES (super-precisione + RAO B + no fonti visibili all'utente)
# =========================================================
RULES = (
    "Sei l’assistente UILCOM per lavoratori IPZS. "
    "Rispondi in modo chiaro, professionale e pratico basandoti SOLO sul contesto del CCNL fornito. "
    "Non inventare informazioni. "
    "Se non trovi la risposta nel contesto, scrivi: 'Non ho trovato la risposta nel CCNL caricato'. "
    "Non mostrare pagine o riferimenti all’utente.\n\n"

    "REGOLA PROVA: puoi affermare un dato (numero, percentuale, durata, diritto, condizione) SOLO se è supportato dal contesto. "
    "Se non è supportato, dichiaralo come 'Non emerge dal CCNL recuperato'.\n\n"

    "REGOLA RAO (OPZIONE B): se l’utente cita 'RAO', NON affermare che sia una voce del CCNL salvo prova testuale. "
    "Trattala come possibile dicitura aziendale/busta paga e collega (se nel contesto) a ROL/ex festività/riduzione orario. "
    "Se nel contesto non emergono ROL/ex festività, scrivi che non emergono.\n\n"

    "REGOLA NOTTURNO: distinguere SEMPRE 'lavoro notturno ordinario' da 'straordinario notturno'. "
    "Non usare la percentuale dello straordinario per rispondere al notturno ordinario.\n\n"

    "REGOLA MANSIONI/CATEGORIA: se nel contesto sono presenti 30 giorni continuativi / 60 giorni non continuativi "
    "e le esclusioni (sostituzione con diritto alla conservazione del posto; formazione/affiancamento), riportale esplicitamente.\n\n"

    "Quando la domanda è ampia (permessi, malattia, ferie, straordinari, livelli), organizza in sotto-punti e scrivi solo ciò che è supportato."
)


# =========================================================
# SIDEBAR: INDICE / SESSIONE
# =========================================================
with st.sidebar:
    st.header("⚙️ Controlli")
    ok_index = os.path.exists(VEC_PATH) and os.path.exists(META_PATH)
    st.write("📦 Indice presente:", "✅" if ok_index else "❌")

    if st.button("Indicizza CCNL (prima volta / dopo cambio PDF)", use_container_width=True):
        try:
            with st.spinner("Indicizzazione in corso..."):
                n = build_index()
            st.success(f"Indicizzazione completata! Chunk creati: {n}")
        except Exception as e:
            st.error(str(e))

    if st.button("🧹 Nuova chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    if st.session_state.is_admin:
        st.divider()
        st.subheader("🛠️ Admin Debug")
        st.caption("Debug visibile solo a te.")
        st.session_state.setdefault("admin_show_sources", True)
        st.session_state.admin_show_sources = st.toggle("Mostra fonti/chunk (admin)", value=True)
        st.session_state.setdefault("admin_show_queries", True)
        st.session_state.admin_show_queries = st.toggle("Mostra query & evidenze (admin)", value=True)


# =========================================================
# CHAT STATE
# =========================================================
if "messages" not in st.session_state:
    st.session_state.messages = []


# Render chat (solo contenuti, NO fonti per utenti)
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

    # Admin: fonti/debug per messaggi assistant
    if st.session_state.is_admin and m.get("role") == "assistant":
        dbg = m.get("debug", {})
        if st.session_state.get("admin_show_sources") and dbg.get("selected_chunks"):
            with st.expander("🧾 Admin: Chunk recuperati (PDF page/foglio)", expanded=False):
                for c in dbg["selected_chunks"]:
                    st.write(f"PDF pag/foglio: {c.get('page', '?')}")
                    st.write((c.get("text","") or "")[:1200] + ("..." if len(c.get("text","") or "") > 1200 else ""))
                    st.divider()
        if st.session_state.get("admin_show_queries") and dbg:
            with st.expander("🧠 Admin: Query / Evidenze / Note guardrail", expanded=False):
                st.write("**Queries:**")
                for q in dbg.get("queries", []):
                    st.write("-", q)
                st.divider()
                st.write("**Evidenze estratte:**")
                for e in dbg.get("evidence", []):
                    st.write("-", e)
                st.divider()
                if dbg.get("guardrails"):
                    st.write("**Guardrails:**")
                    for g in dbg["guardrails"]:
                        st.write("-", g)


# =========================================================
# INPUT
# =========================================================
user_input = st.chat_input("Scrivi una domanda sul CCNL (permessi, ROL/ex-fest, malattia, straordinari, livelli...)")

if not user_input:
    st.stop()

st.session_state.messages.append({"role": "user", "content": user_input})

if not (os.path.exists(VEC_PATH) and os.path.exists(META_PATH)):
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Prima devo indicizzare il CCNL: apri il menu a sinistra e clicca **Indicizza CCNL**."
    })
    st.rerun()


# =========================================================
# RETRIEVAL + GUARDRAILS
# =========================================================
enriched_q = build_enriched_question(user_input)

vectors, meta = load_index()
mat_norm = normalize_rows(vectors)
emb = OpenAIEmbeddings()

user_is_rol = is_rol_question(enriched_q)
user_is_perm = is_permessi_question(enriched_q)
user_is_mal = is_malattia_question(enriched_q)
user_is_passcat = is_passaggio_categoria_question(enriched_q)
user_is_mans = is_mansioni_superiori_question(enriched_q) or user_is_passcat
user_is_conserv = any(t in enriched_q.lower() for t in CONSERVAZIONE_TRIGGERS)
user_mentions_ipzs = is_ipzs_context(enriched_q)
user_mentions_rao = mentions_rao(enriched_q)

q_is_notturno_ordinario = is_notturno_ordinario(enriched_q)
q_is_straord_notturno = is_straordinario_notturno(enriched_q)

# Se domanda è "oltre ROL/exfest" → trattiamo come esclusione
exclude_rol = False
qlow = enriched_q.lower()
if ("oltre" in qlow or "al di là" in qlow or "al di la" in qlow) and any(x in qlow for x in ["rol", "ex fest", "festività soppresse", "festivita soppresse", "rao"]):
    exclude_rol = True


# Pass 1 retrieval
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

guardrails: List[str] = []

# Mansioni/categoria: se non troviamo segnali, Super Pass 2 mirato
if user_is_mans:
    has_30_60_now = evidence_has_30_60(provisional_evidence)
    sig = mansioni_signal(provisional_selected)
    if (not has_30_60_now) and (sig < MANSIONI_MIN_SIGNAL):
        guardrails.append("Super Pass 2 mansioni: rilancio query mirate su 30/60 + posto vacante + conservazione del posto.")
        extra_qs = build_mansioni_expansion_queries(enriched_q)
        for q in extra_qs:
            qvec = np.array(emb.embed_query(q), dtype=np.float32)
            sims = cosine_scores(qvec, mat_norm)
            top_idx = np.argsort(-sims)[:TOP_K_PER_QUERY]
            for i in top_idx:
                s = float(sims[i])
                if (i not in scores_best) or (s > scores_best[i]):
                    scores_best[i] = s

# Permessi: se domanda generica e copertura bassa, Pass 2 permessi
if user_is_perm and (not user_is_rol):
    cov_n, cov_list = permessi_category_coverage(provisional_selected)
    if cov_n < PERMESSI_MIN_CATEGORY_COVERAGE:
        guardrails.append(f"Super Pass 2 permessi: copertura bassa ({cov_n} categorie: {cov_list}). Rilancio query per categorie.")
        extra_queries = build_permessi_expansion_queries(enriched_q)
        for q in extra_queries:
            qvec = np.array(emb.embed_query(q), dtype=np.float32)
            sims = cosine_scores(qvec, mat_norm)
            top_idx = np.argsort(-sims)[:TOP_K_PER_QUERY]
            for i in top_idx:
                s = float(sims[i])
                if (i not in scores_best) or (s > scores_best[i]):
                    scores_best[i] = s


# Re-ranking con boost/penalty mirati
for i in list(scores_best.keys()):
    txt = (meta[i].get("text", "") or "").lower()
    boost = 0.0

    # ====== Mansioni/categoria ======
    if user_is_mans:
        if (re.search(r"\b30\b", txt) or "trenta" in txt) and (re.search(r"\b60\b", txt) or "sessanta" in txt):
            boost += 0.20
        if "mansioni superior" in txt:
            boost += 0.10
        if "posto vacante" in txt:
            boost += 0.08
        if "non si applica" in txt or "non si applicano" in txt:
            boost += 0.07
        if "conservazione del posto" in txt or "diritto alla conservazione" in txt:
            boost += 0.08
        if "affianc" in txt or "formaz" in txt or "addestr" in txt:
            boost += 0.05

    # ====== ROL / ex-fest (quando serve) ======
    if user_is_perm and (not user_is_rol) and (not exclude_rol):
        # forza ROL/exfest quando l'utente chiede permessi in generale
        if re.search(r"\brol\b", txt) or "riduzione orario" in txt:
            boost += 0.18
        if "ex festiv" in txt or "festività soppresse" in txt or "festivita soppresse" in txt:
            boost += 0.18

    if user_is_rol:
        if re.search(r"\brol\b", txt) or "riduzione orario" in txt:
            boost += 0.20
        if "ex festiv" in txt or "festività soppresse" in txt or "festivita soppresse" in txt:
            boost += 0.20
        # penalizza studio se domanda è ROL
        if "diritto allo studio" in txt or "150 ore" in txt:
            boost -= 0.12

    # ====== RAO: non è CCNL standard => non boost su "rao" se non supportato ======
    # (qui non facciamo boost su RAO in chunk perché spesso non compare nel CCNL)

    # ====== NOTTURNO vs STRAORDINARIO NOTTURNO ======
    if q_is_notturno_ordinario:
        # penalizza chunk che parlano di straordinario (per evitare 60% se domanda è notturno ordinario)
        if "straordin" in txt:
            boost -= 0.18
        # penalizza match tipici 60% se legati a straordinario
        if "60" in txt and "straordin" in txt:
            boost -= 0.15
        # boost se parla di notturno ma NON di straordinario
        if "notturn" in txt and ("straordin" not in txt):
            boost += 0.14

    if q_is_straord_notturno:
        # qui vogliamo proprio lo straordinario notturno
        if "straordin" in txt and "notturn" in txt:
            boost += 0.20
        if "60" in txt:
            boost += 0.08

    # ====== Malattia ======
    if user_is_mal:
        if "comporto" in txt:
            boost += 0.10
        if "malatt" in txt:
            boost += 0.06
        if "%" in txt or "percent" in txt or "trattamento econom" in txt:
            boost += 0.06
        if "certificat" in txt or "comunicaz" in txt:
            boost += 0.05
        if "reperibil" in txt or "visita fiscale" in txt:
            boost += 0.05

    scores_best[i] = scores_best[i] + boost


final_idx = sorted(scores_best.keys(), key=lambda i: scores_best[i], reverse=True)[:TOP_K_FINAL]
selected = [meta[i] for i in final_idx]

# Context per LLM
context = "\n\n---\n\n".join([f"[PDF_FOGLIO {c.get('page','?')}] {c.get('text','')}" for c in selected])

key_evidence = extract_key_evidence(selected)

# Guardrail testo per LLM (interno, non utente)
guardrail_notes = ""
if user_is_mans and evidence_has_30_60(key_evidence):
    guardrail_notes += (
        "\nGUARDRAIL MANSIONI/CATEGORIA: Nel contesto sono presenti 30/60 (o trenta/sessanta) + esclusioni. "
        "Devi riportare ESPLICITAMENTE 30/60 e le esclusioni (sostituzione con conservazione del posto; affiancamento/formazione). "
        "Se nel contesto compaiono anche '3 mesi/tre mesi' ma non è nello stesso passaggio sulle mansioni superiori, NON usarlo.\n"
    )

if q_is_notturno_ordinario:
    guardrail_notes += (
        "\nGUARDRAIL NOTTURNO: La domanda è su lavoro notturno ORDINARIO. "
        "Non usare percentuali dello straordinario notturno (es. 60%) se nel contesto sono riferite a straordinario.\n"
    )

if q_is_straord_notturno:
    guardrail_notes += (
        "\nGUARDRAIL STRAORDINARIO NOTTURNO: La domanda è su straordinario notturno, usa le maggiorazioni specifiche di straordinario.\n"
    )

if user_mentions_rao:
    guardrail_notes += (
        "\nGUARDRAIL RAO: Se l'utente cita RAO, trattalo come possibile voce aziendale/busta paga e NON come voce CCNL senza prova testuale.\n"
    )

if exclude_rol:
    guardrail_notes += (
        "\nNOTA UTENTE: l'utente chiede 'oltre ROL/ex festività/RAO' — non includere ROL/ex festività tra le categorie elencate, "
        "a meno che servano solo per chiarire la distinzione.\n"
    )

ipzs_note = ""
if user_mentions_ipzs:
    ipzs_note = "NOTA IPZS: se la questione può dipendere da accordi/prassi aziendali IPZS, segnalalo nel consiglio pratico.\n"


# =========================================================
# LLM ANSWER (NO FONTI PER UTENTI)
# =========================================================
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

evidence_block = "\n".join([f"- {e}" for e in key_evidence]) if key_evidence else "- (nessuna evidenza estratta)"

PROMPT = f"""
{RULES}

{guardrail_notes}
{ipzs_note}

DOMANDA UTENTE:
{user_input}

CONTESTO CCNL (estratti):
{context}

EVIDENZE OPERATIVE (non mostrare all'utente):
{evidence_block}

ISTRUZIONI DI OUTPUT (OBBLIGATORIE):
- NON mostrare pagine, numeri pagina o riferimenti "PDF_FOGLIO" all’utente.
- Scrivi una risposta *esatta*, *operativa* e *breve*.
- Se mancano dati dal contesto, scrivi: "Non emerge dal CCNL recuperato" (senza inventare).
- Se la domanda riguarda permessi: separa chiaramente ROL/ex-fest vs altri permessi solo se pertinente; se l'utente chiede "oltre ROL", non includere ROL tra le categorie.
- Inserisci sempre 1–2 suggerimenti in "Consiglio pratico UILCOM".

FORMATO:
Risposta UILCOM:
(2–5 righe)

Dettagli operativi:
(3–8 bullet)

Consiglio pratico UILCOM:
(1–2 bullet)

Nota UILCOM:
(riga breve di cautela)

RISPOSTA:
"""

try:
    answer = llm.invoke(PROMPT).content
except Exception as e:
    answer = f"Errore nel generare la risposta: {e}"

# Salva debug SOLO per admin
debug_payload = {
    "queries": queries,
    "evidence": key_evidence,
    "guardrails": guardrails + ([guardrail_notes.strip()] if guardrail_notes.strip() else []),
    "selected_chunks": selected,
    "ts": datetime.now().isoformat(timespec="seconds"),
}

st.session_state.messages.append({
    "role": "assistant",
    "content": answer,
    "debug": debug_payload
})

st.rerun()
