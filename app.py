import os
import json
import re
import time
import numpy as np
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# BM25 (hybrid retrieval)
from rank_bm25 import BM25Okapi


# =========================
# PAGE: Clean ChatGPT-like UI
# =========================
st.set_page_config(page_title="UILCOM CCNL Chat", page_icon="💬", layout="wide", initial_sidebar_state="collapsed")
st.markdown(
    """
    <style>
      header[data-testid="stHeader"] {display:none;}
      footer {visibility:hidden;}
      .block-container {padding-top: 1.1rem; padding-bottom: 2rem; max-width: 980px;}
      /* remove extra whitespace above chat */
      [data-testid="stChatInput"] {position: sticky; bottom: 0; background: white; padding-top: 0.5rem;}
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown("### 💬 UILCOM CCNL Chat")


# =========================
# CONFIG
# =========================
PDF_PATH = os.path.join("documenti", "ccnl.pdf")

INDEX_DIR = "index_ccnl"
VEC_PATH = os.path.join(INDEX_DIR, "vectors.npy")
META_PATH = os.path.join(INDEX_DIR, "chunks.json")

TOP_K_FINAL = 18
TOP_K_PER_QUERY = 12
MAX_MULTI_QUERIES = 12
MEMORY_USER_TURNS = 3

# Hybrid weights
W_EMB = 0.70
W_BM25 = 0.30

# Permessi coverage
PERMESSI_MIN_CATEGORY_COVERAGE = 3


# =========================
# SECRETS
# =========================
def get_secret(name: str):
    try:
        v = st.secrets.get(name, None)  # type: ignore
    except Exception:
        v = None
    if not v:
        v = os.getenv(name)
    return v

UILCOM_PASSWORD = get_secret("UILCOM_PASSWORD")
ADMIN_PASSWORD = get_secret("ADMIN_PASSWORD")


# =========================
# AUTH: users
# =========================
if "auth_ok" not in st.session_state:
    st.session_state.auth_ok = False

if UILCOM_PASSWORD:
    if not st.session_state.auth_ok:
        with st.container(border=True):
            st.markdown("**Accesso iscritti UILCOM**")
            pwd_in = st.text_input("Password", type="password", label_visibility="collapsed", placeholder="Password")
            if st.button("Entra", use_container_width=True):
                if pwd_in == UILCOM_PASSWORD:
                    st.session_state.auth_ok = True
                    st.rerun()
                else:
                    st.error("Password non corretta.")
    if not st.session_state.auth_ok:
        st.stop()
else:
    # locale/test
    st.session_state.auth_ok = True


# =========================
# ADMIN: hidden mode ?admin=1
# =========================
qs = st.query_params
admin_mode = str(qs.get("admin", "0")).strip() == "1"

if "admin_ok" not in st.session_state:
    st.session_state.admin_ok = False

if admin_mode and ADMIN_PASSWORD and not st.session_state.admin_ok:
    with st.expander("🛠️ Admin", expanded=True):
        ap = st.text_input("Password admin", type="password", placeholder="Admin password")
        if st.button("Sblocca"):
            if ap == ADMIN_PASSWORD:
                st.session_state.admin_ok = True
                st.success("Admin OK")
                st.rerun()
            else:
                st.error("Admin password errata")

SHOW_ADMIN = bool(admin_mode and st.session_state.admin_ok)


# =========================
# HELPERS
# =========================
def safe_str(x) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    try:
        return str(x)
    except Exception:
        return ""

def ensure_chunk_dicts(items):
    out = []
    for it in items or []:
        if isinstance(it, dict):
            out.append({"page": it.get("page", "?"), "text": it.get("text", "")})
        elif isinstance(it, str):
            out.append({"page": "?", "text": it})
        else:
            out.append({"page": "?", "text": safe_str(it)})
    return out

def normalize_rows(mat: np.ndarray) -> np.ndarray:
    return mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)

def cosine_scores(query_vec: np.ndarray, mat_norm: np.ndarray) -> np.ndarray:
    q = query_vec / (np.linalg.norm(query_vec) + 1e-12)
    return mat_norm @ q

def tokenize(text: str) -> list[str]:
    # semplice tokenizzazione robusta per italiano
    text = safe_str(text).lower()
    return re.findall(r"[a-zàèéìòù0-9]+", text, flags=re.IGNORECASE)

def has_index() -> bool:
    return os.path.exists(VEC_PATH) and os.path.exists(META_PATH)


# =========================
# TOPIC TRIGGERS
# =========================
CONSERVAZIONE_TRIGGERS = [
    "maternità", "maternita", "congedo maternità", "congedo maternita",
    "congedo parentale", "parentale",
    "malattia", "infortunio",
    "aspettativa",
    "assente", "assenza", "sostituzione", "sostituendo", "sto sostituendo",
]
MANSIONI_ALTE_TRIGGERS = [
    "mansioni più alte", "mansioni piu alte",
    "mansioni superiori", "mansione superiore",
    "sostituisco", "sto sostituendo",
    "livello superiore", "categoria superiore", "inquadramento superiore",
    "passaggio di livello", "passaggio categoria",
    "posto vacante",
]
MALATTIA_TRIGGERS = [
    "malattia", "certificat", "certificato", "comporto",
    "visita fiscale", "reperibil", "fasce",
    "trattamento economico", "indennità", "indennita",
]
PERMESSI_TRIGGERS = [
    "permess", "retribuit", "assenze retribuite",
    "visita medica", "visite mediche",
    "lutto", "matrimonio", "nozze",
    "studio", "formazione", "esami",
    "104", "legge 104",
    "donazione sangue",
    "sindacal", "assemblea", "rsu",
    "rol", "ex festiv", "exfestiv", "festivit", "festività",
]
ROL_TRIGGERS = [
    "rol", "riduzione orario", "riduzione dell'orario",
    "ex festiv", "exfestiv", "ex festività", "ex festivita",
    "festività soppresse", "festivita soppresse",
]
STRAORD_NOTT_TRIGGERS = [
    "straordin", "notturn", "lavoro notturno", "turno di notte",
    "maggior", "festiv", "domenica"
]
IPZS_TRIGGERS = [
    "ipzs", "poligrafico", "zecca", "accordo aziendale", "accordi aziendali",
    "ordine di servizio", "ods", "turni",
]

PERMESSI_CATEGORIES = {
    "visite_mediche": [r"visite?\s+med", r"visita\s+med", r"specialist"],
    "lutto": [r"\blutto\b", r"decesso"],
    "matrimonio": [r"matrimon", r"nozz"],
    "studio_formazione": [r"diritto\s+allo\s+studio", r"\b150\s+ore\b", r"\besami\b", r"formazion"],
    "legge_104": [r"\b104\b", r"legge\s*104"],
    "sindacali": [r"sindacal", r"\brsu\b", r"assemblea"],
    "donazione_sangue": [r"donazion", r"sangue"],
}

def is_mansioni(q: str) -> bool:
    ql = q.lower()
    return any(t in ql for t in MANSIONI_ALTE_TRIGGERS) or any(t in ql for t in CONSERVAZIONE_TRIGGERS)

def is_malattia(q: str) -> bool:
    ql = q.lower()
    return any(t in ql for t in MALATTIA_TRIGGERS)

def is_permessi(q: str) -> bool:
    ql = q.lower()
    return any(t in ql for t in PERMESSI_TRIGGERS)

def is_rol(q: str) -> bool:
    ql = q.lower()
    return any(t in ql for t in ROL_TRIGGERS)

def is_straord_notturno(q: str) -> bool:
    ql = q.lower()
    return any(t in ql for t in STRAORD_NOTT_TRIGGERS)

def is_ipzs(q: str) -> bool:
    ql = q.lower()
    return any(t in ql for t in IPZS_TRIGGERS)

def permessi_coverage(chunks) -> int:
    joined = " ".join([safe_str(c.get("text", "")) for c in ensure_chunk_dicts(chunks)]).lower()
    found = 0
    for pats in PERMESSI_CATEGORIES.values():
        if any(re.search(p, joined, flags=re.IGNORECASE) for p in pats):
            found += 1
    return found


# =========================
# MEMORY
# =========================
def build_enriched_question(current_q: str) -> str:
    if "messages" not in st.session_state:
        return current_q.strip()
    user_msgs = [m["content"] for m in st.session_state.messages if m.get("role") == "user" and m.get("content")]
    prev = user_msgs[:-1] if user_msgs and user_msgs[-1].strip() == current_q.strip() else user_msgs
    last = [x.strip() for x in (prev[-MEMORY_USER_TURNS:] if prev else []) if x.strip()]
    if not last:
        return current_q.strip()
    return (
        "CONTESTO CONVERSAZIONE:\n" + "\n".join([f"- {x}" for x in last]) +
        "\n\nDOMANDA ATTUALE:\n" + current_q.strip()
    )


# =========================
# QUERY BUILDER
# =========================
def build_permessi_expansion_queries(q: str) -> list[str]:
    base = q.strip()
    return [
        f"{base} permessi visite mediche",
        f"{base} permessi lutto",
        f"{base} permessi matrimonio",
        f"{base} permessi legge 104",
        f"{base} permessi sindacali assemblea",
        f"{base} permessi diritto allo studio 150 ore",
        f"{base} permessi donazione sangue",
        f"{base} assenze retribuite elenco",
    ][:MAX_MULTI_QUERIES]

def build_queries(q: str) -> list[str]:
    q0 = q.strip()
    qs = [q0, f"{q0} CCNL", f"{q0} regole", f"{q0} articolo", f"{q0} condizioni"]

    if is_rol(q0):
        qs += [
            "ROL riduzione orario monte ore maturazione fruizione",
            "ex festività festività soppresse ore giorni spettanti",
        ]
    elif is_permessi(q0):
        qs += [
            "permessi retribuiti tipologie elenco",
            "assenze retribuite visite mediche lutto matrimonio 104 sindacali studio donazione sangue",
        ]

    if is_malattia(q0):
        qs += [
            "malattia trattamento economico percentuali integrazione",
            "malattia periodo di comporto conteggio",
            "certificato malattia obblighi comunicazione",
            "visita fiscale reperibilità fasce",
        ]

    if is_straord_notturno(q0):
        qs += [
            "lavoro notturno maggiorazione",
            "straordinario notturno maggiorazione",
            "straordinario maggiorazioni",
            "lavoro festivo domenicale maggiorazioni",
        ]

    if is_mansioni(q0):
        qs += [
            "mansioni superiori posto vacante 30 giorni 60 giorni",
            "non si applica sostituzione assente conservazione del posto",
            "formazione affiancamento non costituisce mansioni superiori",
        ]

    out, seen = [], set()
    for x in qs:
        x = x.strip()
        if x and x not in seen:
            out.append(x)
            seen.add(x)
    return out[:MAX_MULTI_QUERIES]


# =========================
# EVIDENCE EXTRACTION (internal proofs)
# =========================
EVID_PATTERNS = [
    r"\b30\b", r"\b60\b", r"\b\d{1,3}\b", r"%", r"percent",
    r"mansioni?\s+superiori?", r"posto\s+vacante", r"sostituzion",
    r"conservazion.*posto", r"diritto.*conservazion.*posto",
    r"non\s+si\s+applica", r"non\s+costituisc", r"affianc", r"formazion", r"addestr",
    r"malatt", r"comporto", r"certificat", r"reperibil", r"visita\s+fiscale",
    r"permess", r"\brol\b", r"ex\s*fest", r"festivit", r"lutto", r"matrimon", r"\b104\b",
    r"notturn", r"straordin", r"maggiorazion", r"festiv", r"domenic",
]

def extract_evidence_lines(chunks, max_lines=14):
    chunks = ensure_chunk_dicts(chunks)
    ev = []
    for c in chunks:
        page = c.get("page", "?")
        text = safe_str(c.get("text", ""))
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        for ln in lines:
            ln_low = ln.lower()
            if any(re.search(p, ln_low, flags=re.IGNORECASE) for p in EVID_PATTERNS):
                ln_clean = " ".join(ln.split())
                if 20 <= len(ln_clean) <= 420:
                    ev.append(f"[foglioPDF {page}] {ln_clean}")
    # dedup
    out, seen = [], set()
    for e in ev:
        if e not in seen:
            out.append(e)
            seen.add(e)
    return out[:max_lines]

def evidence_has_30_60(ev: list[str]) -> bool:
    joined = " ".join(ev).lower()
    return (re.search(r"\b30\b", joined) is not None) and (re.search(r"\b60\b", joined) is not None)

def evidence_mentions_only_straord_60(ev: list[str]) -> bool:
    joined = " ".join(ev).lower()
    # euristica: se "60%" vicino a "straordin" e "notturn" e NON trovi "lavoro notturno" come ordinario
    has60 = ("60%" in joined) or (re.search(r"\b60\s*%\b", joined) is not None)
    has_straord = "straordin" in joined
    has_notturn = "notturn" in joined
    # se non c'è una frase che parla chiaramente di "lavoro notturno" separato
    has_turno_notte = "lavoro notturno" in joined or "turno notturno" in joined
    return bool(has60 and has_straord and has_notturn and not has_turno_notte)


# =========================
# INDEX BUILD/LOAD
# =========================
def build_index():
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"Non trovo il PDF: {PDF_PATH} (metti 'ccnl.pdf' in /documenti)")
    os.makedirs(INDEX_DIR, exist_ok=True)

    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    texts = [c.page_content for c in chunks]
    pages = [(c.metadata.get("page", 0) + 1) for c in chunks]  # foglio PDF

    emb = OpenAIEmbeddings()
    vectors = emb.embed_documents(texts)
    vectors = np.array(vectors, dtype=np.float32)

    np.save(VEC_PATH, vectors)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump([{"page": p, "text": t} for p, t in zip(pages, texts)], f, ensure_ascii=False)

    return len(chunks)

@st.cache_resource(show_spinner=False)
def load_index_cached():
    vectors = np.load(VEC_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    meta = ensure_chunk_dicts(meta)
    mat_norm = normalize_rows(vectors)

    corpus_tokens = [tokenize(m.get("text", "")) for m in meta]
    bm25 = BM25Okapi(corpus_tokens)

    return vectors, mat_norm, meta, bm25


# =========================
# RETRIEVAL: Hybrid (Embedding + BM25)
# =========================
def hybrid_retrieve(enriched_q: str):
    vectors, mat_norm, meta, bm25 = load_index_cached()
    emb = OpenAIEmbeddings()

    queries = build_queries(enriched_q)

    scores = {}  # idx -> combined score
    debug_hits = {"queries": queries, "emb_top": [], "bm25_top": []}

    for q in queries:
        # Embedding scores
        qvec = np.array(emb.embed_query(q), dtype=np.float32)
        sims = cosine_scores(qvec, mat_norm)
        top_emb = np.argsort(-sims)[:TOP_K_PER_QUERY]
        debug_hits["emb_top"].append([int(i) for i in top_emb])

        # BM25 scores
        q_tokens = tokenize(q)
        bm = bm25.get_scores(q_tokens)
        bm = np.array(bm, dtype=np.float32)
        top_bm = np.argsort(-bm)[:TOP_K_PER_QUERY]
        debug_hits["bm25_top"].append([int(i) for i in top_bm])

        # Merge
        # normalize bm25 slice for stability
        bm_max = float(np.max(bm[top_bm])) if len(top_bm) else 1.0
        bm_max = bm_max if bm_max > 0 else 1.0

        for i in top_emb:
            s = float(sims[i])
            scores[i] = max(scores.get(i, 0.0), W_EMB * s)

        for i in top_bm:
            s = float(bm[i]) / bm_max
            scores[i] = max(scores.get(i, 0.0), scores.get(i, 0.0) + W_BM25 * s)

    # Topic boosts (small, safe)
    is_mans = is_mansioni(enriched_q)
    is_mal = is_malattia(enriched_q)
    is_perm = is_permessi(enriched_q)
    is_rol_q = is_rol(enriched_q)
    is_stra = is_straord_notturno(enriched_q)

    for i in list(scores.keys()):
        txt = safe_str(meta[i].get("text", "")).lower()
        boost = 0.0

        if is_mans:
            if re.search(r"\b30\b", txt) and re.search(r"\b60\b", txt):
                boost += 0.08
            if "conservazione del posto" in txt or "diritto alla conservazione" in txt:
                boost += 0.05
            if "non si applica" in txt or "non si applicano" in txt:
                boost += 0.05
            if "affianc" in txt or "formaz" in txt or "addestr" in txt:
                boost += 0.03
            if "posto vacante" in txt:
                boost += 0.03

        if is_stra:
            if "notturn" in txt:
                boost += 0.05
            if "straordin" in txt:
                boost += 0.05
            if "maggiorazion" in txt or "%" in txt:
                boost += 0.05

        if is_mal:
            if "comporto" in txt:
                boost += 0.05
            if "malatt" in txt:
                boost += 0.04
            if "certificat" in txt or "comunicaz" in txt:
                boost += 0.04
            if "reperibil" in txt or "visita fiscale" in txt:
                boost += 0.04
            if "%" in txt:
                boost += 0.04

        if is_perm and not is_rol_q:
            if "permess" in txt or "assenze retribuite" in txt:
                boost += 0.05
            for pats in PERMESSI_CATEGORIES.values():
                if any(re.search(p, txt, flags=re.IGNORECASE) for p in pats):
                    boost += 0.02
                    break

        if is_rol_q:
            if re.search(r"\brol\b", txt) or "riduzione orario" in txt:
                boost += 0.07
            if "ex festiv" in txt or "festività soppresse" in txt or "festivita soppresse" in txt:
                boost += 0.07
            # evita confusione con studio
            if "diritto allo studio" in txt or "150 ore" in txt:
                boost -= 0.05

        scores[i] = scores[i] + boost

    final_idx = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)[:TOP_K_FINAL]
    selected = [meta[i] for i in final_idx]

    # extra pass per permessi generici: se copertura bassa, espandi query e riprendi
    if is_perm and not is_rol_q:
        cov = permessi_coverage(selected)
        if cov < PERMESSI_MIN_CATEGORY_COVERAGE:
            extra_qs = build_permessi_expansion_queries(enriched_q)
            for q in extra_qs:
                q_tokens = tokenize(q)
                bm = bm25.get_scores(q_tokens)
                bm = np.array(bm, dtype=np.float32)
                top_bm = np.argsort(-bm)[:TOP_K_PER_QUERY]
                bm_max = float(np.max(bm[top_bm])) if len(top_bm) else 1.0
                bm_max = bm_max if bm_max > 0 else 1.0
                for i in top_bm:
                    scores[i] = scores.get(i, 0.0) + W_BM25 * (float(bm[i]) / bm_max)
            final_idx = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)[:TOP_K_FINAL]
            selected = [meta[i] for i in final_idx]

    return selected, debug_hits


# =========================
# LLM PROMPT (internal proof-based)
# =========================
BASE_RULES = (
    "Sei l’assistente UILCOM per lavoratori IPZS.\n"
    "Regole CRITICHE:\n"
    "1) Usa SOLO le EVIDENZE fornite. Non inventare numeri, percentuali, durate.\n"
    "2) Se una informazione non è dimostrata dalle evidenze, scrivi: 'Non emerge dal CCNL nel contesto recuperato'.\n"
    "3) Se nelle evidenze compaiono limitazioni/esclusioni (es. 'non si applica', sostituzione con conservazione del posto, affiancamento/formazione), riportale.\n"
    "4) Notturno: distingui SEMPRE 'lavoro notturno' (ordinario) da 'straordinario notturno'. Se il 60% è legato allo straordinario notturno, NON attribuirlo al notturno ordinario.\n"
    "5) Mansioni superiori: se compaiono 30 giorni continuativi / 60 discontinui, hanno priorità.\n"
    "6) Chiudi sempre con 'Consiglio pratico UILCOM' (1–2 bullet operativi brevi).\n\n"
    "Formato risposta:\n"
    "Risposta UILCOM:\n"
    "(2–5 righe)\n\n"
    "Dettagli operativi:\n"
    "- (bullet)\n\n"
    "Consiglio pratico UILCOM:\n"
    "- (bullet)\n\n"
    "Nota UILCOM:\n"
    "Questa risposta è informativa; per casi specifici verificare con RSU/UILCOM o HR e con il testo ufficiale.\n"
)

def build_prompt(user_input: str, enriched_q: str, evidences: list[str], ipzs_hint: bool, topic_note: str):
    ev_block = "\n".join([f"- {e}" for e in (evidences or [])])
    if not ev_block.strip():
        ev_block = "- (Nessuna evidenza estratta automaticamente dal contesto recuperato.)"

    ipzs_note = ""
    if ipzs_hint:
        ipzs_note = (
            "\nNOTA IPZS: se la risposta può dipendere da prassi/accordi aziendali (turni, ODS, procedure interne), "
            "segnalalo nel Consiglio pratico.\n"
        )

    return f"""
{BASE_RULES}
{topic_note}
{ipzs_note}

DOMANDA UTENTE:
{user_input}

DOMANDA ARRICCHITA (contesto conversazione):
{enriched_q}

EVIDENZE (PROVE) — devi basarti SOLO su queste:
{ev_block}

SCRIVI ORA LA RISPOSTA (seguendo il formato).
"""


# =========================
# TEST SUITE (admin)
# =========================
TESTS = [
    {
        "name": "Mansioni superiori (30/60 + esclusioni)",
        "q": "Dopo quanto scatta il livello se faccio mansioni superiori su posto vacante?",
        "must": ["30", "60"],
        "must_not": ["potresti", "forse", "probabilmente"],
    },
    {
        "name": "Sostituzione maternità (no scatto)",
        "q": "Sto sostituendo un collega in maternità: mi spetta il livello superiore?",
        "must": ["non", "sostituz"],
        "must_not": ["60% lavoro notturno"],
    },
    {
        "name": "Notturno vs straordinario notturno",
        "q": "Quanto viene pagato il lavoro notturno?",
        "must": ["notturn"],
        "must_not": ["60% (se non provato)"],
    },
    {
        "name": "Straordinario notturno",
        "q": "Quanto viene pagato lo straordinario notturno?",
        "must": ["straordin", "notturn"],
        "must_not": [],
    },
    {
        "name": "Permessi lutto",
        "q": "Permessi per lutto: quanti giorni?",
        "must": ["lutto"],
        "must_not": ["non emerge (se invece c'è prova)"],
    },
]


# =========================
# ADMIN PANEL
# =========================
if SHOW_ADMIN:
    with st.expander("🧩 Admin tools", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("Indicizza CCNL", use_container_width=True):
                try:
                    with st.spinner("Indicizzazione in corso..."):
                        n = build_index()
                    # reset cache
                    load_index_cached.clear()
                    st.success(f"Indicizzazione completata: {n} chunk")
                except Exception as e:
                    st.error(str(e))
        with c2:
            st.write("Indice presente:", "✅" if has_index() else "❌")
        with c3:
            if st.button("Reset chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()

        st.divider()
        st.caption("Test suite (non perfetta, ma ti dà subito FAIL/PASS dopo ogni modifica).")
        if st.button("Esegui test", use_container_width=True):
            if not has_index():
                st.error("Indice mancante: indicizza prima.")
            else:
                fails = []
                for t in TESTS:
                    try:
                        sel, _dbg = hybrid_retrieve(t["q"])
                        ev = extract_evidence_lines(sel)
                        topic_note = ""
                        if "notturn" in t["q"].lower():
                            if evidence_mentions_only_straord_60(ev):
                                topic_note += "\nGUARDRAIL: Nelle evidenze il 60% sembra riferito allo straordinario notturno. Non attribuirlo al notturno ordinario.\n"
                        prompt = build_prompt(t["q"], t["q"], ev, False, topic_note)
                        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
                        ans = llm.invoke(prompt).content
                        low = ans.lower()

                        ok = True
                        for m in t.get("must", []):
                            if m.lower() not in low:
                                ok = False
                        for mn in t.get("must_not", []):
                            if mn.lower() in low:
                                ok = False

                        if not ok:
                            fails.append((t["name"], ans))
                    except Exception as e:
                        fails.append((t["name"], f"Errore: {e}"))

                if not fails:
                    st.success("✅ Tutti i test PASS")
                else:
                    st.error(f"❌ Test FAIL: {len(fails)}")
                    for name, out in fails[:6]:
                        with st.expander(f"FAIL: {name}", expanded=False):
                            st.write(out)


# =========================
# CHAT
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

        # debug only admin
        if SHOW_ADMIN and m.get("role") == "assistant" and m.get("debug"):
            with st.expander("🛠 Debug (solo admin)"):
                dbg = m["debug"]
                st.write("Queries:", dbg.get("queries"))
                st.write("Selezione fogli (PDF):", dbg.get("selected_pages"))
                st.write("Evidenze:", dbg.get("evidences"))
                st.text_area("Prompt", dbg.get("prompt", ""), height=220)
                st.text_area("Risposta grezza", dbg.get("raw_answer", ""), height=220)

user_input = st.chat_input("Scrivi una domanda sul CCNL...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    if not has_index():
        st.session_state.messages.append(
            {"role": "assistant", "content": "Non posso rispondere: il CCNL non è ancora indicizzato (admin: apri con ?admin=1 e indicizza)."}
        )
        st.rerun()

    enriched_q = build_enriched_question(user_input)

    # retrieval
    selected, dbg_hits = hybrid_retrieve(enriched_q)
    selected = ensure_chunk_dicts(selected)

    evidences = extract_evidence_lines(selected)

    # topic guardrails
    topic_note = ""
    if is_mansioni(enriched_q) and evidence_has_30_60(evidences):
        topic_note += "\nGUARDRAIL MANSIONI: nelle evidenze compaiono 30/60: usali come riferimento principale.\n"

    if is_straord_notturno(enriched_q) and ("notturn" in enriched_q.lower()):
        if evidence_mentions_only_straord_60(evidences):
            topic_note += (
                "\nGUARDRAIL NOTTURNO: dalle evidenze il 60% risulta legato allo straordinario notturno. "
                "Non attribuire il 60% al lavoro notturno ordinario se non dimostrato.\n"
            )

    ipzs_hint = is_ipzs(enriched_q)

    prompt = build_prompt(user_input, enriched_q, evidences, ipzs_hint, topic_note)

    # LLM
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        raw_answer = llm.invoke(prompt).content
        answer = raw_answer.strip()
    except Exception as e:
        answer = f"Errore nel generare la risposta: {e}"
        raw_answer = answer

    debug_payload = None
    if SHOW_ADMIN:
        debug_payload = {
            "queries": dbg_hits.get("queries", []),
            "selected_pages": [c.get("page", "?") for c in selected],
            "evidences": evidences,
            "prompt": prompt,
            "raw_answer": raw_answer,
        }

    # USER: clean output (no fonti)
    st.session_state.messages.append({"role": "assistant", "content": answer, "debug": debug_payload})
    st.rerun()
