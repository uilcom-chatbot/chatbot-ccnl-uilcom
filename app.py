# app.py — Assistente Contrattuale UILCOM IPZS (CCNL + Indice IPZS Permessi)
# ✅ Risposte SOLO dai documenti caricati (CCNL PDF + IPZS Permessi TXT)
# ✅ Pubblico: include SEMPRE citazioni (pagine/schede)
# ✅ Admin: debug + evidenze + chunk/pagine usate
# ✅ Topic reset: se cambia argomento, NON usa memoria breve (evita contaminazioni)
# ✅ Guardrail HARD: se retrieval debole -> "Non ho trovato..."
# ✅ Mansioni superiori: risposta deterministica (NO LLM) + pagine
# ✅ Migliorie 2026-02 (rev3):
#    - Matching robusto apostrofi tipografici (’)
#    - Query “chiave” per mansioni superiori
#    - Fallback retrieval mirato se manca “trattamento corrispondente”
#    - Eccezione “sostituzione/conservazione del posto” mostrata SOLO se l’utente chiede passaggio livello/definitività (30/60)

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

# CCNL
PDF_PATH = os.path.join("documenti", "ccnl.pdf")
INDEX_DIR = "index_ccnl"
VEC_PATH = os.path.join(INDEX_DIR, "vectors.npy")
META_PATH = os.path.join(INDEX_DIR, "chunks.json")

# IPZS Permessi (TXT da screenshot)
IPZS_TXT_PATH = os.path.join("documenti", "PERMESSI_IPZS_COMPLETO_FINALE.txt")
INDEX_DIR_IPZS = "index_ipzs_permessi"
VEC_PATH_IPZS = os.path.join(INDEX_DIR_IPZS, "vectors.npy")
META_PATH_IPZS = os.path.join(INDEX_DIR_IPZS, "chunks.json")

# Chunking
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150

IPZS_CHUNK_SIZE = 1000
IPZS_CHUNK_OVERLAP = 120

# Retrieval
TOP_K_PER_QUERY = 10
TOP_K_FINAL = 18
MAX_MULTI_QUERIES = 8

# Dedup / diversity
MAX_CHUNKS_PER_PAGE = 3
NEAR_DUP_JACCARD = 0.92

# Memoria: usata SOLO se stesso argomento (topic)
MEMORY_USER_TURNS = 3

# Hard guardrail retrieval
MIN_BEST_SIMILARITY = 0.245
MIN_AVG_TOP3 = 0.220
MIN_SPREAD = 0.020
MIN_SELECTED_CHUNKS = 3

# Admin debug
MAX_EVIDENCE_LINES = 18

# LLM
LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0


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
ADMIN_PASSWORD = get_secret("ADMIN_PASSWORD")
OPENAI_API_KEY = get_secret("OPENAI_API_KEY")


# ============================================================
# PAGE SETUP
# ============================================================
st.set_page_config(page_title="Assistente UILCOM IPZS", page_icon="🟦", layout="centered")
st.title(APP_TITLE)
st.markdown(
    "**Accesso riservato agli iscritti UILCOM**  \n"
    "Strumento informativo per facilitare la consultazione del **CCNL Grafici Editoria** "
    "e delle **schede permessi IPZS**.  \n\n"
    "⚠️ Le risposte sono generate **solo** in base ai documenti caricati. "
    "Le citazioni (pagine/schede) sono incluse per permettere la verifica diretta. "
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

if not st.session_state.auth_ok:
    st.stop()


# ============================================================
# ADMIN MODE (debug)
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
        "Locale: variabile d’ambiente OPENAI_API_KEY"
    )
    st.stop()


# ============================================================
# TEXT NORMALIZATION (apostrophes etc.)
# ============================================================
def normalize_text_for_match(s: str) -> str:
    if not s:
        return ""
    s = s.replace("’", "'").replace("“", '"').replace("”", '"')
    s = re.sub(r"\s+", " ", s)
    return s.lower()


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
    return re.findall(r"[a-zàèéìòù0-9]+", (s or "").lower())


def _jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))


def _dedup_near_identical(chunks: List[Dict[str, Any]], thr: float = NEAR_DUP_JACCARD) -> List[Dict[str, Any]]:
    kept: List[Dict[str, Any]] = []
    kept_tok: List[List[str]] = []
    for c in chunks:
        tok = _tokenize_light(c.get("text", ""))
        if any(_jaccard(tok, kt) >= thr for kt in kept_tok):
            continue
        kept.append(c)
        kept_tok.append(tok)
    return kept


def _limit_per_page(chunks: List[Dict[str, Any]], max_per_page: int = MAX_CHUNKS_PER_PAGE) -> List[Dict[str, Any]]:
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
# TRIGGERS / TOPIC
# ============================================================
MANSIONI_TRIGGERS = [
    "mansioni superiori", "mansione superiore", "mansioni più alte", "mansioni piu alte",
    "categoria superiore", "livello superiore", "passaggio di categoria", "cambio categoria",
    "inquadramento superiore", "posto vacante", "sostituzione", "sto sostituendo",
    "differenza paga", "differenza di paga", "pagato di più", "pagato di piu",
    "trattamento corrispondente", "retribuzione corrispondente",
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


def detect_topic(q: str) -> str:
    ql = q.lower()
    if is_malattia_question(ql):
        return "malattia"
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
    qlow = q0.lower()
    qs = [q0, f"{q0} CCNL regola"]

    user_is_rol = is_rol_exfest_question(q0)
    user_is_perm = is_permessi_question(q0)
    user_is_mal = is_malattia_question(q0)
    user_is_mans = is_mansioni_question(q0)

    if user_is_rol:
        qs += [
            "RAO festività infrasettimanali abolite riposi retribuiti",
            "ROL riduzione orario di lavoro maturazione fruizione monte ore",
        ]
    if user_is_perm and (not user_is_rol):
        qs += [
            "permessi retribuiti tipologie elenco",
            "permesso studio una settimana l'anno",
            "donazione sangue permesso retribuito",
            "legge 104 art 33 comma 3 permesso",
        ]
    if user_is_mal:
        qs += [
            "malattia trattamento economico integrazione",
            "malattia periodo di comporto conservazione posto",
            "malattia visite fiscali reperibilità fasce",
        ]
    if any(t in qlow for t in STRAORDINARI_TRIGGERS):
        qs += [
            "lavoro straordinario maggiorazioni limiti",
            "straordinario notturno maggiorazione",
            "lavoro notturno maggiorazione",
            "lavoro festivo maggiorazioni",
        ]
    if user_is_mans:
        qs += [
            "mansioni superiori 30 giorni consecutivi 60 giorni non consecutivi",
            "mansioni superiori trattamento corrispondente all'attività svolta",
            "sostituzione lavoratore assente conservazione del posto",
            "ha diritto al trattamento corrispondente all’attività svolta 30 giorni 60 giorni non si applica sostituzione conservazione del posto",
        ]

    out, seen = [], set()
    for x in qs:
        x = x.strip()
        if x and x not in seen:
            out.append(x)
            seen.add(x)
    return out[:MAX_MULTI_QUERIES]


# ============================================================
# EVIDENCE EXTRACTION
# ============================================================
def extract_key_evidence(chunks: List[Dict[str, Any]], source: str) -> List[str]:
    patterns = [
        r"\b30\b", r"\b60\b", r"mansioni?\s+superiori?",
        r"sostituzion", r"conservazion.*posto",
        r"ha\s+diritto\s+al\s+trattamento", r"trattamento\s+corrispondente",
        r"\brao\b", r"\brol\b",
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
# OPTIONAL BM25 RERANK
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
def extract_mansioni_rules(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    txt_all = " ".join([normalize_text_for_match(c.get("text") or "") for c in chunks])

    found_30 = re.search(r"\b30\b", txt_all) is not None and re.search(r"(giorn|gg)", txt_all) is not None
    found_60 = re.search(r"\b60\b", txt_all) is not None and re.search(r"(giorn|gg)", txt_all) is not None

    has_trattamento = re.search(r"ha\s+diritto\s+al\s+trattamento\s+corrispondente|trattamento\s+corrispondente", txt_all) is not None
    has_esclusione = re.search(r"non\s+si\s+applica.*sostituzion|sostituzion.*conservazion|diritto.*conservazion.*posto", txt_all) is not None
    has_formazione = (re.search(r"formazion|addestramento|affiancamento", txt_all) is not None) and (re.search(r"non\s+costituisc", txt_all) is not None)

    pages = set()
    for c in chunks:
        try:
            pages.add(int(c.get("page", 0)))
        except Exception:
            pass

    return {
        "found_30": found_30,
        "found_60": found_60,
        "has_trattamento": has_trattamento,
        "has_esclusione": has_esclusione,
        "has_formazione": has_formazione,
        "pages": sorted([p for p in pages if p]),
    }


def mansioni_public_answer(user_q: str, rules: Dict[str, Any]) -> str:
    ql = user_q.lower()

    # ✅ eccezione SOLO se domanda su stabilizzazione/passaggio livello/30-60
    ask_stabilizzazione = any(x in ql for x in [
        "categoria", "livello", "passaggio", "inquadramento", "definitiv",
        "stabilizz", "30", "60", "giorni", "matur", "diviene definitiva"
    ])

    diff_paga = rules.get("has_trattamento", False)
    has_30_60 = bool(rules.get("found_30", False) and rules.get("found_60", False))

    parts: List[str] = []

    # Differenza paga (trattamento)
    if any(x in ql for x in ["pagato", "pagata", "differenza", "trattamento", "retribuzione", "piu", "più"]):
        if diff_paga:
            parts.append(
                "Nel caso di assegnazione a **mansioni superiori**, il dipendente ha diritto al **trattamento corrispondente all’attività svolta** "
                "(quindi alla differenza retributiva per il periodo in cui svolge quelle mansioni)."
            )
        else:
            parts.append("Non ho trovato la regola sul trattamento economico per mansioni superiori nei documenti caricati (nel materiale recuperato).")

    # Stabilizzazione (30/60) + eccezione solo qui
    if ask_stabilizzazione:
        if has_30_60:
            parts.append(
    "La categoria superiore si **matura** quando il lavoratore viene adibito a mansioni di livello più alto "
    "per un determinato periodo di tempo.\n\n"
    "In particolare, l’assegnazione diventa **definitiva** dopo:\n"
    "- **30 giorni consecutivi** di svolgimento delle mansioni superiori;\n"
    "- **60 giorni complessivi**, anche non continuativi."
)
        else:
            parts.append("Non ho trovato nei documenti caricati la regola specifica sui giorni (30/60) per la definitività (nel materiale recuperato).")

        if rules.get("has_esclusione", False):
            parts.append(
    "⚠️ **Eccezione:** questi termini non si applicano se l’assegnazione avviene per **sostituzione** "
    "di un dipendente assente con **diritto alla conservazione del posto**. "
    "In questo caso non matura il passaggio definitivo di categoria."
)

    # Nota formazione (neutra)
    if rules.get("has_formazione", False):
        parts.append(
            "ℹ️ **Nota:** i periodi di **formazione/addestramento in affiancamento** a dipendenti di categorie più elevate **non costituiscono** assegnazione a mansioni superiori."
        )

    if not parts:
        if diff_paga and has_30_60:
            parts.append("Per mansioni superiori: diritto al **trattamento corrispondente** e definitività dopo **30 giorni continuativi** o **60 non continuativi**.")
        elif diff_paga:
            parts.append("Per mansioni superiori: diritto al **trattamento corrispondente** all’attività svolta.")
        else:
            parts.append("Non ho trovato la risposta nei documenti caricati.")

    cit = format_public_citations("CCNL", rules.get("pages", []) or [])
    if cit:
        parts.append(cit)

    return "\n\n".join(parts).strip()


# ============================================================
# SYSTEM RULES (LLM)
# ============================================================
RULES_PUBLIC = """
Sei l’assistente UILCOM per lavoratori IPZS.
Devi rispondere in modo chiaro e pratico basandoti SOLO sul contesto fornito (estratti dai documenti indicizzati).
Non inventare informazioni.

REGOLE IMPORTANTI:
1) Se non trovi nel contesto, scrivi: "Non ho trovato la risposta nei documenti caricati."
2) NON confondere lavoro notturno con straordinario notturno.
3) TERMINOLOGIA EX FESTIVITÀ: se trovi "festività soppresse/abolite/infrasettimanali abolite" spiega che è la dicitura equivalente.
4) Permessi: elenca SOLO le tipologie che trovi nel contesto.
5) Devi SEMPRE includere una riga finale con la fonte:
   - "Fonte: CCNL (pag. ...)" oppure "Fonte: IPZS Permessi (scheda ...)".

FORMATO:
Risposta diretta e verificabile.
"""


# ============================================================
# IPZS TXT SPLIT (schede)
# ============================================================
def split_ipzs_blocks(raw_txt: str) -> List[str]:
    txt = (raw_txt or "").replace("\r\n", "\n")
    lines = txt.split("\n")

    sep_idx = [i for i, ln in enumerate(lines) if re.fullmatch(r"[\-\=\*]{5,}", ln.strip() or "") is not None]
    if len(sep_idx) >= 1:
        blocks = []
        start = 0
        for i in sep_idx:
            block = "\n".join(lines[start:i]).strip()
            if len(block) >= 120:
                blocks.append(block)
            start = i + 1
        tail = "\n".join(lines[start:]).strip()
        if len(tail) >= 120:
            blocks.append(tail)
        if blocks:
            return blocks

    starts = []
    for i, ln in enumerate(lines):
        s = (ln or "").strip()
        if not s:
            continue
        if 4 <= len(s) <= 90 and re.fullmatch(r"[A-Z0-9ÀÈÉÌÒÙ\.\-\/\(\)\s]+", s) is not None:
            nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
            nxt2 = lines[i + 2].strip() if i + 2 < len(lines) else ""
            cond = (nxt == "") or (nxt and re.fullmatch(r"[A-Z0-9ÀÈÉÌÒÙ\.\-\/\(\)\s]+", nxt) is None)
            cond = cond or (nxt == "" and nxt2 != "")
            if cond:
                starts.append(i)

    if len(starts) >= 2:
        blocks = []
        for k in range(len(starts)):
            a = starts[k]
            b = starts[k + 1] if k + 1 < len(starts) else len(lines)
            block = "\n".join(lines[a:b]).strip()
            if len(block) >= 120:
                blocks.append(block)
        if blocks:
            return blocks

    return [txt.strip()]


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
        st.caption("ADMIN_PASSWORD non impostata (Secrets).")

    st.divider()

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
                pages = [(int(c.metadata.get("page", 0)) + 1) for c in chunks]

                emb = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
                vectors = np.array(emb.embed_documents(texts), dtype=np.float32)

                np.save(VEC_PATH, vectors)
                with open(META_PATH, "w", encoding="utf-8") as f:
                    json.dump([{"page": p, "text": t} for p, t in zip(pages, texts)], f, ensure_ascii=False)

            st.success(f"Indicizzazione CCNL completata. Chunk: {len(chunks)}")
            st.rerun()
        except Exception as e:
            st.error(str(e))

    st.divider()

    st.subheader("📦 Indice IPZS (permessi)")
    ok_ipzs = os.path.exists(VEC_PATH_IPZS) and os.path.exists(META_PATH_IPZS)
    st.write("Indice IPZS presente:", "✅" if ok_ipzs else "❌")

    if st.button("Indicizza / Reindicizza IPZS (permessi)", use_container_width=True):
        try:
            with st.spinner("Indicizzazione IPZS in corso..."):
                if not os.path.exists(IPZS_TXT_PATH):
                    raise FileNotFoundError(f"Non trovo il file: {IPZS_TXT_PATH} (metti il TXT in /documenti)")
                os.makedirs(INDEX_DIR_IPZS, exist_ok=True)

                with open(IPZS_TXT_PATH, "r", encoding="utf-8") as f:
                    raw_txt = f.read()

                blocks = split_ipzs_blocks(raw_txt)
                splitter = RecursiveCharacterTextSplitter(chunk_size=IPZS_CHUNK_SIZE, chunk_overlap=IPZS_CHUNK_OVERLAP)

                chunks: List[Dict[str, Any]] = []
                scheda = 0
                for b in blocks:
                    scheda += 1
                    parts = splitter.split_text(b)
                    for p in parts:
                        chunks.append({"page": scheda, "text": p})

                texts = [c["text"] for c in chunks]
                emb = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
                vectors_ipzs = np.array(emb.embed_documents(texts), dtype=np.float32)

                np.save(VEC_PATH_IPZS, vectors_ipzs)
                with open(META_PATH_IPZS, "w", encoding="utf-8") as f:
                    json.dump(chunks, f, ensure_ascii=False)

            st.success(f"Indicizzazione IPZS completata. Schede: {len(blocks)} — Chunk: {len(chunks)}")
            st.rerun()
        except Exception as e:
            st.error(str(e))

    st.divider()

    if st.button("🧹 Nuova chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_topic = None
        st.rerun()


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
user_input = st.chat_input("Scrivi una domanda (permessi, ROL/RAO, malattia, straordinari, mansioni superiori...)")
if not user_input:
    st.stop()

st.session_state.messages.append({"role": "user", "content": user_input})


# ============================================================
# SCEGLI SORGENTE
# ============================================================
topic = detect_topic(user_input)
enriched_q = build_enriched_question(user_input, topic)

use_ipzs = topic in ("permessi", "rol_exfest")
source = "IPZS" if use_ipzs else "CCNL"

if source == "CCNL":
    if not (os.path.exists(VEC_PATH) and os.path.exists(META_PATH)):
        st.session_state.messages.append({"role": "assistant", "content": "Prima devo indicizzare il CCNL: apri la barra laterale e clicca **Indicizza / Reindicizza CCNL**."})
        st.rerun()
else:
    if not (os.path.exists(VEC_PATH_IPZS) and os.path.exists(META_PATH_IPZS)):
        st.session_state.messages.append({"role": "assistant", "content": "Prima devo indicizzare le **schede IPZS (permessi)**: apri la barra laterale e clicca **Indicizza / Reindicizza IPZS (permessi)**."})
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

ranked_idx = sorted(scores_best.keys(), key=lambda i: scores_best[i], reverse=True)

candidates: List[Dict[str, Any]] = []
cand_scores: List[float] = []
for i in ranked_idx[: max(TOP_K_FINAL * 3, TOP_K_FINAL) ]:
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
selected_scores = tmp_scores[: len(selected)]

selected = bm25_rerank(enriched_q, selected)

key_evidence = extract_key_evidence(selected, source)
confidence = compute_retrieval_confidence(selected_scores)

retrieval_ok = (
    confidence["best"] >= MIN_BEST_SIMILARITY
    and confidence["avg_top3"] >= MIN_AVG_TOP3
    and confidence["spread"] >= MIN_SPREAD
    and len(selected) >= MIN_SELECTED_CHUNKS
)

public_pages = unique_pages(selected, max_pages=8)
public_cit_line = format_public_citations(source, public_pages)

def hard_not_found_message() -> str:
    return "Non ho trovato la risposta nei documenti caricati."


# ============================================================
# MANSIONI: deterministico + fallback
# ============================================================
if topic == "mansioni":
    if source != "CCNL":
        st.session_state.messages.append({"role": "assistant", "content": "Questa domanda riguarda **mansioni superiori**: la tratto sul **CCNL**. Indicizza il CCNL e riprova."})
        st.rerun()

    if retrieval_ok:
        rules_probe = extract_mansioni_rules(selected)
        if not rules_probe.get("has_trattamento", False):
            extra_q = "ha diritto al trattamento corrispondente all’attività svolta mansioni superiori 30 giorni 60 giorni non si applica sostituzione conservazione del posto"
            qvec2 = np.array(emb.embed_query(extra_q), dtype=np.float32)
            sims2 = cosine_scores(qvec2, mat_norm)
            top2 = np.argsort(-sims2)[:10]
            extra = [meta[int(i)] for i in top2]
            extra = _limit_per_page(_dedup_near_identical(extra), max_per_page=2)
            selected = _dedup_near_identical(selected + extra)

        rules_m = extract_mansioni_rules(selected)
        public_ans = mansioni_public_answer(user_input, rules_m)
    else:
        public_ans = hard_not_found_message()
        rules_m = {}

    payload = {"role": "assistant", "content": public_ans}
    if st.session_state.is_admin:
        payload["debug"] = {
            "topic": topic,
            "queries": queries,
            "confidence": confidence,
            "evidence": key_evidence,
            "pages": rules_m.get("pages") if isinstance(rules_m, dict) else None,
        }

    st.session_state.last_topic = topic
    st.session_state.messages.append(payload)
    st.rerun()


# ============================================================
# LLM per gli altri topic
# ============================================================
if not retrieval_ok:
    payload = {"role": "assistant", "content": hard_not_found_message()}
    if st.session_state.is_admin:
        payload["debug"] = {"topic": topic, "queries": queries, "confidence": confidence, "evidence": key_evidence}
    st.session_state.last_topic = topic
    st.session_state.messages.append(payload)
    st.rerun()

context = "\n\n---\n\n".join(
    [f"[{'Scheda' if source=='IPZS' else 'Pagina'} {c.get('page','?')}] {c.get('text','')}" for c in selected]
)
evidence_block = "\n".join([f"- {e}" for e in key_evidence]) if key_evidence else "- (Nessuna evidenza estratta automaticamente.)"

guardrail_notturno = ""
if is_lavoro_notturno_question(enriched_q):
    guardrail_notturno = "GUARDRAIL NOTTURNO: domanda su lavoro notturno ordinario. NON usare percentuali straordinario notturno.\n"
elif is_straordinario_notturno_question(enriched_q):
    guardrail_notturno = "GUARDRAIL STRAORD. NOTTURNO: domanda su straordinario notturno. Usa SOLO le percentuali relative allo straordinario notturno.\n"

llm = ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE, api_key=OPENAI_API_KEY)

prompt_public = f"""
{RULES_PUBLIC}

SORGENTE ATTIVA: {source}
{guardrail_notturno}

DOMANDA (UTENTE):
{user_input}

DOMANDA ARRICCHITA (MEMORIA BREVE - solo stesso topic):
{enriched_q}

EVIDENZE:
{evidence_block}

CONTESTO:
{context}

RICORDA:
- Chiudi SEMPRE con una riga "Fonte: ..." coerente con la sorgente.
"""

try:
    public_raw = llm.invoke(prompt_public).content
except Exception as e:
    public_raw = f"Errore nel generare la risposta: {e}"

if not re.search(r"\bfonte\b\s*:", (public_raw or ""), flags=re.IGNORECASE):
    public_raw = (public_raw or "").rstrip() + "\n\n" + public_cit_line

st.session_state.last_topic = topic
st.session_state.messages.append({"role": "assistant", "content": (public_raw or "").strip()})
st.rerun()


