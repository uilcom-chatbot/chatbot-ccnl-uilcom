# app.py — Assistente Contrattuale UILCOM IPZS (definitiva + FORCED RETRIEVAL NOTTURNO ORDINARIO)
# ✅ Risposte SOLO dal CCNL
# ✅ Utenti: risposta pulita (senza fonti)
# ✅ Admin: debug + evidenze + chunk/pagine usate
# ✅ Fix: ex festività = festività soppresse/abolite/infrasettimanali abolite
# ✅ Fix: mansioni superiori (30/60 + posto vacante + esclusione conservazione posto)
# ✅ Fix: lavoro notturno vs straordinario notturno (NON confondere %)
# ✅ FIX FINALE: se nel CCNL esiste % del NOTTURNO ORDINARIO, viene SEMPRE recuperata (forced retrieval)
# ✅ Indice vettoriale persistente (vectors.npy + chunks.json)

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
PERMESSI_MIN_CATEGORY_COVERAGE = 3
MAX_EVIDENCE_LINES = 18

LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0

# Forced retrieval: quante "prove" forzare nel contesto se parliamo di NOTTURNO ORDINARIO
FORCED_NOTTURNO_TOPN = 6


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

UILCOM_PASSWORD = get_secret("UILCOM_PASSWORD")
ADMIN_PASSWORD = get_secret("ADMIN_PASSWORD")
OPENAI_API_KEY = get_secret("OPENAI_API_KEY")


# ============================================================
# UI
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
    st.warning("Password iscritti non impostata. Imposta UILCOM_PASSWORD in Secrets (Streamlit) o variabile d’ambiente.")

if not st.session_state.auth_ok:
    st.stop()


# ============================================================
# ADMIN (debug)
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

    st.subheader("📦 Indice CCNL")
    ok_index = os.path.exists(VEC_PATH) and os.path.exists(META_PATH)
    st.write("Indice presente:", "✅" if ok_index else "❌")

    if st.button("Indicizza / Reindicizza CCNL", use_container_width=True):
        try:
            with st.spinner("Indicizzazione in corso..."):
                if not os.path.exists(PDF_PATH):
                    raise FileNotFoundError(f"Non trovo il PDF: {PDF_PATH} (metti ccnl.pdf in /documenti)")

                os.makedirs(INDEX_DIR, exist_ok=True)

                loader = PyPDFLoader(PDF_PATH)
                docs = loader.load()

                splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
                chunks = splitter.split_documents(docs)

                texts = [c.page_content for c in chunks]
                pages = [(int(c.metadata.get("page", 0)) + 1) for c in chunks]  # foglio PDF

                if not OPENAI_API_KEY:
                    raise RuntimeError("Manca OPENAI_API_KEY in Secrets/variabili ambiente.")

                emb = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
                vectors = np.array(emb.embed_documents(texts), dtype=np.float32)

                np.save(VEC_PATH, vectors)
                with open(META_PATH, "w", encoding="utf-8") as f:
                    json.dump([{"page": p, "text": t} for p, t in zip(pages, texts)], f, ensure_ascii=False)

            st.success(f"Indicizzazione completata. Chunk: {len(chunks)}")
            st.rerun()
        except Exception as e:
            st.error(str(e))

    if st.button("🧹 Nuova chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("Dopo commit su GitHub Streamlit Cloud di solito fa auto-redeploy. Se no: **Reboot app**.")


# ============================================================
# OPENAI KEY REQUIRED
# ============================================================
if not OPENAI_API_KEY:
    st.error(
        "Manca **OPENAI_API_KEY**.\n\n"
        "Streamlit Cloud: Settings → Secrets → OPENAI_API_KEY\n"
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

    fixed: List[Dict[str, Any]] = []
    for item in meta:
        if isinstance(item, dict) and "text" in item and "page" in item:
            fixed.append({"page": item.get("page", "?"), "text": item.get("text", "")})
        elif isinstance(item, str):
            fixed.append({"page": "?", "text": item})
        else:
            fixed.append({"page": "?", "text": str(item)})
    return vectors, fixed


# ============================================================
# CLASSIFIERS
# ============================================================
def is_straordinario_notturno_question(q: str) -> bool:
    ql = q.lower()
    return ("straordin" in ql) and ("notturn" in ql)

def is_lavoro_notturno_question(q: str) -> bool:
    ql = q.lower()
    return ("notturn" in ql) and ("straordin" not in ql)

def is_straordinario_question(q: str) -> bool:
    ql = q.lower()
    return ("straordin" in ql) or ("maggioraz" in ql) or ("oltre orario" in ql) or ("supplementare" in ql)

def is_permessi_question(q: str) -> bool:
    ql = q.lower()
    return any(t in ql for t in [
        "permessi", "permesso", "assenze retribuite", "permessi retribuiti",
        "rol", "ex festiv", "festività soppresse", "festivita soppresse",
        "studio", "esami", "104", "assemblea", "sindac", "donazione", "lutto", "matrimonio"
    ])

def is_rol_exfest_question(q: str) -> bool:
    ql = q.lower()
    return any(t in ql for t in [
        "rol", "r.o.l", "riduzione orario",
        "ex festiv", "ex-festiv", "exfestiv",
        "festività soppresse", "festivita soppresse",
        "festività abolite", "festivita abolite",
        "festività infrasettimanali abolite", "festivita infrasettimanali abolite",
    ])


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
# QUERY BUILDER
# ============================================================
def build_queries(q: str) -> List[str]:
    q0 = q.strip()
    qlow = q0.lower()

    qs = [q0, f"{q0} CCNL", f"{q0} regole condizioni", f"{q0} definizione procedura"]

    user_is_notturno = is_lavoro_notturno_question(q0)
    user_is_straord_notturno = is_straordinario_notturno_question(q0)
    user_is_straord = is_straordinario_question(q0)
    user_is_perm = is_permessi_question(q0)
    user_is_rol = is_rol_exfest_question(q0)

    # NOTTURNO ORDINARIO (molto specifico)
    if user_is_notturno and (not user_is_straord_notturno):
        qs += [
            "lavoro notturno ordinario maggiorazione percentuale",
            "turno notturno maggiorazione retribuzione",
            "indennità lavoro notturno percentuale",
            "maggiorazione lavoro notturno (non straordinario)",
            "definizione lavoro notturno fascia oraria",
        ]

    # STRAORDINARI
    if user_is_straord or ("maggioraz" in qlow) or ("straordin" in qlow):
        qs += [
            "lavoro straordinario maggiorazioni percentuali CCNL",
            "percentuali maggiorazione straordinario diurno",
            "percentuali maggiorazione straordinario notturno",
            "straordinario festivo maggiorazione percentuale",
            "tabella maggiorazioni lavoro straordinario",
        ]

    # Permessi + ROL/ex fest
    if user_is_perm and (not user_is_rol):
        qs += [
            "permessi retribuiti tipologie elenco",
            "assenze retribuite tipologie",
            "assemblea sindacale ore retribuite",
            "diritto allo studio 150 ore triennio",
            "ROL riduzione orario festività soppresse abolite",
        ]

    if user_is_rol:
        qs += [
            "ROL riduzione orario di lavoro monte ore annuo maturazione fruizione",
            "festività soppresse abolite riposi retribuiti",
            "festività infrasettimanali abolite riposi retribuiti",
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
def extract_key_evidence(chunks: List[Dict[str, Any]]) -> List[str]:
    patterns = [
        r"\b30\b", r"\b60\b", r"\b\d{1,3}\b", r"%", r"maggior", r"indennit",
        r"notturn", r"straordin", r"festiv",
        r"\brol\b", r"festivit", r"soppresse", r"abolite", r"infrasettimanali",
        r"permess", r"assenze?\s+retribuit",
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

def evidence_has_notturno_ordinario_percent(evidence_lines: List[str]) -> bool:
    for e in evidence_lines:
        el = e.lower()
        if ("notturn" in el) and ("%" in el) and ("straordin" not in el):
            return True
    return False


# ============================================================
# BM25 RERANK (optional)
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
# ✅ FORCED RETRIEVAL: NOTTURNO ORDINARIO % (deterministico su TUTTO il CCNL)
# Cerca chunk con: notturn + % e NON straordin
# ============================================================
def forced_find_notturno_ordinario_chunks(meta: List[Dict[str, Any]], topn: int = 6) -> List[Dict[str, Any]]:
    scored: List[Tuple[int, int]] = []  # (score, idx)
    for idx, c in enumerate(meta):
        txt = (c.get("text") or "").lower()
        if "notturn" in txt and "%" in txt and "straordin" not in txt:
            # score grezzo: quante volte compaiono parole chiave
            score = txt.count("notturn") * 3 + txt.count("%") * 2 + txt.count("maggior") + txt.count("indennit")
            scored.append((score, idx))

    scored.sort(key=lambda x: x[0], reverse=True)
    picked = [meta[i] for _, i in scored[:topn]]
    return picked


# ============================================================
# RULES
# ============================================================
RULES = """
Sei l’assistente UILCOM per lavoratori IPZS.
Rispondi in modo chiaro basandoti SOLO sul contesto CCNL fornito.
Non inventare informazioni.

CRITICO NOTTURNO:
- Se la domanda è su lavoro notturno ordinario, usa SOLO la percentuale/indennità del notturno ordinario.
- NON usare percentuali di "straordinario notturno" (es. 60%) per il notturno ordinario.
- Se nel contesto non c’è la percentuale del notturno ordinario, devi dirlo.

FORMATO OUTPUT:

<PUBLIC>
...testo per l’utente (senza pagine)...
</PUBLIC>

<ADMIN>
...evidenze con (pag. X) + pagine/chunk...
</ADMIN>
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
            dbg = m.get("debug")
            if dbg:
                with st.expander("🧠 Admin: Debug", expanded=False):
                    st.write("**Domanda arricchita:**")
                    st.code(dbg.get("enriched_q", ""))
                    st.write("**Query:**")
                    st.code("\n".join(dbg.get("queries", [])))
                    st.write("**Evidenze:**")
                    st.code("\n".join(dbg.get("evidence", [])) or "(nessuna)")
                    st.write("**Forced chunks notturno ordinario:**")
                    st.code("\n".join(dbg.get("forced_notturno_pages", [])) or "(nessuno)")
                    st.write("**Chunk finali (prime righe):**")
                    for c in dbg.get("selected", []):
                        st.write(f"**Pagina {c.get('page')}**")
                        txt = c.get("text", "") or ""
                        st.write(txt[:800] + ("..." if len(txt) > 800 else ""))
                        st.divider()

user_input = st.chat_input("Scrivi una domanda sul CCNL (notturno, straordinari, permessi, ROL...)")
if not user_input:
    st.stop()

if not (os.path.exists(VEC_PATH) and os.path.exists(META_PATH)):
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.messages.append({"role": "assistant", "content": "Prima devi indicizzare il CCNL dalla sidebar."})
    st.rerun()

st.session_state.messages.append({"role": "user", "content": user_input})


# ============================================================
# RETRIEVAL
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
        ii = int(i)
        s = float(sims[ii])
        if (ii not in scores_best) or (s > scores_best[ii]):
            scores_best[ii] = s

# Re-ranking boosts (notturno ordinario anti-straordinario)
user_is_notturno = is_lavoro_notturno_question(enriched_q)
user_is_straord_notturno = is_straordinario_notturno_question(enriched_q)
user_is_straord = is_straordinario_question(enriched_q)

for i in list(scores_best.keys()):
    txt = (meta[i].get("text") or "").lower()
    boost = 0.0

    if user_is_notturno and (not user_is_straord_notturno):
        if ("notturn" in txt) and ("straordin" not in txt) and ("%" in txt):
            boost += 0.28
        if ("notturn" in txt) and ("straordin" not in txt):
            boost += 0.12
        if ("straordin" in txt) and ("notturn" in txt):
            boost -= 0.28
        if "60%" in txt or "60 %" in txt:
            boost -= 0.18

    if user_is_straord_notturno:
        if ("straordin" in txt) and ("notturn" in txt) and ("%" in txt):
            boost += 0.22

    if user_is_straord:
        if "straordin" in txt:
            boost += 0.16
        if "maggior" in txt or "%" in txt:
            boost += 0.08

    scores_best[i] = scores_best[i] + boost

final_idx = sorted(scores_best.keys(), key=lambda i: scores_best[i], reverse=True)[:TOP_K_FINAL]
selected = [meta[i] for i in final_idx]

# ✅ Forced retrieval injection for notturno ordinario:
forced_notturno = []
if user_is_notturno and (not user_is_straord_notturno):
    forced_notturno = forced_find_notturno_ordinario_chunks(meta, topn=FORCED_NOTTURNO_TOPN)
    # Merge: prepend forced chunks (no duplicates by page+text hash)
    seen = set()
    merged: List[Dict[str, Any]] = []
    for c in forced_notturno + selected:
        key = (str(c.get("page")), (c.get("text") or "")[:200])
        if key not in seen:
            merged.append(c)
            seen.add(key)
    selected = merged[:TOP_K_FINAL]  # keep limit

# BM25 rerank final
selected = bm25_rerank(enriched_q, selected)

context = "\n\n---\n\n".join([f"[Pagina {c.get('page','?')}] {c.get('text','')}" for c in selected])

key_evidence = extract_key_evidence(selected)
evidence_block = "\n".join([f"- {e}" for e in key_evidence]) if key_evidence else "- (Nessuna evidenza estratta automaticamente.)"


# ============================================================
# LLM
# ============================================================
llm = ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE, api_key=OPENAI_API_KEY)

guardrail = ""
if user_is_notturno and (not user_is_straord_notturno):
    guardrail = (
        "GUARDRAIL: domanda su NOTTURNO ORDINARIO. "
        "Dai percentuale SOLO se nel contesto c’è una frase su notturno (senza 'straordinario') con quella percentuale. "
        "NON usare 60% se è riferito a straordinario notturno.\n"
    )

prompt = f"""
{RULES}

{guardrail}

DOMANDA UTENTE:
{user_input}

EVIDENZE (operative se presenti):
{evidence_block}

CONTESTO CCNL:
{context}

RICORDA:
- PUBLIC: senza pagine.
- ADMIN: evidenze con pagine.
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

# ============================================================
# SANITY CHECK (blocca 60% sul notturno ordinario SOLO se non c’è evidenza)
# ============================================================
if user_is_notturno and (not user_is_straord_notturno):
    has_ord_percent = evidence_has_notturno_ordinario_percent(key_evidence)
    says_60 = ("60%" in public_ans) or ("60 %" in public_ans)
    if says_60 and (not has_ord_percent):
        public_ans = (
            "Nel CCNL caricato emerge una maggiorazione del **60%** riferita allo **straordinario notturno**. "
            "Per il **lavoro notturno ordinario** non posso attribuire quella percentuale.\n\n"
            "Nel contesto recuperato in questa risposta non trovo una riga chiara con la percentuale del notturno ordinario. "
            "Riprovare/raffinare il recupero (admin) per agganciare il paragrafo corretto del CCNL."
        )
        if st.session_state.is_admin:
            admin_ans = (admin_ans or "") + "\n\n[SANITY] Bloccato 60% su notturno ordinario: nessuna evidenza notturno ordinario con % nel contesto."

assistant_payload: Dict[str, Any] = {"role": "assistant", "content": public_ans}

if st.session_state.is_admin:
    assistant_payload["debug"] = {
        "enriched_q": enriched_q,
        "queries": queries,
        "evidence": key_evidence,
        "selected": selected,
        "admin_llm_section": admin_ans,
        "bm25_available": BM25_AVAILABLE,
        "forced_notturno_pages": [f"pag.{c.get('page')}" for c in forced_notturno] if forced_notturno else [],
    }

st.session_state.messages.append(assistant_payload)
st.rerun()
