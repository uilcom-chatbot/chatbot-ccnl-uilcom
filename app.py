# app.py — Assistente Contrattuale UILCOM IPZS (CCNL + IPZS Permessi)
# ✅ Chat stile ChatGPT
# ✅ Risposte CCNL con citazioni pagine
# ✅ Risposte IPZS Permessi (RAO/ROL ecc) + elenco completo quando richiesto
# ✅ Guardrail HARD: se retrieval debole -> "Non ho trovato..."
# ✅ Admin: reindicizza CCNL + Permessi + debug chunk/pagine usate

import os
import re
import json
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
IPZS_PERMESSI_PATH = os.path.join("documenti", "PERMESSI_IPZS_COMPLETO_FINALE.txt")

# Logo (opzionale)
LOGO_PATH_1 = os.path.join("logo", "logo_uilcom.png")   # consigliato
LOGO_PATH_2 = "logo_uilcom.png"                         # fallback se lo metti in root

# Indici
INDEX_CCNL_DIR = "index_ccnl"
CCNL_VEC_PATH = os.path.join(INDEX_CCNL_DIR, "vectors.npy")
CCNL_META_PATH = os.path.join(INDEX_CCNL_DIR, "chunks.json")

INDEX_IPZS_DIR = "index_ipzs_permessi"
IPZS_VEC_PATH = os.path.join(INDEX_IPZS_DIR, "vectors.npy")
IPZS_META_PATH = os.path.join(INDEX_IPZS_DIR, "chunks.json")

# Chunking
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150

TOP_K_PER_QUERY = 12
TOP_K_FINAL = 18
MAX_MULTI_QUERIES = 10

# Guardrail retrieval
MIN_BEST_SIMILARITY = 0.24
MIN_SELECTED_CHUNKS = 3

# Admin
MAX_EVIDENCE_LINES = 18

# LLM
LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0


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

def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

UILCOM_PASSWORD = get_secret("UILCOM_PASSWORD")        # password iscritti
ADMIN_PASSWORD = get_secret("ADMIN_PASSWORD")          # password admin
OPENAI_API_KEY = get_secret("OPENAI_API_KEY")          # obbligatoria


# ============================================================
# UI SETUP
# ============================================================
st.set_page_config(page_title="Assistente UILCOM IPZS", page_icon="🟦", layout="centered")

# Logo (non deve mai rompere l'app)
logo_to_use = LOGO_PATH_1 if os.path.exists(LOGO_PATH_1) else (LOGO_PATH_2 if os.path.exists(LOGO_PATH_2) else None)
if logo_to_use:
    try:
        st.image(logo_to_use, width=140)
    except Exception:
        pass

st.title(APP_TITLE)
st.markdown(
    "**Accesso riservato agli iscritti UILCOM**  \n"
    "Strumento informativo per facilitare la consultazione del **CCNL** e del documento **IPZS Permessi/Giustificativi**.  \n\n"
    "⚠️ Le risposte sono generate **solo** in base ai documenti caricati. "
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
    st.warning("Password iscritti non impostata. Imposta UILCOM_PASSWORD in Secrets.")
    # st.session_state.auth_ok = True  # per sviluppo locale

if not st.session_state.auth_ok:
    st.stop()


# ============================================================
# ADMIN SIDEBAR
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

    st.subheader("📦 Indici documenti")
    ok_ccnl = os.path.exists(CCNL_VEC_PATH) and os.path.exists(CCNL_META_PATH)
    ok_ipzs = os.path.exists(IPZS_VEC_PATH) and os.path.exists(IPZS_META_PATH)

    st.write("Indice CCNL:", "✅" if ok_ccnl else "❌")
    st.write("Indice IPZS Permessi:", "✅" if ok_ipzs else "❌")

    if st.button("Indicizza / Reindicizza (CCNL + IPZS)", use_container_width=True):
        if not OPENAI_API_KEY:
            st.error("Manca OPENAI_API_KEY nei Secrets.")
        else:
            try:
                with st.spinner("Indicizzazione in corso..."):
                    os.makedirs(INDEX_CCNL_DIR, exist_ok=True)
                    os.makedirs(INDEX_IPZS_DIR, exist_ok=True)

                    emb = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
                    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

                    # ---- CCNL PDF ----
                    if not os.path.exists(PDF_PATH):
                        raise FileNotFoundError(f"Non trovo il PDF CCNL: {PDF_PATH}")

                    loader = PyPDFLoader(PDF_PATH)
                    docs = loader.load()
                    ccnl_chunks = splitter.split_documents(docs)

                    ccnl_texts = [c.page_content for c in ccnl_chunks]
                    ccnl_pages = [(int(c.metadata.get("page", 0)) + 1) for c in ccnl_chunks]  # 1-based

                    ccnl_vectors = np.array(emb.embed_documents(ccnl_texts), dtype=np.float32)
                    np.save(CCNL_VEC_PATH, ccnl_vectors)
                    with open(CCNL_META_PATH, "w", encoding="utf-8") as f:
                        json.dump([{"page": p, "text": t} for p, t in zip(ccnl_pages, ccnl_texts)], f, ensure_ascii=False)

                    # ---- IPZS Permessi TXT ----
                    if not os.path.exists(IPZS_PERMESSI_PATH):
                        raise FileNotFoundError(f"Non trovo IPZS Permessi: {IPZS_PERMESSI_PATH}")

                    with open(IPZS_PERMESSI_PATH, "r", encoding="utf-8", errors="ignore") as f:
                        ipzs_raw = f.read()

                    # chunk a testo con "pseudo-page" (sezione)
                    # Spezziamo comunque con splitter per retrieval robusto
                    ipzs_docs = [{"page": "IPZS", "text": ipzs_raw}]
                    ipzs_texts = [ipzs_raw]
                    # split
                    ipzs_chunks = splitter.split_text(ipzs_raw)
                    ipzs_vectors = np.array(emb.embed_documents(ipzs_chunks), dtype=np.float32)
                    np.save(IPZS_VEC_PATH, ipzs_vectors)
                    with open(IPZS_META_PATH, "w", encoding="utf-8") as f:
                        json.dump([{"page": "IPZS", "text": t} for t in ipzs_chunks], f, ensure_ascii=False)

                st.success("Indicizzazione completata ✅")
                st.rerun()
            except Exception as e:
                st.error(str(e))

    if st.button("🧹 Nuova chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("Suggerimento: dopo aggiornamenti su GitHub → Streamlit Cloud fa auto-redeploy. Se no: **Reboot app**.")


# ============================================================
# HARD FAIL IF NO OPENAI KEY
# ============================================================
if not OPENAI_API_KEY:
    st.error(
        "Manca la variabile **OPENAI_API_KEY**.\n\n"
        "Streamlit Cloud: **Settings → Secrets → OPENAI_API_KEY**"
    )
    st.stop()


# ============================================================
# HELPERS: VECTOR SEARCH
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

    fixed = []
    for item in meta:
        if isinstance(item, dict) and "text" in item:
            fixed.append({"page": item.get("page", "?"), "text": item.get("text", "")})
        else:
            fixed.append({"page": "?", "text": str(item)})
    return vectors, fixed

def bm25_rerank(query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not BM25_AVAILABLE or not candidates:
        return candidates
    corpus = [(c.get("text") or "").lower().split() for c in candidates]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(query.lower().split())
    idx = np.argsort(-np.array(scores))
    return [candidates[int(i)] for i in idx]

def unique_pages_ccnl(chunks: List[Dict[str, Any]], max_pages: int = 8) -> List[int]:
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

def format_ccnl_citations(pages: List[int]) -> str:
    if not pages:
        return ""
    pages_sorted = sorted(pages)
    if len(pages_sorted) == 1:
        return f"**Fonte:** CCNL (pag. {pages_sorted[0]})"
    return f"**Fonte:** CCNL (pagg. {', '.join(map(str, pages_sorted))})"

def extract_evidence(chunks: List[Dict[str, Any]], max_lines: int = 18) -> List[str]:
    patterns = [
        r"rao", r"r\.a\.o", r"rol", r"r\.o\.l", r"riduzion",
        r"permess", r"conged", r"retribuit", r"non retribuit",
        r"donazion", r"decesso", r"lutto", r"studio", r"esam", r"104",
        r"cure termali", r"elettoral", r"testimonianza",
    ]
    out = []
    for c in chunks:
        page = c.get("page", "?")
        text = c.get("text", "") or ""
        for ln in [x.strip() for x in text.splitlines() if x.strip()]:
            low = ln.lower()
            if any(re.search(p, low) for p in patterns):
                clean = " ".join(ln.split())
                if 20 <= len(clean) <= 420:
                    out.append(f"({page}) {clean}")
            if len(out) >= max_lines:
                return out
    return out[:max_lines]


# ============================================================
# IPZS PERMESSI: PARSER ELENCO COMPLETO (deterministico)
# ============================================================
def parse_ipzs_permessi_sections(raw_text: str) -> List[Dict[str, str]]:
    """
    Prova a ricostruire un elenco di "permessi" dal file IPZS:
    - Riconosce titoli in MAIUSCOLO (es: DONAZIONE SANGUE, RAO, R.O.L., CURE TERMALI)
    - Prende il blocco "Descrizione:" sotto al titolo fino al prossimo titolo
    """
    lines = raw_text.splitlines()
    sections: List[Dict[str, str]] = []

    def is_title(line: str) -> bool:
        s = line.strip()
        if not s:
            return False
        # escludi righe decorative
        if set(s) <= set("-_= "):
            return False
        # titolo: molti caratteri maiuscoli + pochi segni
        # oppure RAO / R.O.L.
        if s in ["RAO", "R.O.L.", "ROL", "R.A.O."]:
            return True
        # se è quasi tutto uppercase
        letters = [ch for ch in s if ch.isalpha()]
        if len(letters) >= 6 and sum(ch.isupper() for ch in letters) / max(len(letters), 1) > 0.8:
            return True
        # contiene pattern tipici
        if re.search(r"\bL\.\s*\d+/\d+\b", s):
            return True
        return False

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if is_title(line):
            title = line
            # salta eventuali righe decorative sotto
            j = i + 1
            while j < len(lines) and (lines[j].strip() == "" or set(lines[j].strip()) <= set("-_= ")):
                j += 1

            # cerca descrizione
            desc_lines = []
            # se c'è "Descrizione:" la includiamo e prendiamo le righe successive
            if j < len(lines) and lines[j].strip().lower().startswith("descrizione"):
                desc_lines.append(lines[j].strip())
                j += 1

            while j < len(lines):
                nxt = lines[j].strip()
                if is_title(nxt):
                    break
                desc_lines.append(lines[j].rstrip())
                j += 1

            desc = "\n".join([x for x in desc_lines if x.strip()]).strip()
            if desc:
                sections.append({"title": title.strip(), "desc": desc})
            i = j
        else:
            i += 1

    # dedup titoli
    out = []
    seen = set()
    for s in sections:
        t = re.sub(r"\s+", " ", s["title"]).strip()
        key = t.lower()
        if key not in seen:
            out.append({"title": t, "desc": s["desc"]})
            seen.add(key)
    return out

def ipzs_source_line() -> str:
    return "**Fonte:** Documento IPZS Permessi/Giustificativi (PERMESSI_IPZS_COMPLETO_FINALE.txt)"

def is_generic_permessi_list_question(q: str) -> bool:
    ql = q.lower()
    triggers = [
        "a quali permessi ha diritto",
        "quali permessi ha diritto",
        "quali permessi ha a disposizione",
        "elenco permessi",
        "tutti i permessi",
        "lista permessi",
        "permessi disponibili",
        "permessi ipzs",
        "giustificativi",
    ]
    return any(t in ql for t in triggers)

def extract_specific_permesso_name(q: str, sections_titles: List[str]) -> Optional[str]:
    """
    Se l'utente chiede un permesso specifico, proviamo a matchare:
    - RAO / ROL
    - parole contenute in un titolo
    """
    ql = q.lower()

    # priorità RAO / ROL
    if re.search(r"\brao\b|\br\.a\.o\b", ql):
        return "RAO"
    if re.search(r"\brol\b|\br\.o\.l\b", ql):
        return "R.O.L."

    # match per inclusione (titolo dentro domanda o viceversa)
    for t in sections_titles:
        tl = t.lower()
        # evita titoli troppo generici
        if len(tl) < 4:
            continue
        if tl in ql:
            return t
    # match per parole chiave importanti
    keywords_map = {
        "donazione": ["DONAZIONE SANGUE", "DONAZIONE"],
        "sangue": ["DONAZIONE SANGUE"],
        "cure termali": ["CURE TERMALI"],
        "elettoral": ["PERMESSI ELETTORALI"],
        "decesso": ["DECESSO", "DECESSO FAMILIARI"],
        "testimon": ["TESTIMONIANZA CIVILE"],
        "esami": ["PERM. ESAMI", "ESAMI"],
        "104": ["L. 104/92", "L.104/92", "104"],
        "maternit": ["MATERNITA' OBBLIGATORIA", "MATERNITA'"],
        "padre": ["CONG.OBBLIGATORIO PADRE"],
        "nido": ["INSERIM.NIDO", "ACC.NIDO", "NIDO"],
    }
    for k, cand in keywords_map.items():
        if k in ql:
            # ritorna il primo che esiste davvero tra i titoli
            for c in cand:
                for real in sections_titles:
                    if c.lower() in real.lower() or real.lower() in c.lower():
                        return real
    return None


# ============================================================
# ROUTING: capire se rispondere con IPZS o CCNL
# ============================================================
def detect_route(q: str) -> str:
    """
    - Se domanda è lista permessi -> IPZS (elenco completo)
    - Se domanda su permesso specifico (rao/rol/donazione ecc) -> IPZS
    - Altrimenti -> CCNL
    """
    ql = q.lower()
    if is_generic_permessi_list_question(ql):
        return "IPZS_LIST"

    # segnali di permesso specifico
    if any(x in ql for x in ["rao", "r.a.o", "rol", "r.o.l", "donazione", "cure termali", "permessi elettorali", "testimonianza", "decesso", "104", "perm. esami", "maternit", "padre", "nido"]):
        return "IPZS_SINGLE"

    # se contiene "permessi" ma non chiede elenco, proviamo IPZS prima
    if "permess" in ql or "giustific" in ql:
        return "MIXED"

    return "CCNL"


# ============================================================
# MULTI-QUERY BUILDER
# ============================================================
def build_queries(q: str) -> List[str]:
    q0 = q.strip()
    qs = [q0, f"{q0} CCNL", f"{q0} regole condizioni", f"{q0} definizione procedura"]
    out, seen = [], set()
    for x in qs:
        x = x.strip()
        if x and x not in seen:
            out.append(x)
            seen.add(x)
    return out[:MAX_MULTI_QUERIES]


# ============================================================
# RETRIEVE (generic)
# ============================================================
def retrieve_from_index(
    query: str,
    vec_path: str,
    meta_path: str,
    emb: OpenAIEmbeddings,
    top_k_per_query: int = TOP_K_PER_QUERY,
    top_k_final: int = TOP_K_FINAL,
) -> Tuple[List[Dict[str, Any]], float, List[str], List[Dict[str, Any]]]:
    vectors, meta = load_index(vec_path, meta_path)
    mat_norm = normalize_rows(vectors)

    queries = build_queries(query)

    scores_best: Dict[int, float] = {}
    best_similarity = 0.0

    for q in queries:
        qvec = np.array(emb.embed_query(q), dtype=np.float32)
        sims = cosine_scores(qvec, mat_norm)
        top_idx = np.argsort(-sims)[:top_k_per_query]
        for i in top_idx:
            s = float(sims[int(i)])
            best_similarity = max(best_similarity, s)
            if (int(i) not in scores_best) or (s > scores_best[int(i)]):
                scores_best[int(i)] = s

    final_idx = sorted(scores_best.keys(), key=lambda i: scores_best[i], reverse=True)[:top_k_final]
    selected = [meta[i] for i in final_idx]
    selected = bm25_rerank(query, selected)

    evidence = extract_evidence(selected, max_lines=MAX_EVIDENCE_LINES)
    return selected, best_similarity, queries, selected


def hard_not_found() -> str:
    return "Non ho trovato la risposta nei documenti caricati."


# ============================================================
# CCNL LLM PROMPT
# ============================================================
RULES_CCNL = """
Sei l’assistente UILCOM per lavoratori IPZS.
Devi rispondere in modo chiaro e pratico basandoti SOLO sul contesto (estratti CCNL) fornito.
Non inventare informazioni.

REGOLE:
1) Se non trovi nel contesto, scrivi: "Non ho trovato la risposta nel CCNL caricato."
2) Chiudi SEMPRE con: "Fonte: CCNL (pag. ...)" usando solo le pagine presenti nel contesto.
3) Se l’utente chiede permessi ma nel contesto non ci sono tutti, non inventare.
"""

def split_public(text: str) -> str:
    # compatibile anche se il modello non usa tag
    m = re.search(r"<PUBLIC>(.*?)</PUBLIC>", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return text.strip()


# ============================================================
# CHAT STATE
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render chat history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if st.session_state.is_admin and m["role"] == "assistant":
            dbg = m.get("debug")
            if dbg:
                with st.expander("🧠 Admin: debug", expanded=False):
                    st.write("**Route:**", dbg.get("route"))
                    st.write("**Best similarity:**", dbg.get("best_similarity"))
                    st.write("**Query usate:**")
                    st.code("\n".join(dbg.get("queries", [])))
                    st.write("**Evidenze:**")
                    st.code("\n".join(dbg.get("evidence", [])) or "(nessuna)")
                    if dbg.get("pages_ccnl"):
                        st.write("**Pagine CCNL:**", dbg.get("pages_ccnl"))
                    st.write("**Chunk selezionati (prime righe):**")
                    for c in dbg.get("selected", [])[:4]:
                        st.write(f"**Pagina/Origine: {c.get('page')}**")
                        txt = (c.get("text") or "")
                        st.write(txt[:700] + ("..." if len(txt) > 700 else ""))
                        st.divider()

user_input = st.chat_input("Scrivi una domanda (CCNL / permessi IPZS: RAO, ROL, donazione sangue, ecc.)")

if not user_input:
    st.stop()

# Require indexes
need_ccnl = not (os.path.exists(CCNL_VEC_PATH) and os.path.exists(CCNL_META_PATH))
need_ipzs = not (os.path.exists(IPZS_VEC_PATH) and os.path.exists(IPZS_META_PATH))

if need_ccnl or need_ipzs:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Prima devo indicizzare i documenti: apri la barra laterale e clicca **Indicizza / Reindicizza (CCNL + IPZS)**.",
    })
    st.rerun()

st.session_state.messages.append({"role": "user", "content": user_input})

emb = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
route = detect_route(user_input)

# ============================================================
# IPZS: ELENCO COMPLETO (deterministico)
# ============================================================
if route == "IPZS_LIST":
    # Leggi file intero e costruisci elenco
    try:
        with open(IPZS_PERMESSI_PATH, "r", encoding="utf-8", errors="ignore") as f:
            ipzs_raw = f.read()
        sections = parse_ipzs_permessi_sections(ipzs_raw)

        if not sections:
            public_ans = hard_not_found()
        else:
            # elenco completo
            lines = []
            lines.append("Ecco l’**elenco dei permessi/giustificativi** presenti nel documento IPZS, con una breve spiegazione:")
            lines.append("")
            for s in sections:
                title = s["title"]
                desc = s["desc"]
                # prendi 1-3 righe utili
                desc_short = []
                for ln in desc.splitlines():
                    ln = ln.strip()
                    if not ln:
                        continue
                    # evita solo "Descrizione:"
                    if ln.lower() == "descrizione:":
                        continue
                    desc_short.append(ln)
                    if len(desc_short) >= 3:
                        break
                preview = " ".join(desc_short).strip()
                if not preview:
                    preview = "Vedi descrizione nel documento."
                lines.append(f"- **{title}**: {preview}")

            lines.append("")
            lines.append(ipzs_source_line())
            public_ans = "\n".join(lines).strip()

        assistant_payload = {"role": "assistant", "content": public_ans}

        if st.session_state.is_admin:
            # retrieval non serve qui: è elenco deterministico
            assistant_payload["debug"] = {
                "route": route,
                "best_similarity": None,
                "queries": [],
                "evidence": [],
                "pages_ccnl": [],
                "selected": [],
                "note": "Elenco IPZS generato deterministico dal file completo.",
            }

        st.session_state.messages.append(assistant_payload)
        st.rerun()
    except Exception:
        assistant_payload = {"role": "assistant", "content": hard_not_found()}
        st.session_state.messages.append(assistant_payload)
        st.rerun()


# ============================================================
# IPZS: PERMESSO SPECIFICO (RAO/ROL/altro)
# ============================================================
if route in ["IPZS_SINGLE", "MIXED"]:
    # prima prova IPZS
    selected_ipzs, best_ipzs, queries_ipzs, _ = retrieve_from_index(
        user_input, IPZS_VEC_PATH, IPZS_META_PATH, emb
    )
    retrieval_ok_ipzs = (best_ipzs >= MIN_BEST_SIMILARITY) and (len(selected_ipzs) >= MIN_SELECTED_CHUNKS)

    # carica tutte le sezioni per capire il "titolo" richiesto
    try:
        with open(IPZS_PERMESSI_PATH, "r", encoding="utf-8", errors="ignore") as f:
            ipzs_raw = f.read()
        sections = parse_ipzs_permessi_sections(ipzs_raw)
        titles = [s["title"] for s in sections]
        wanted = extract_specific_permesso_name(user_input, titles)
    except Exception:
        sections, titles, wanted = [], [], None

    if retrieval_ok_ipzs and sections:
        # Se l'utente ha chiesto un permesso specifico, rispondi SOLO quello
        if wanted:
            match = None
            # normalizza RAO/ROL
            if wanted == "RAO":
                for s in sections:
                    if s["title"].strip().lower() in ["rao", "r.a.o."]:
                        match = s
                        break
                if not match:
                    for s in sections:
                        if "rao" in s["title"].lower():
                            match = s
                            break
            elif wanted in ["R.O.L.", "ROL"]:
                for s in sections:
                    if "rol" in s["title"].lower() or "r.o.l" in s["title"].lower():
                        match = s
                        break
            else:
                for s in sections:
                    if s["title"].lower() == wanted.lower():
                        match = s
                        break
                if not match:
                    for s in sections:
                        if wanted.lower() in s["title"].lower():
                            match = s
                            break

            if match:
                public_ans = f"**{match['title']}**\n\n{match['desc'].strip()}\n\n{ipzs_source_line()}"
            else:
                # fallback: usa i chunk recuperati
                context = "\n\n---\n\n".join([c.get("text", "") for c in selected_ipzs[:6]])
                public_ans = f"Ho trovato riferimenti nel documento IPZS, ma non riesco a isolare la sezione esatta.\n\n{context}\n\n{ipzs_source_line()}"
        else:
            # Se NON è chiaro quale permesso, e la route è MIXED (generica),
            # allora non spariamo l'elenco: rispondi chiedendo parola chiave (ma senza bloccare troppo)
            if route == "MIXED" and ("permess" in user_input.lower()):
                public_ans = (
                    "Mi puoi dire **quale permesso** ti interessa (es. **RAO**, **ROL**, **donazione sangue**, **cure termali**, **104**, ecc.)?\n\n"
                    + ipzs_source_line()
                )
            else:
                public_ans = hard_not_found()

        assistant_payload = {"role": "assistant", "content": public_ans}

        if st.session_state.is_admin:
            assistant_payload["debug"] = {
                "route": route,
                "best_similarity": best_ipzs,
                "queries": queries_ipzs,
                "evidence": extract_evidence(selected_ipzs),
                "pages_ccnl": [],
                "selected": selected_ipzs[:6],
                "note": f"Permesso specifico richiesto: {wanted}",
            }

        st.session_state.messages.append(assistant_payload)
        st.rerun()

    # se IPZS debole e route MIXED, proviamo CCNL
    if route == "MIXED":
        # continua su CCNL sotto
        pass
    else:
        assistant_payload = {"role": "assistant", "content": hard_not_found()}
        st.session_state.messages.append(assistant_payload)
        st.rerun()


# ============================================================
# CCNL: Retrieval + LLM con citazioni
# ============================================================
selected_ccnl, best_ccnl, queries_ccnl, _ = retrieve_from_index(
    user_input, CCNL_VEC_PATH, CCNL_META_PATH, emb
)
retrieval_ok_ccnl = (best_ccnl >= MIN_BEST_SIMILARITY) and (len(selected_ccnl) >= MIN_SELECTED_CHUNKS)

if not retrieval_ok_ccnl:
    assistant_payload = {"role": "assistant", "content": "Non ho trovato la risposta nel CCNL caricato."}
    if st.session_state.is_admin:
        assistant_payload["debug"] = {
            "route": "CCNL",
            "best_similarity": best_ccnl,
            "queries": queries_ccnl,
            "evidence": extract_evidence(selected_ccnl),
            "pages_ccnl": unique_pages_ccnl(selected_ccnl),
            "selected": selected_ccnl[:6],
            "note": "retrieval_ok=False -> risposta bloccata",
        }
    st.session_state.messages.append(assistant_payload)
    st.rerun()

pages = unique_pages_ccnl(selected_ccnl, max_pages=8)
cit_line = format_ccnl_citations(pages)

context = "\n\n---\n\n".join([f"[Pagina {c.get('page','?')}] {c.get('text','')}" for c in selected_ccnl])

llm = ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE, api_key=OPENAI_API_KEY)

prompt = f"""
{RULES_CCNL}

DOMANDA (UTENTE):
{user_input}

CONTESTO CCNL (estratti):
{context}

SCRIVI UNA RISPOSTA CHIARA E PRATICA.
CHIUDI SEMPRE CON:
{cit_line}

FORMATO:
<PUBLIC>
...testo...
{cit_line}
</PUBLIC>
"""

try:
    raw = llm.invoke(prompt).content
except Exception as e:
    raw = f"<PUBLIC>Errore nel generare la risposta: {e}\n\n{cit_line}</PUBLIC>"

public_ans = split_public(raw).strip()

# Forza citazione se manca
if cit_line and ("fonte" not in public_ans.lower()):
    public_ans = public_ans.rstrip() + "\n\n" + cit_line

assistant_payload = {"role": "assistant", "content": public_ans}

if st.session_state.is_admin:
    assistant_payload["debug"] = {
        "route": "CCNL",
        "best_similarity": best_ccnl,
        "queries": queries_ccnl,
        "evidence": extract_evidence(selected_ccnl),
        "pages_ccnl": pages,
        "selected": selected_ccnl[:6],
        "bm25_available": BM25_AVAILABLE,
    }

st.session_state.messages.append(assistant_payload)
st.rerun()
