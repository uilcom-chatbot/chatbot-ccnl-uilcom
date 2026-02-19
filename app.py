import os
import json
import re
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# Optional BM25
try:
    from rank_bm25 import BM25Okapi  # type: ignore
    HAS_BM25 = True
except Exception:
    HAS_BM25 = False


# ============================================================
# UI / CONFIG
# ============================================================
st.set_page_config(page_title="Assistente Contrattuale UILCOM IPZS", page_icon="🟦", layout="centered")

st.title("🟦 Assistente Contrattuale UILCOM IPZS")
st.markdown(
    "**Accesso riservato agli iscritti UILCOM**  \n"
    "Strumento informativo per consultare il CCNL Grafici Editoria applicabile ai lavoratori.  \n"
    "Le risposte sono basate **solo** sul CCNL caricato. Per casi specifici rivolgersi a RSU/UILCOM o HR."
)
st.divider()

PDF_PATH = os.path.join("documenti", "ccnl.pdf")
INDEX_DIR = "index_ccnl"
VEC_PATH = os.path.join(INDEX_DIR, "vectors.npy")
META_PATH = os.path.join(INDEX_DIR, "chunks.json")

TOP_K_FINAL = 18
TOP_K_PER_QUERY = 12
MAX_MULTI_QUERIES = 14

MEMORY_USER_TURNS = 3
PERMESSI_MIN_CATEGORY_COVERAGE = 3


# ============================================================
# SECRETS / ENV
# ============================================================
def get_secret(name: str) -> Optional[str]:
    try:
        v = st.secrets.get(name, None)  # type: ignore
    except Exception:
        v = None
    if not v:
        v = os.getenv(name)
    return v

OPENAI_API_KEY = get_secret("OPENAI_API_KEY")
UILCOM_PASSWORD = get_secret("UILCOM_PASSWORD")
ADMIN_PASSWORD = get_secret("ADMIN_PASSWORD")  # opzionale

if not OPENAI_API_KEY:
    st.error(
        "Manca **OPENAI_API_KEY**.\n\n"
        "Streamlit Cloud: Settings → Secrets → aggiungi OPENAI_API_KEY.\n"
        "Locale: imposta OPENAI_API_KEY come variabile d'ambiente."
    )
    st.stop()

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


# ============================================================
# AUTH (iscritti)
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
    st.info("🔐 Password iscritti non impostata. (Per produzione imposta UILCOM_PASSWORD nei Secrets.)")
    st.session_state.auth_ok = True

if not st.session_state.auth_ok:
    st.stop()


# ============================================================
# ADMIN DEBUG (solo admin; invisibile ai colleghi)
# ============================================================
if "admin_ok" not in st.session_state:
    st.session_state.admin_ok = False

if "debug_on" not in st.session_state:
    st.session_state.debug_on = False

def admin_box():
    if not ADMIN_PASSWORD:
        return
    with st.sidebar.expander("🛠️ Admin (debug)", expanded=False):
        p = st.text_input("Password admin", type="password", placeholder="Solo amministratori")
        if st.button("Sblocca debug"):
            st.session_state.admin_ok = (p == ADMIN_PASSWORD)
            if st.session_state.admin_ok:
                st.success("Debug attivo (solo admin).")
            else:
                st.error("Password admin errata.")
        if st.session_state.admin_ok:
            st.session_state.debug_on = st.toggle("Mostra fonti/estratti", value=st.session_state.debug_on)

admin_box()


# ============================================================
# HELPERS
# ============================================================
def ensure_meta_dicts(meta: Any) -> List[Dict[str, Any]]:
    if isinstance(meta, list) and meta and isinstance(meta[0], dict):
        return meta
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

def normalize_rows(mat: np.ndarray) -> np.ndarray:
    return mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)

def cosine_scores(query_vec: np.ndarray, mat_norm: np.ndarray) -> np.ndarray:
    q = query_vec / (np.linalg.norm(query_vec) + 1e-12)
    return mat_norm @ q


# ============================================================
# CLASSIFICAZIONE DOMANDA (fix notturno/straordinario)
# ============================================================
NOTTURNO_WORDS = ["notturn", "notte", "turno notturno", "lavoro notturno"]
STRAO_WORDS = ["straordin", "ore extra", "lavoro straordinario"]
FESTIVO_WORDS = ["festiv", "domenic", "festivo"]
TURNI_WORDS = ["turn", "turni", "avvicend", "ciclo"]

PAGAMENTO_WORDS = ["quanto viene pagato", "quanto si prende", "maggiorazione", "percentuale", "indennità", "indennita", "retribuzione", "maggior"]

def q_contains_any(q: str, arr: List[str]) -> bool:
    ql = q.lower()
    return any(w in ql for w in arr)

def classify_pay_question(user_q: str) -> Dict[str, bool]:
    ql = user_q.lower()
    return {
        "is_pay_q": q_contains_any(ql, PAGAMENTO_WORDS) or ("%" in ql),
        "is_notturno": q_contains_any(ql, NOTTURNO_WORDS),
        "is_strao": q_contains_any(ql, STRAO_WORDS),
        "is_festivo": q_contains_any(ql, FESTIVO_WORDS),
        "mentions_turni": q_contains_any(ql, TURNI_WORDS),
        "is_notturno_ordinario": (q_contains_any(ql, NOTTURNO_WORDS) and (not q_contains_any(ql, STRAO_WORDS))),
        "is_strao_notturno": (q_contains_any(ql, NOTTURNO_WORDS) and q_contains_any(ql, STRAO_WORDS)),
    }


# ============================================================
# PERMESSI (copertura)
# ============================================================
PERMESSI_TRIGGERS = [
    "permess", "permesso", "permessi", "retribuit", "assenze retribuite",
    "visita medica", "visite mediche", "lutto", "matrimonio", "104",
    "sindacal", "assemblea", "rsu", "studio", "formazione",
    "donazione", "sangue", "rol", "ex festiv", "festivit", "festività",
]
ROL_TRIGGERS = [
    "rol", "r.o.l", "riduzione orario",
    "ex festiv", "ex festività", "festività soppresse", "festivita soppresse",
]

PERMESSI_CATEGORIES = {
    "visite_mediche": [r"visite?\s+med", r"visita\s+med", r"accertament", r"specialist"],
    "lutto": [r"\blutto\b", r"decesso", r"grave\s+lutto"],
    "matrimonio": [r"matrimon", r"nozz"],
    "studio_formazione": [r"diritto\s+allo\s+studio", r"\b150\s+ore\b", r"\bstudio\b", r"\besami\b", r"formazion"],
    "legge_104": [r"\b104\b", r"legge\s*104", r"handicap"],
    "sindacali": [r"sindacal", r"\brsu\b", r"assemblea", r"permessi?\s+sindacal"],
    "donazione_sangue": [r"donazion", r"sangue"],
    "altri_permessi": [r"permessi?\s+retribuit", r"assenze?\s+retribuit", r"conged"],
}

def is_permessi_question(q: str) -> bool:
    ql = q.lower()
    return any(t in ql for t in PERMESSI_TRIGGERS)

def is_rol_question(q: str) -> bool:
    ql = q.lower()
    return any(t in ql for t in ROL_TRIGGERS)

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
# MEMORY
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
    ql = q0.lower()
    qs = [q0, f"{q0} CCNL", f"{q0} regole", f"{q0} maggiorazioni indennità percentuali"]

    pay_cls = classify_pay_question(q0)

    # notturno ordinario
    if pay_cls["is_notturno_ordinario"]:
        qs += [
            "lavoro notturno ordinario maggiorazione",
            "turno notturno ordinario indennità notturna",
            "lavoro a turni notturni maggiorazioni",
            "maggiorazione lavoro notturno (non straordinario)",
            "indennità per turno notturno",
        ]

    # straordinario notturno
    if pay_cls["is_strao_notturno"]:
        qs += [
            "straordinario notturno maggiorazione percentuale",
            "lavoro straordinario notturno maggiorato",
            "ore straordinarie notturne percentuali",
        ]

    # straordinario generico
    if ("straordin" in ql) and (not pay_cls["is_strao_notturno"]):
        qs += [
            "lavoro straordinario maggiorazioni percentuali",
            "straordinario diurno maggiorazioni",
            "straordinario festivo maggiorazioni",
        ]

    # festivo (ordinario o straordinario)
    if ("festiv" in ql) or ("domenic" in ql):
        qs += [
            "lavoro festivo maggiorazione",
            "lavoro domenicale maggiorazione",
            "straordinario festivo maggiorazione percentuale",
        ]

    # permessi
    if is_rol_question(q0):
        qs += [
            "ROL riduzione orario monte ore annuo maturazione",
            "ex festività festività soppresse permessi ore giorni",
            "permessi ROL ex festività utilizzo preavviso",
        ]
    elif is_permessi_question(q0):
        qs += [
            "permessi retribuiti tipologie elenco",
            "assenze retribuite visite mediche lutto matrimonio 104 sindacali studio donazione sangue",
        ]

    # dedup
    out, seen = [], set()
    for x in qs:
        x = x.strip()
        if x and x not in seen:
            out.append(x)
            seen.add(x)
    return out[:MAX_MULTI_QUERIES]


# ============================================================
# ESTRATTO "FATTI" SU PERCENTUALI (ANTI-ERRORE)
# ============================================================
PCT_RE = re.compile(r"(\d{1,3})\s*%")

def window_has_keywords(text: str, pct_span: Tuple[int, int], keywords: List[str], window: int = 90) -> bool:
    lo = max(0, pct_span[0] - window)
    hi = min(len(text), pct_span[1] + window)
    w = text[lo:hi].lower()
    return any(k in w for k in keywords)

def extract_pay_facts(selected: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Estrae candidate percentuali/indennità e le classifica in:
    - notturno_ordinario
    - straordinario_notturno
    - straordinario_diurno
    - festivo
    - altro
    Senza inventare: prende SOLO se vicino ci sono parole chiave coerenti.
    """
    facts = {
        "notturno_ordinario": [],
        "straordinario_notturno": [],
        "straordinario_diurno": [],
        "festivo": [],
        "altro": [],
    }

    for c in selected:
        page = c.get("page", "?")
        t = (c.get("text", "") or "")
        tl = t.lower()

        for m in PCT_RE.finditer(t):
            pct = m.group(1)
            span = m.span()

            # straordinario notturno
            if window_has_keywords(t, span, ["straordin", "ore straord", "lavoro straordin"]):
                if window_has_keywords(t, span, ["notturn", "notte"]):
                    facts["straordinario_notturno"].append({"pct": pct, "page": page, "snippet": t[max(0, span[0]-110):min(len(t), span[1]+160)]})
                    continue
                # straordinario diurno (se non ci sono parole notturno)
                facts["straordinario_diurno"].append({"pct": pct, "page": page, "snippet": t[max(0, span[0]-110):min(len(t), span[1]+160)]})
                continue

            # notturno ordinario: deve esserci notturno ma NON straordinario vicino
            if window_has_keywords(t, span, ["notturn", "notte", "turno"]) and (not window_has_keywords(t, span, ["straordin", "ore straord", "lavoro straordin"])):
                facts["notturno_ordinario"].append({"pct": pct, "page": page, "snippet": t[max(0, span[0]-110):min(len(t), span[1]+160)]})
                continue

            # festivo
            if window_has_keywords(t, span, ["festiv", "domenic"]):
                facts["festivo"].append({"pct": pct, "page": page, "snippet": t[max(0, span[0]-110):min(len(t), span[1]+160)]})
                continue

            facts["altro"].append({"pct": pct, "page": page, "snippet": t[max(0, span[0]-110):min(len(t), span[1]+160)]})

        # anche “maggiorata del 60%” senza simbolo %? (copre casi)
        if "maggiorata del 60" in tl or "maggiorato del 60" in tl:
            # se vicino c'è straordinario => straordinario
            if "straordin" in tl:
                if "notturn" in tl:
                    facts["straordinario_notturno"].append({"pct": "60", "page": page, "snippet": t[:400]})
                else:
                    facts["straordinario_diurno"].append({"pct": "60", "page": page, "snippet": t[:400]})

    # Dedup base (pct+page+snippet head)
    def dedup(lst: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out, seen = [], set()
        for x in lst:
            k = (x.get("pct"), x.get("page"), (x.get("snippet") or "")[:80])
            if k not in seen:
                out.append(x)
                seen.add(k)
        return out

    for k in facts:
        facts[k] = dedup(facts[k])[:10]

    return facts


# ============================================================
# INDEX
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
    pages = [(c.metadata.get("page", 0) + 1) for c in chunks]  # foglio PDF

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
    return vectors, ensure_meta_dicts(meta)


# ============================================================
# RULES (super precisione + fix notturno)
# ============================================================
RULES = (
    "Sei l’assistente UILCOM per lavoratori IPZS. "
    "Rispondi in modo chiaro e professionale basandoti SOLO sul contesto del CCNL fornito. "
    "Non inventare informazioni.\n\n"

    "REGOLA PROVA: afferma numeri/percentuali/durate SOLO se sono nel contesto recuperato. "
    "Se non emerge, devi dirlo esplicitamente.\n\n"

    "GUARDRAIL NOTTURNO: non confondere 'lavoro notturno ordinario' con 'straordinario notturno'. "
    "Se nel contesto trovi 60% legato allo 'straordinario notturno', NON attribuirlo al notturno ordinario. "
    "Se il CCNL recuperato non indica chiaramente la percentuale del notturno ordinario, scrivi: "
    "'Non emerge dal CCNL recuperato la percentuale del notturno ordinario'.\n\n"

    "PER DOMANDE SU PAGAMENTO/MAGGIORAZIONI: "
    "prima chiarisci se si parla di lavoro ordinario, straordinario, notturno, festivo o combinazioni. "
    "Riporta percentuali SOLO se coerenti con quella tipologia nel contesto.\n\n"

    "CONSIGLIO PRATICO: chiudi sempre con 1–2 bullet operativi coerenti (RSU/UILCOM, HR, ordine di servizio, accordi IPZS se rilevanti).\n"
)


# ============================================================
# SIDEBAR CONTROLS
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

    if st.session_state.admin_ok and st.session_state.debug_on:
        st.caption("Debug attivo (solo admin).")


# ============================================================
# CHAT
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant" and st.session_state.admin_ok and st.session_state.debug_on and m.get("sources"):
            with st.expander("📚 Debug: estratti CCNL usati"):
                for s in m["sources"]:
                    st.write(f"**Foglio PDF {s.get('page','?')}**")
                    txt = s.get("text", "") or ""
                    st.write(txt[:900] + ("..." if len(txt) > 900 else ""))
                    st.divider()


user_input = st.chat_input("Scrivi una domanda sul CCNL (notturno, straordinari, permessi, malattia, ferie...)")
if not user_input:
    st.stop()

st.session_state.messages.append({"role": "user", "content": user_input})

if not (os.path.exists(VEC_PATH) and os.path.exists(META_PATH)):
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Prima devo indicizzare il CCNL: apri il menu a sinistra e clicca **Indicizza CCNL**.",
        "sources": []
    })
    st.rerun()


# ============================================================
# RETRIEVAL
# ============================================================
enriched_q = build_enriched_question(user_input)

vectors, meta = load_index()
mat_norm = normalize_rows(vectors)
emb = OpenAIEmbeddings()

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

# SUPER PASS permessi (se necessario)
if is_permessi_question(enriched_q) and (not is_rol_question(enriched_q)):
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

# Rerank: penalizza chunk "straordinario" se domanda è notturno ordinario
pay_cls = classify_pay_question(enriched_q)

for i in list(scores_best.keys()):
    txt = (meta[i].get("text", "") or "").lower()
    boost = 0.0

    if pay_cls["is_notturno_ordinario"]:
        # penalizza chunk che parlano di straordinario
        if "straordin" in txt:
            boost -= 0.20
        # premia chunk notturno+turni
        if "notturn" in txt:
            boost += 0.10
        if "turn" in txt or "avvicend" in txt:
            boost += 0.04

    if pay_cls["is_strao_notturno"]:
        if "straordin" in txt:
            boost += 0.10
        if "notturn" in txt:
            boost += 0.10

    if pay_cls["is_festivo"]:
        if "festiv" in txt or "domenic" in txt:
            boost += 0.08

    scores_best[i] = scores_best[i] + boost

final_idx = sorted(scores_best.keys(), key=lambda i: scores_best[i], reverse=True)[:TOP_K_FINAL]
selected = [meta[i] for i in final_idx]

# Optional BM25 on selected
if HAS_BM25 and len(selected) >= 6:
    docs_texts = [(c.get("text", "") or "") for c in selected]
    tokenized = [re.findall(r"\w+", t.lower()) for t in docs_texts]
    bm25 = BM25Okapi(tokenized)
    q_tokens = re.findall(r"\w+", enriched_q.lower())
    bm_scores = bm25.get_scores(q_tokens)
    order = np.argsort(-bm_scores)
    selected = [selected[int(j)] for j in order[:TOP_K_FINAL]]

context = "\n\n---\n\n".join([f"[Foglio PDF {c.get('page','?')}] {c.get('text','')}" for c in selected])

# Estrazione fatti paghe (anti-errore percentuali)
pay_facts = extract_pay_facts(selected)

# Costruiamo un blocco "fatti" da passare al modello (NON visibile agli utenti)
facts_block = json.dumps(pay_facts, ensure_ascii=False, indent=2)

# ============================================================
# LLM
# ============================================================
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = f"""
{RULES}

DOMANDA (UTENTE):
{user_input}

DOMANDA ARRICCHITA (MEMORIA BREVE):
{enriched_q}

FATTI ESTRATTI (INTERNO - usali per evitare errori percentuali):
- Sono già classificati per: notturno_ordinario / straordinario_notturno / straordinario_diurno / festivo / altro.
- Se una percentuale è in 'straordinario_notturno', NON usarla per il notturno ordinario.
{facts_block}

CONTESTO (estratti CCNL):
{context}

ISTRUZIONI DI OUTPUT (NON mostrare fonti):
Scrivi la risposta con questa struttura:

Risposta UILCOM:
(2–6 righe)

Dettagli:
(4–10 bullet; se è una domanda “pagamento”, prima chiarisci la tipologia: ordinario vs straordinario; notturno vs straordinario notturno; festivo ecc.)
- Inserisci percentuali SOLO se nel contesto/fatti estratti sono coerenti con la tipologia.
- Se l’utente chiede lavoro notturno ordinario e NON c’è la percentuale del notturno ordinario, devi dirlo chiaramente.

Consiglio pratico UILCOM:
(1–2 bullet)

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
    "sources": selected if (st.session_state.admin_ok and st.session_state.debug_on) else []
})
st.rerun()
