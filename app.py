import os
import json
import re
import numpy as np
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# =========================================================
# UILCOM IPZS — Chatbot CCNL (ONLINE) + Debug Admin + Fix Straordinari
# =========================================================

# =========================
# UI Header (UILCOM)
# =========================
st.set_page_config(page_title="Assistente Contrattuale UILCOM IPZS", page_icon="🟦")

st.title("🟦 Assistente Contrattuale UILCOM IPZS")
st.markdown(
    "**Accesso riservato agli iscritti UILCOM**  \n"
    "Strumento informativo per facilitare la consultazione del CCNL Grafici Editoria e norme applicabili ai lavoratori IPZS.  \n\n"
    "Le risposte sono generate **solo** in base al CCNL caricato.  \n"
    "Per casi specifici e interpretazioni, rivolgersi a RSU/UILCOM o struttura sindacale competente."
)
st.divider()

# =========================
# Config / Paths
# =========================
PDF_PATH = os.path.join("documenti", "ccnl.pdf")
INDEX_DIR = "index_ccnl"
VEC_PATH = os.path.join(INDEX_DIR, "vectors.npy")
META_PATH = os.path.join(INDEX_DIR, "chunks.json")

TOP_K_FINAL = 18
TOP_K_PER_QUERY = 12
MAX_MULTI_QUERIES = 12

# Memoria breve (ultime richieste utente)
MEMORY_USER_TURNS = 3

# Super-modulo permessi: soglia copertura (categorie diverse da trovare nei chunk)
PERMESSI_MIN_CATEGORY_COVERAGE = 3

# =========================
# Secrets / Env helpers
# =========================
def get_secret(name: str, default=None):
    try:
        v = st.secrets.get(name, None)  # type: ignore
        if v is not None:
            return v
    except Exception:
        pass
    return os.getenv(name, default)

OPENAI_API_KEY = get_secret("OPENAI_API_KEY", None)
UILCOM_PASSWORD = get_secret("UILCOM_PASSWORD", None)
ADMIN_PASSWORD = get_secret("UILCOM_ADMIN_PASSWORD", None)  # consigliato
if not ADMIN_PASSWORD:
    # fallback per uso locale se non impostata (non mostrata agli utenti)
    ADMIN_PASSWORD = get_secret("ADMIN_PASSWORD", "admin123")

# Blocca subito se manca la chiave OpenAI
if not OPENAI_API_KEY:
    st.error(
        "Manca la variabile **OPENAI_API_KEY**.\n\n"
        "• In locale: imposta OPENAI_API_KEY nelle variabili d'ambiente\n"
        "• Online (Streamlit Cloud): Settings → Secrets → OPENAI_API_KEY"
    )
    st.stop()

# =========================
# Password Gate (Iscritti)
# =========================
if "auth_ok" not in st.session_state:
    st.session_state.auth_ok = False

if UILCOM_PASSWORD:
    with st.expander("🔒 Accesso iscritti UILCOM", expanded=not st.session_state.auth_ok):
        pwd_in = st.text_input("Password iscritti", type="password", placeholder="Inserisci la password")
        if st.button("Entra"):
            if pwd_in == UILCOM_PASSWORD:
                st.session_state.auth_ok = True
                st.success("Accesso consentito.")
            else:
                st.session_state.auth_ok = False
                st.error("Password non corretta.")
else:
    # se non impostata (test locale), lascia entrare
    st.info("🔐 Password iscritti non impostata (solo test locale). Online usare Secrets.")
    st.session_state.auth_ok = True

if not st.session_state.auth_ok:
    st.stop()

# =========================
# Admin Debug Login (solo sidebar)
# =========================
if "is_admin" not in st.session_state:
    st.session_state.is_admin = False

with st.sidebar:
    st.markdown("### 🔐 Admin (solo UILCOM)")
    admin_pwd = st.text_input("Password admin", type="password", placeholder="Solo per debug")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Login"):
            if admin_pwd and admin_pwd == ADMIN_PASSWORD:
                st.session_state.is_admin = True
                st.success("Modalità admin attiva")
            else:
                st.error("Password admin errata")
    with col_b:
        if st.button("Logout"):
            st.session_state.is_admin = False
            st.info("Modalità admin disattivata")

    st.divider()
    st.header("⚙️ Controlli")
    st.caption("Se hai già indicizzato, non serve rifarlo (a meno che cambi il PDF).")

# =========================
# Retrieval helpers
# =========================
def normalize_rows(mat: np.ndarray) -> np.ndarray:
    return mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)

def cosine_scores(query_vec: np.ndarray, mat_norm: np.ndarray) -> np.ndarray:
    q = query_vec / (np.linalg.norm(query_vec) + 1e-12)
    return mat_norm @ q

# =========================
# Trigger dizionari
# =========================
CONSERVAZIONE_TRIGGERS = [
    "maternità", "maternita", "congedo maternità", "congedo maternita",
    "congedo parentale", "parentale",
    "malattia", "infortunio", "aspettativa",
    "assente", "assenza", "sostituzione", "sostituendo", "sto sostituendo",
]

MANSIONI_ALTE_TRIGGERS = [
    "mansioni più alte", "mansioni piu alte",
    "mansioni più elevate", "mansioni piu elevate",
    "mansioni superiori", "mansioni superiore", "mansione superiore",
    "sto facendo il lavoro", "mi fanno fare il lavoro", "mi stanno facendo fare",
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
    "quanti rol", "quante ore rol", "quante ex festività", "quante ex festivita",
]

STRAORDINARIO_TRIGGERS = [
    "straordin", "maggioraz", "maggiorazione",
    "notturn", "festiv", "festivo",
    "supplementare", "oltre orario", "lavoro oltre"
]

IPZS_TRIGGERS = [
    "ipzs", "poligrafico", "zecca",
    "accordo aziendale", "accordi aziendali",
    "ordine di servizio", "ods",
    "turni", "reparto", "linea", "impianto",
]

# =========================
# Super-modulo permessi: categorie + copertura
# =========================
PERMESSI_CATEGORIES = {
    "visite_mediche": [r"visite?\s+med", r"visita\s+med", r"accertament", r"specialist"],
    "lutto": [r"\blutto\b", r"decesso", r"grave\s+lutto", r"familiare"],
    "matrimonio": [r"matrimon", r"nozz"],
    "studio_formazione": [r"diritto\s+allo\s+studio", r"\b150\s+ore\b", r"\bstudio\b", r"\besami\b", r"formazion"],
    "legge_104": [r"\b104\b", r"legge\s*104", r"handicap"],
    "sindacali": [r"sindacal", r"\brsu\b", r"assemblea", r"permessi?\s+sindacal"],
    "donazione_sangue": [r"donazion", r"sangue", r"emocomponent"],
    "altri_permessi": [r"permessi?\s+retribuit", r"assenze?\s+retribuit", r"conged"],
}

def permessi_category_coverage(selected_chunks) -> tuple[int, list[str]]:
    found = set()
    joined = " ".join([(c.get("text", "") or "") for c in selected_chunks if isinstance(c, dict)]).lower()
    for cat, pats in PERMESSI_CATEGORIES.items():
        for p in pats:
            if re.search(p, joined, flags=re.IGNORECASE):
                found.add(cat)
                break
    return len(found), sorted(found)

def build_permessi_expansion_queries(user_q: str) -> list[str]:
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

# =========================
# Classificatori
# =========================
def is_mansioni_superiori_question(q: str) -> bool:
    ql = (q or "").lower()
    return any(t in ql for t in MANSIONI_ALTE_TRIGGERS)

def is_malattia_question(q: str) -> bool:
    ql = (q or "").lower()
    return any(t in ql for t in MALATTIA_TRIGGERS)

def is_permessi_question(q: str) -> bool:
    ql = (q or "").lower()
    return any(t in ql for t in PERMESSI_TRIGGERS)

def is_rol_question(q: str) -> bool:
    ql = (q or "").lower()
    return any(t in ql for t in ROL_TRIGGERS)

def is_straordinario_question(q: str) -> bool:
    ql = (q or "").lower()
    return any(t in ql for t in STRAORDINARIO_TRIGGERS)

def is_ipzs_context(q: str) -> bool:
    ql = (q or "").lower()
    return any(t in ql for t in IPZS_TRIGGERS)

# =========================
# MEMORIA BREVE (domanda arricchita)
# =========================
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

# =========================
# Multi-query principale (con modulo straordinario)
# =========================
def build_queries(q: str) -> list[str]:
    q0 = q.strip()
    qs = [q0]
    qlow = q0.lower()

    qs += [
        f"{q0} CCNL",
        f"{q0} regole condizioni",
        f"{q0} definizione procedura",
    ]

    user_is_rol = is_rol_question(q0)
    user_is_perm = is_permessi_question(q0)
    user_is_mal = is_malattia_question(q0)
    user_is_mans = is_mansioni_superiori_question(q0)
    user_is_straord = is_straordinario_question(q0)
    user_is_conserv = any(t in qlow for t in CONSERVAZIONE_TRIGGERS)

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

    # Straordinari (SUPER PRECISIONE)
    if user_is_straord:
        qs += [
            "lavoro straordinario maggiorazioni percentuali",
            "straordinario notturno maggiorazioni percentuali",
            "straordinario festivo maggiorazioni percentuali",
            "lavoro notturno festivo straordinario tabella maggiorazioni",
            "compenso straordinario come si calcola",
            "limiti e autorizzazione straordinario",
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

    # Dedup + limit
    out, seen = [], set()
    for x in qs:
        x = x.strip()
        if x and x not in seen:
            out.append(x)
            seen.add(x)

    return out[:MAX_MULTI_QUERIES]

# =========================
# Evidence extraction (robusto)
# =========================
def extract_key_evidence(chunks, only_straord: bool = False):
    """
    Estrae righe 'operative' dai chunk.
    Fix straordinario: se only_straord=True, ammette % SOLO se nella stessa riga c'è 'straordin' o 'maggioraz'.
    """
    evidences = []

    # patterns generali
    patterns = [
        r"\b30\b", r"\b60\b", r"\b\d{1,3}\b", r"\b%\b",
        r"posto\s+vacante", r"mansioni?\s+superiori?", r"sostituzion",
        r"conservazion.*posto", r"diritto.*conservazion.*posto",
        r"non\s+si\s+applica", r"non\s+costituisc",
        r"formazion", r"addestrament", r"affiancament",
        r"malatt", r"comporto", r"certificat", r"reperibil", r"visita\s+fiscale",
        r"permess", r"\brol\b", r"riduzione\s+orario", r"ex\s*fest", r"festivit",
        r"lutto", r"matrimon", r"nozz", r"\b104\b", r"sindacal", r"assemblea",
        r"straordin", r"maggioraz", r"notturn", r"festiv",
    ]

    for c in chunks:
        if not isinstance(c, dict):
            continue

        page = c.get("pdf_page", c.get("page", "?"))
        text = c.get("text", "") or ""

        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        for ln in lines:
            ln_low = ln.lower()

            if only_straord:
                # ✅ FIX AUTOMATICO: percentuali solo se agganciate a straordinario/maggiorazione nella stessa riga
                has_percent = ("%" in ln_low) or ("per cento" in ln_low)
                has_anchor = ("straordin" in ln_low) or ("maggioraz" in ln_low)
                if has_percent and has_anchor:
                    ln_clean = " ".join(ln.split())
                    if 10 <= len(ln_clean) <= 420:
                        evidences.append(f"(pdf {page}) {ln_clean}")
                continue

            if any(re.search(p, ln_low) for p in patterns):
                ln_clean = " ".join(ln.split())
                if 20 <= len(ln_clean) <= 420:
                    evidences.append(f"(pdf {page}) {ln_clean}")

    # Dedup + cut
    out, seen = [], set()
    for e in evidences:
        if e not in seen:
            out.append(e)
            seen.add(e)

    return out[:18]

def evidence_has_percent_straord(evidence_lines: list[str]) -> bool:
    joined = " ".join(evidence_lines).lower()
    return (("%" in joined) or ("per cento" in joined)) and (("straordin" in joined) or ("maggioraz" in joined))

# =========================
# Index build/load
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
    pdf_pages = [(c.metadata.get("page", 0) + 1) for c in chunks]

    emb = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    vectors = emb.embed_documents(texts)
    vectors = np.array(vectors, dtype=np.float32)

    np.save(VEC_PATH, vectors)

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(
            [{"pdf_page": p, "text": t} for p, t in zip(pdf_pages, texts)],
            f,
            ensure_ascii=False
        )

    return len(chunks)

def load_index():
    vectors = np.load(VEC_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # robustezza: garantisci lista di dict
    fixed = []
    for item in meta:
        if isinstance(item, dict):
            if "text" in item:
                fixed.append(item)
        elif isinstance(item, str):
            fixed.append({"pdf_page": "?", "text": item})
    return vectors, fixed

# =========================
# Rules (precisione contrattuale + niente fonti in risposta)
# =========================
rules = (
    "Sei l’assistente UILCOM per lavoratori IPZS. "
    "Rispondi in modo chiaro, professionale e pratico basandoti SOLO sul contesto del CCNL fornito. "
    "Non inventare informazioni.\n\n"

    "REGOLA PROVA: puoi affermare numeri/percentuali/durate SOLO se sono presenti nel contesto recuperato. "
    "Se un numero/percentuale non è dimostrabile nel contesto, devi dire che non emerge dal CCNL recuperato.\n\n"

    "STRAORDINARI (FIX): se la domanda riguarda lo straordinario, usa percentuali SOLO se nel contesto la percentuale "
    "è nella stessa riga/frase di 'straordinario' o 'maggiorazione'. "
    "Se trovi percentuali non agganciate, ignorale.\n\n"

    "REGOLA GENERALE: quando la domanda è ampia (malattia/ferie/straordinari/permessi/livelli/trasferimenti), "
    "organizza per sotto-temi e riassumi solo ciò che emerge.\n\n"

    "OBBLIGO LIMITAZIONI: se nel contesto compaiono frasi come 'non si applica', 'non costituisce', 'sostituzione', "
    "'diritto alla conservazione del posto', devi riportarle chiaramente.\n\n"

    "OUTPUT: NON mostrare all'utente pagine o citazioni. "
    "Devi però essere preciso: se qualcosa non è nel contesto, dichiaralo."
)

# =========================
# Sidebar actions: indicizzazione
# =========================
with st.sidebar:
    if st.button("Indicizza CCNL (prima volta)"):
        try:
            with st.spinner("Indicizzazione in corso..."):
                n = build_index()
            st.success(f"Indicizzazione completata! Chunk creati: {n}")
        except Exception as e:
            st.error(str(e))

    ok_index = os.path.exists(VEC_PATH) and os.path.exists(META_PATH)
    st.write("📦 Indice presente:", "✅" if ok_index else "❌")

    if st.button("🧹 Nuova chat"):
        st.session_state.messages = []
        st.rerun()

# =========================
# Chat UI (stile ChatGPT)
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_input = st.chat_input("Scrivi una domanda sul CCNL (ferie, malattia, permessi, straordinari, livelli...)")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    if not (os.path.exists(VEC_PATH) and os.path.exists(META_PATH)):
        msg = "Prima devo indicizzare il CCNL: apri il menu a sinistra e clicca **Indicizza CCNL (prima volta)**."
        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.rerun()

    enriched_q = build_enriched_question(user_input)

    vectors, meta = load_index()
    mat_norm = normalize_rows(vectors)
    emb = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

    user_is_perm = is_permessi_question(enriched_q)
    user_is_rol = is_rol_question(enriched_q)
    user_is_mal = is_malattia_question(enriched_q)
    user_is_mans = is_mansioni_superiori_question(enriched_q)
    user_is_straord = is_straordinario_question(enriched_q)
    user_mentions_ipzs = is_ipzs_context(enriched_q)

    # =========================
    # Pass 1 retrieval (multi-query)
    # =========================
    queries = build_queries(enriched_q)

    scores_best = {}
    for q in queries:
        qvec = np.array(emb.embed_query(q), dtype=np.float32)
        sims = cosine_scores(qvec, mat_norm)
        top_idx = np.argsort(-sims)[:TOP_K_PER_QUERY]
        for i in top_idx:
            s = float(sims[i])
            if (i not in scores_best) or (s > scores_best[i]):
                scores_best[i] = s

    provisional_idx = sorted(scores_best.keys(), key=lambda i: scores_best[i], reverse=True)[:TOP_K_FINAL]
    provisional_selected = [meta[i] for i in provisional_idx if i < len(meta) and isinstance(meta[i], dict)]

    # =========================
    # SUPER PASS 2 permessi: se copertura bassa, espandi categorie
    # =========================
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

    # =========================
    # Re-ranking con boost (include FIX straordinari)
    # =========================
    for i in list(scores_best.keys()):
        if i >= len(meta) or not isinstance(meta[i], dict):
            continue

        txt = (meta[i].get("text", "") or "").lower()
        boost = 0.0

        # Straordinario: BOOST forte su righe con % agganciata
        if user_is_straord:
            if (("straordin" in txt or "maggioraz" in txt) and (("%" in txt) or ("per cento" in txt))):
                boost += 0.22
            if ("tabella" in txt) or ("maggiorazioni" in txt):
                boost += 0.06

        # Permessi generici
        if user_is_perm and (not user_is_rol):
            if "permess" in txt or "assenze retribuite" in txt:
                boost += 0.06
            for _, pats in PERMESSI_CATEGORIES.items():
                if any(re.search(p, txt, flags=re.IGNORECASE) for p in pats):
                    boost += 0.02
                    break

        # ROL / ex festività
        if user_is_rol:
            if re.search(r"\brol\b", txt) or "riduzione orario" in txt:
                boost += 0.12
            if "ex festiv" in txt or "festività soppresse" in txt or "festivita soppresse" in txt:
                boost += 0.12
            if "diritto allo studio" in txt or "150 ore" in txt:
                boost -= 0.10

        # Malattia
        if user_is_mal:
            if "comporto" in txt:
                boost += 0.07
            if "malatt" in txt:
                boost += 0.05
            if "%" in txt or "percent" in txt or "trattamento econom" in txt:
                boost += 0.05
            if "certificat" in txt or "comunicaz" in txt:
                boost += 0.03

        # Mansioni superiori
        if user_is_mans:
            if re.search(r"\b30\b", txt) and re.search(r"\b60\b", txt):
                boost += 0.12
            if "conservazione del posto" in txt or "diritto alla conservazione" in txt:
                boost += 0.06
            if "non si applica" in txt or "non si applicano" in txt:
                boost += 0.06

        scores_best[i] = scores_best[i] + boost

    final_idx = sorted(scores_best.keys(), key=lambda i: scores_best[i], reverse=True)[:TOP_K_FINAL]
    selected = [meta[i] for i in final_idx if i < len(meta) and isinstance(meta[i], dict)]

    # Context (solo dict)
    context_parts = []
    for c in selected:
        t = c.get("text", "")
        p = c.get("pdf_page", "?")
        context_parts.append(f"[PDF {p}] {t}")
    context = "\n\n---\n\n".join(context_parts)

    # Evidence
    key_evidence = extract_key_evidence(selected, only_straord=user_is_straord)
    evidence_block = "\n".join([f"- {e}" for e in key_evidence]) if key_evidence else "- (Nessuna evidenza estratta automaticamente)"

    # Guardrail: se domanda straordinario ma non troviamo % agganciate → obbliga prudenza
    straord_guardrail = ""
    if user_is_straord and not evidence_has_percent_straord(key_evidence):
        straord_guardrail = (
            "\nATTENZIONE STRAORDINARIO: non risultano percentuali 'agganciate' a straordinario/maggiorazione nelle evidenze. "
            "Quindi NON devi inventare percentuali: devi dire che nel CCNL recuperato non emergono chiaramente le maggiorazioni "
            "e suggerire di verificare la tabella/parte specifica nel testo.\n"
        )

    # Nota IPZS
    ipzs_note = ""
    if user_mentions_ipzs:
        ipzs_note = (
            "\nNOTA IPZS: se serve, segnala che turni/prassi/accordi aziendali possono integrare la disciplina del CCNL.\n"
        )

    # =========================
    # Admin Debug Panel (professionale)
    # =========================
    if st.session_state.get("is_admin", False):
        with st.expander("🛠 Debug admin (UILCOM) — visibile solo a te", expanded=False):
            st.markdown("**Domanda utente:**")
            st.code(user_input)

            st.markdown("**Domanda arricchita (memoria):**")
            st.code(enriched_q)

            st.markdown("**Query usate (multi-query):**")
            for q in queries:
                st.write("•", q)

            st.markdown("**Evidence estratte (prioritarie):**")
            if key_evidence:
                for e in key_evidence:
                    st.write(e)
            else:
                st.write("— nessuna evidenza")

            st.markdown("**Chunk selezionati (top):**")
            for i, c in enumerate(selected, start=1):
                st.write(f"**Chunk {i} — PDF {c.get('pdf_page','?')}**")
                st.write((c.get("text", "") or "")[:1200])
                st.divider()

            st.markdown("**Nota fix straordinari:**")
            st.write("• Percentuali ammesse solo se nella stessa riga ci sono 'straordin'/'maggioraz'.")
            st.write("• Se non trovate, risposta prudente (no invenzioni).")

    # =========================
    # LLM
    # =========================
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)

    # Prompt: interno con evidenze e pagine, ma risposta senza fonti
    prompt = f"""
{rules}
{straord_guardrail}
{ipzs_note}

DOMANDA UTENTE:
{user_input}

DOMANDA ARRICCHITA (memoria breve):
{enriched_q}

EVIDENZE (interne, da usare per precisione):
{evidence_block}

CONTESTO (estratti CCNL):
{context}

SCRIVI LA RISPOSTA CON QUESTA STRUTTURA (SENZA CITARE PAGINE O FONTI):

Risposta UILCOM:
(2–5 righe, chiara)

Dettagli operativi:
(4–10 punti max; se c'è una esclusione/limitazione che cambia l'esito mettila come primo punto)

Cosa significa per te:
(1–2 righe pratiche)

Consiglio pratico UILCOM:
(1–2 bullet brevi)

Nota UILCOM:
Questa risposta è informativa; per casi specifici verificare con RSU/UILCOM o HR e con il testo ufficiale.

REGOLE DI SICUREZZA:
- Non inventare percentuali o numeri.
- Se non trovi nel contesto, scrivi che non emerge dal CCNL recuperato.
"""

    try:
        answer = llm.invoke(prompt).content
    except Exception as e:
        answer = f"Errore nel generare la risposta: {e}"

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()
