import os
import json
import re
import numpy as np
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# =========================
# UI Header (UILCOM)
# =========================
st.set_page_config(page_title="Assistente Contrattuale UILCOM IPZS", page_icon="🟦")

st.title("🟦 Assistente Contrattuale UILCOM IPZS")
st.markdown(
    "**Accesso riservato agli iscritti UILCOM**  \n"
    "Questo servizio digitale è uno strumento informativo messo a disposizione dalla UILCOM per facilitare la consultazione "
    "del CCNL Grafici Editoria e delle principali norme contrattuali applicabili ai lavoratori.  \n\n"
    "Le risposte fornite dall’assistente sono basate sul contenuto del contratto collettivo caricato nel sistema e hanno finalità esclusivamente informative.  \n"
    "Per situazioni specifiche, interpretazioni contrattuali o verifiche individuali si raccomanda di rivolgersi alle RSU UILCOM o alla struttura sindacale competente."
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

# Memoria breve
MEMORY_USER_TURNS = 3

# Super-modulo permessi: soglia copertura (quante categorie diverse vogliamo trovare nei chunk)
PERMESSI_MIN_CATEGORY_COVERAGE = 3

# =========================
# Password Gate (Solo iscritti)
# =========================
def get_secret_or_env(name: str):
    val = None
    try:
        val = st.secrets.get(name, None)  # type: ignore
    except Exception:
        val = None
    if not val:
        val = os.getenv(name)
    return val

required_pwd = get_secret_or_env("UILCOM_PASSWORD")
admin_pwd = get_secret_or_env("ADMIN_PASSWORD")

if "auth_ok" not in st.session_state:
    st.session_state.auth_ok = False

if "admin_ok" not in st.session_state:
    st.session_state.admin_ok = False

# UILCOM login
if required_pwd:
    with st.expander("🔒 Accesso iscritti UILCOM", expanded=not st.session_state.auth_ok):
        pwd_in = st.text_input("Password iscritti", type="password", placeholder="Inserisci la password iscritti")
        if st.button("Entra"):
            if pwd_in == required_pwd:
                st.session_state.auth_ok = True
                st.success("Accesso consentito.")
            else:
                st.session_state.auth_ok = False
                st.error("Password non corretta.")
else:
    # locale/test
    st.info("🔐 Password iscritti non impostata (in locale puoi usare UILCOM_PASSWORD; online useremo Secrets).")
    st.session_state.auth_ok = True

if not st.session_state.auth_ok:
    st.stop()

# Admin login (solo per vedere debug/fonti)
if admin_pwd:
    with st.sidebar.expander("🛠️ Admin (debug)", expanded=False):
        ap = st.text_input("Admin password", type="password", placeholder="Solo amministratore")
        if st.button("Accedi come admin"):
            if ap == admin_pwd:
                st.session_state.admin_ok = True
                st.success("Admin attivo.")
            else:
                st.session_state.admin_ok = False
                st.error("Admin password errata.")
else:
    # se non impostata, niente admin
    st.session_state.admin_ok = False

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
    "malattia", "infortunio",
    "aspettativa",
    "assente", "assenza", "sostituzione", "sostituendo", "sto sostituendo",
]

MANSIONI_ALTE_TRIGGERS = [
    "mansioni più alte", "mansioni piu alte",
    "mansioni più elevate", "mansioni piu elevate",
    "mansioni superiori", "mansioni superiore", "mansione superiore",
    "lavoro più alto", "lavoro piu alto",
    "sto facendo il lavoro", "mi fanno fare il lavoro", "mi stanno facendo fare",
    "sto facendo mansioni", "faccio mansioni", "faccio il lavoro di",
    "capoturno", "capo turno", "caporeparto", "capo reparto",
    "sostituisco", "sto sostituendo",
    "livello superiore", "categoria superiore", "inquadramento superiore",
    "passaggio di livello", "passaggio categoria", "avanzamento", "promozione",
    "posto vacante",
]

# ✅ UPGRADE DIFFERENZA PAGA
DIFF_PAGA_TRIGGERS = [
    "differenza paga", "differenze paga", "differenza pag", "differenza retribut",
    "differenze retribut", "differenza stipend", "più soldi", "piu soldi",
    "mi spetta di più", "mi spetta di piu", "mi devono pagare di più", "mi devono pagare di piu",
    "mi devono pagare di più", "pagarmi di più", "pagarmi di piu",
    "adeguamento retributivo", "trattamento economico", "trattamento retributivo",
    "quanto mi spetta in più", "quanto mi spetta in piu", "arretrati", "arretrato",
    "conguaglio", "differenze in busta paga", "busta paga differenze",
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

# =========================
# Super-modulo permessi: categorie + copertura
# =========================
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

def permessi_category_coverage(selected_chunks) -> tuple[int, list[str]]:
    found = set()
    joined = " ".join([(c.get("text", "") or "") for c in selected_chunks]).lower()
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
def is_mansioni_superiori_question(user_q: str) -> bool:
    ql = user_q.lower()
    return any(t in ql for t in MANSIONI_ALTE_TRIGGERS)

def is_diff_paga_question(user_q: str) -> bool:
    ql = user_q.lower()
    return any(t in ql for t in DIFF_PAGA_TRIGGERS)

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
# Multi-query principale
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
    user_is_diff = is_diff_paga_question(q0)
    user_is_conserv = any(t in qlow for t in CONSERVAZIONE_TRIGGERS)

    # ✅ DIFFERENZA PAGA: aggancio forte a mansioni superiori / trattamento economico
    if user_is_diff:
        qs += [
            "differenze retributive mansioni superiori trattamento economico corrispondente",
            "trattamento economico corrispondente all'attività svolta mansioni superiori",
            "assegnazione mansioni superiori diritto alla retribuzione corrispondente",
            "arretrati differenze retributive mansioni superiori",
            "passaggio livello categoria mansioni superiori retribuzione",
        ]

    # ROL / ex festività
    if user_is_rol:
        qs += [
            "ROL riduzione orario di lavoro monte ore annuo maturazione fruizione",
            "ex festività festività soppresse permessi ore giorni spettanti",
            "permessi ROL ed ex festività: quanti, come maturano e come si usano",
            "ROL ex festività richiesta fruizione preavviso eventuale programmazione",
            "ROL ex festività residui scadenze eventuale monetizzazione (se prevista)",
            "permessi annui retribuiti riduzione orario (ROL) ex festività",
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

    # Straordinari / notturno
    if any(k in qlow for k in ["straordin", "maggior", "notturn", "festiv", "banca ore"]):
        qs += [
            "lavoro straordinario maggiorazioni limiti autorizzazione",
            "straordinario notturno festivo maggiorazioni",
            "lavoro notturno maggiorazione (non straordinario) percentuale",
            "banca ore recuperi straordinario se previsti",
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

    # Mansioni superiori / sostituzioni (anche per differenza paga)
    if user_is_mans or user_is_conserv or user_is_diff:
        qs += [
            "mansioni superiori posto vacante",
            "assegnazione a mansioni superiori regole generali",
            "mansioni superiori 30 giorni consecutivi 60 discontinui",
            "non si applica in caso di sostituzione di dipendente assente con diritto alla conservazione del posto",
            "formazione addestramento affiancamento non costituisce mansioni superiori",
            "trattamento economico durante mansioni superiori",
        ]

    out, seen = [], set()
    for x in qs:
        x = x.strip()
        if x and x not in seen:
            out.append(x)
            seen.add(x)
    return out[:MAX_MULTI_QUERIES]

# =========================
# Evidence extraction
# =========================
def extract_key_evidence(chunks):
    """
    chunks: list[dict] con chiavi page/text (robusto anche se arrivano stringhe)
    """
    evidences = []
    patterns = [
        r"\b30\b", r"\b60\b", r"\b\d{1,3}\b", r"\b%\b",
        r"posto\s+vacante", r"mansioni?\s+superiori?", r"sostituzion",
        r"conservazion.*posto", r"diritto.*conservazion.*posto",
        r"non\s+si\s+applica", r"non\s+costituisc",
        r"formazion", r"addestrament", r"affiancament",
        r"trattamento\s+econom", r"trattamento\s+retribut", r"retribuzion", r"differenz",
        r"malatt", r"comporto", r"certificat", r"reperibil", r"visita\s+fiscale",
        r"permess", r"\brol\b", r"riduzione\s+orario", r"ex\s*fest", r"festivit",
        r"lutto", r"matrimon", r"nozz", r"\b104\b", r"sindacal", r"assemblea",
        r"\b3\s*mesi\b", r"tre\s+mesi",
    ]

    def line_is_interesting(ln_low: str) -> bool:
        return any(re.search(p, ln_low) for p in patterns)

    for c in chunks:
        # robustezza: se per qualche motivo arriva una stringa
        if isinstance(c, str):
            page = "?"
            text = c
        else:
            page = c.get("page", "?")
            text = c.get("text", "")

        lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
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

def evidence_has_30_60(evidence_lines: list[str]) -> bool:
    joined = " ".join(evidence_lines).lower()
    return (re.search(r"\b30\b", joined) is not None) and (re.search(r"\b60\b", joined) is not None)

def evidence_mentions_three_months(evidence_lines: list[str]) -> bool:
    joined = " ".join(evidence_lines).lower()
    return ("tre mesi" in joined) or (re.search(r"\b3\s*mesi\b", joined) is not None)

def text_mentions_three_months(txt: str) -> bool:
    t = (txt or "").lower()
    return ("tre mesi" in t) or (re.search(r"\b3\s*mesi\b", t) is not None)

# =========================
# Index build/load (robusto)
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
    pages = [(c.metadata.get("page", 0) + 1) for c in chunks]  # numero foglio PDF

    emb = OpenAIEmbeddings()
    vectors = emb.embed_documents(texts)
    vectors = np.array(vectors, dtype=np.float32)

    np.save(VEC_PATH, vectors)

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump([{"page": p, "text": t} for p, t in zip(pages, texts)], f, ensure_ascii=False)

    return len(chunks)

def load_index():
    vectors = np.load(VEC_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # robustezza: se per qualche motivo chunks.json contiene lista di stringhe
    if isinstance(meta, list) and meta and isinstance(meta[0], str):
        meta = [{"page": "?", "text": s} for s in meta]
    # robustezza: se ci sono elementi non-dict
    if isinstance(meta, list):
        fixed = []
        for it in meta:
            if isinstance(it, dict):
                fixed.append({"page": it.get("page", "?"), "text": it.get("text", "")})
            else:
                fixed.append({"page": "?", "text": str(it)})
        meta = fixed

    return vectors, meta

# =========================
# UILCOM rules (super precisione + differenza paga)
# =========================
rules = (
    "Sei l’assistente UILCOM per lavoratori IPZS. "
    "Rispondi in modo chiaro, professionale e pratico basandoti SOLO sul contesto del CCNL fornito. "
    "Non inventare informazioni.\n\n"

    "REGOLA PROVA: puoi elencare una categoria (es. lutto, visite mediche, 104, matrimonio, sindacali, donazione sangue, studio) "
    "SOLO se nel contesto recuperato c'è almeno un riferimento testuale. "
    "Se non emerge dal contesto, NON affermare che esiste.\n\n"

    "OBBLIGO NUMERI E LIMITAZIONI: se nel contesto compaiono soglie numeriche o frasi di esclusione/limitazione "
    "(es. 'non si applica', 'non costituisce', 'sostituzione', 'diritto alla conservazione del posto', 'affiancamento/formazione'), "
    "riportale esplicitamente.\n\n"

    "MODULO DIFFERENZA PAGA (IMPORTANTISSIMO): se l’utente chiede 'differenza paga', 'mi devono pagare di più', 'arretrati', "
    "'adeguamento retributivo', devi cercare nel contesto i passaggi su: "
    "(1) 'trattamento economico/retributivo corrispondente all’attività svolta' durante mansioni superiori, "
    "(2) quando scatta/consolida l’assegnazione (30 giorni consecutivi / 60 discontinui se presenti), "
    "(3) esclusione in caso di sostituzione di assente con diritto alla conservazione del posto, "
    "(4) se mancano importi o percentuali nel CCNL, dichiaralo e NON inventare cifre.\n\n"

    "MODULO MALATTIA: quando la domanda riguarda la malattia, se presenti nel contesto includi SEMPRE: "
    "(1) trattamento economico/percentuali, (2) periodo di comporto e conteggio, (3) obblighi certificazione/comunicazione, "
    "(4) eventuali controlli/reperibilità, (5) casi particolari. "
    "Se una voce non emerge dal contesto, dillo.\n\n"

    "SUPER MODULO PERMESSI: quando la domanda riguarda 'permessi retribuiti' in generale (oltre a ROL/ex festività), "
    "devi cercare e riassumere PIÙ categorie (non una sola) ma SOLO se emergono dal contesto.\n\n"

    "FIX ROL/EX FESTIVITÀ: se l’utente chiede ROL o ex festività, "
    "rispondi su ROL/ex festività e NON confondere con permessi studio.\n\n"

    "REGOLA CHIAVE (SOSTITUZIONI): se la domanda riguarda la sostituzione di un lavoratore assente con diritto alla conservazione del posto "
    "(es. maternità, infortunio, malattia, aspettativa), specifica che la regola delle mansioni superiori NON si applica "
    "ai fini dell’inquadramento definitivo, se ciò risulta dal contesto.\n\n"

    "PRIORITÀ NORMATIVA (MANSIONI SUPERIORI): se nel contesto sono presenti le soglie 30 giorni consecutivi / 60 discontinui, "
    "queste hanno priorità. Non sostituirle con altre durate se non chiaramente collegate.\n\n"

    "Se non trovi la risposta, scrivi: 'Non ho trovato la risposta nel CCNL caricato'.\n\n"

    "CONSIGLIO PRATICO: alla fine inserisci sempre una sezione 'Consiglio pratico UILCOM' con 1–2 bullet operativi (brevi). "
    "Non dare consigli legali personalizzati.\n"
)

# =========================
# Sidebar
# =========================
with st.sidebar:
    st.header("⚙️ Controlli")

    ok_index = os.path.exists(VEC_PATH) and os.path.exists(META_PATH)
    st.write("📦 Indice presente:", "✅" if ok_index else "❌")

    if st.button("Indicizza CCNL (prima volta / dopo cambio PDF)"):
        try:
            with st.spinner("Indicizzazione in corso..."):
                n = build_index()
            st.success(f"Indicizzazione completata! Chunk creati: {n}")
        except Exception as e:
            st.error(str(e))

    if st.button("🧹 Nuova chat"):
        st.session_state.messages = []
        st.rerun()

    if st.session_state.admin_ok:
        st.divider()
        st.caption("Admin: debug attivo (visibile solo a te).")

# =========================
# Chat UI (tipo ChatGPT)
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

        # Fonti/Debug SOLO admin
        if st.session_state.admin_ok and m["role"] == "assistant" and m.get("sources"):
            with st.expander("📚 Fonti CCNL (solo admin)"):
                for s in m["sources"]:
                    if isinstance(s, dict):
                        st.write(f"**Foglio PDF {s.get('page', '?')}**")
                        txt = s.get("text", "")
                        st.write(txt[:900] + ("..." if len(txt) > 900 else ""))
                        st.divider()
                    else:
                        st.write(str(s)[:900])
                        st.divider()

        if st.session_state.admin_ok and m["role"] == "assistant" and m.get("debug"):
            with st.expander("🧠 Debug retrieval (solo admin)"):
                st.json(m["debug"])

user_input = st.chat_input("Scrivi una domanda sul CCNL (ferie, malattia, permessi, straordinari, livelli...)")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    if not (os.path.exists(VEC_PATH) and os.path.exists(META_PATH)):
        msg = "Prima devo indicizzare il CCNL: apri il menu a sinistra e clicca **Indicizza CCNL**."
        st.session_state.messages.append({"role": "assistant", "content": msg, "sources": []})
        st.rerun()

    enriched_q = build_enriched_question(user_input)

    vectors, meta = load_index()
    mat_norm = normalize_rows(vectors)
    emb = OpenAIEmbeddings()

    user_is_mans = is_mansioni_superiori_question(enriched_q)
    user_is_diff = is_diff_paga_question(enriched_q)
    user_is_mal = is_malattia_question(enriched_q)
    user_is_perm = is_permessi_question(enriched_q)
    user_is_rol = is_rol_question(enriched_q)
    user_mentions_ipzs = is_ipzs_context(enriched_q)

    # =========================
    # Pass 1 retrieval
    # =========================
    queries = build_queries(enriched_q)

    scores_best = {}
    debug_top = []

    for q in queries:
        qvec = np.array(emb.embed_query(q), dtype=np.float32)
        sims = cosine_scores(qvec, mat_norm)
        top_idx = np.argsort(-sims)[:TOP_K_PER_QUERY]
        debug_top.append({"query": q, "top_idx": top_idx[:5].tolist()})

        for i in top_idx:
            s = float(sims[i])
            if (i not in scores_best) or (s > scores_best[i]):
                scores_best[i] = s

    provisional_idx = sorted(scores_best.keys(), key=lambda i: scores_best[i], reverse=True)[:TOP_K_FINAL]
    provisional_selected = [meta[i] for i in provisional_idx]
    provisional_evidence = extract_key_evidence(provisional_selected)

    # Guardrail 30/60 mansioni
    guardrail_30_60 = evidence_has_30_60(provisional_evidence) and (user_is_mans or user_is_diff)

    # =========================
    # SUPER PASS 2 (solo permessi generici): se copertura bassa, rilanciamo query per categorie
    # =========================
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
                "\nGUARDRAIL PERMESSI: la domanda è generale sui permessi retribuiti. "
                "Devi riportare SOLO le categorie che emergono dal contesto. "
                "Se una categoria non emerge, dichiaralo come 'Non emerso dal CCNL recuperato'.\n"
            )

    # =========================
    # Re-ranking con boost (precisione)
    # =========================
    for i in list(scores_best.keys()):
        txt = (meta[i].get("text", "") or "").lower()
        boost = 0.0

        # Mansioni superiori / differenza paga: super boost su trattamento economico + 30/60 + esclusioni
        if user_is_mans or user_is_diff:
            if "trattamento econom" in txt or "trattamento retribut" in txt or "retribuzion" in txt or "differenz" in txt:
                boost += 0.10
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

        # Permessi generici
        if user_is_perm and (not user_is_rol):
            if "permess" in txt or "assenze retribuite" in txt:
                boost += 0.07
            for cat, pats in PERMESSI_CATEGORIES.items():
                for p in pats:
                    if re.search(p, txt, flags=re.IGNORECASE):
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

        scores_best[i] = scores_best[i] + boost

    final_idx = sorted(scores_best.keys(), key=lambda i: scores_best[i], reverse=True)[:TOP_K_FINAL]
    selected = [meta[i] for i in final_idx]

    context = "\n\n---\n\n".join([f"[Foglio PDF {c.get('page','?')}] {c.get('text','')}" for c in selected])

    key_evidence = extract_key_evidence(selected)
    evidence_block = "\n".join([f"- {e}" for e in key_evidence]) if key_evidence else "- (Nessuna evidenza estratta automaticamente: usare comunque il contesto.)"

    # Guardrail note mansioni/diff paga
    force_30_60_priority = evidence_has_30_60(key_evidence) and (user_is_mans or user_is_diff)
    has_3_months = evidence_mentions_three_months(key_evidence) and (user_is_mans or user_is_diff)

    guardrail_note = ""
    if force_30_60_priority:
        guardrail_note = (
            "\nGUARDRAIL MANSIONI/DIFFERENZA PAGA: nelle evidenze compaiono 30 giorni continuativi / 60 giorni non continuativi. "
            "Devi usare questi valori per indicare quando scatta/consolida l’assegnazione definitiva. "
            "NON usare 'tre mesi/3 mesi' se non è nello stesso passaggio normativo sulle mansioni superiori. "
            "Se compaiono sia 30/60 sia 3 mesi, prevale 30/60.\n"
        )
    elif has_3_months:
        guardrail_note = (
            "\nNOTA: nelle evidenze compare anche un riferimento a 'tre mesi/3 mesi'. "
            "Usalo SOLO se nel contesto è chiaramente riferito alle mansioni superiori. "
            "Se non è chiaro, non usarlo.\n"
        )

    rol_note = ""
    if user_is_rol:
        rol_note = (
            "\nNOTA ROL/EX FESTIVITÀ: L’utente chiede ROL o ex festività. "
            "Non parlare di permessi studio a meno che l’utente lo chieda esplicitamente. "
            "Se non emergono numeri precisi (ore/giorni) dal contesto recuperato, dichiaralo.\n"
        )

    ipzs_note = ""
    if user_mentions_ipzs:
        ipzs_note = (
            "\nNOTA IPZS: Se la domanda può dipendere da prassi o accordi aziendali (turni, ODS, procedure interne), "
            "segnalalo nella risposta e nel Consiglio pratico.\n"
        )

    perm_format_note = ""
    if user_is_perm and (not user_is_rol):
        perm_format_note = (
            "\nFORMATO PERMESSI: elenca più categorie SOLO se emergono nel contesto. "
            "Se una categoria non emerge, inseriscila in 'Non emerso dal CCNL recuperato' senza affermare che esista.\n"
        )

    diff_paga_note = ""
    if user_is_diff:
        diff_paga_note = (
            "\nFOCUS DIFFERENZA PAGA: rispondi chiaramente se (e quando) spetta il trattamento economico del livello/mansione svolta. "
            "Se il CCNL non fornisce importi/valori economici precisi, dillo esplicitamente e NON inventare cifre.\n"
        )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = f"""
{rules}
{guardrail_note}
{diff_paga_note}
{perm_guardrail_note}
{perm_format_note}
{rol_note}
{ipzs_note}

DOMANDA (UTENTE):
{user_input}

DOMANDA ARRICCHITA (MEMORIA BREVE - per capire il contesto):
{enriched_q}

EVIDENZE OBBLIGATORIE (se presenti):
Devi riportare esplicitamente queste regole/numeri/limitazioni nella risposta, perché sono operative.
{evidence_block}

CONTESTO (estratti CCNL):
{context}

SCRIVI LA RISPOSTA CON QUESTA STRUTTURA:

Risposta UILCOM:
(2–5 righe)

Dettagli operativi:
(4–10 punti massimo; se c'è una esclusione/limitazione che cambia l'esito — es. conservazione del posto, "non si applica" — mettila come primo punto)

Non emerso dal CCNL recuperato:
(se serve, 0–3 bullet)

Consiglio pratico UILCOM:
(1–2 bullet operativi, brevi, coerenti)

Nota UILCOM:
Questa risposta è informativa; per casi specifici verificare con RSU/UILCOM o HR e con il testo ufficiale.

RISPOSTA:
"""

    try:
        answer = llm.invoke(prompt).content
    except Exception as e:
        answer = f"Errore nel generare la risposta: {e}"

    debug_payload = None
    if st.session_state.admin_ok:
        debug_payload = {
            "user_input": user_input,
            "enriched_q": enriched_q[:5000],
            "queries_used": queries,
            "top_queries_preview": debug_top[:6],
            "selected_fogli_pdf": [s.get("page", "?") for s in selected if isinstance(s, dict)],
            "force_30_60_priority": bool(force_30_60_priority),
            "is_diff_paga": bool(user_is_diff),
            "is_mansioni": bool(user_is_mans),
        }

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": selected, "debug": debug_payload}
    )
    st.rerun()
