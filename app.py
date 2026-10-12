
import re
import html
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)

import io
import pandas as pd
import streamlit as st

try:
    import plotly.graph_objects as go
except ImportError:
    go = None

from agents.scope_agent import scope_agent
from agents.risk_agent import risk_agent
from agents.blocker_agent import blocker_agent
from agents.document_agent import documentation_agent

from project_health.health_score import (
    analyze_project_health,
    calculate_health_score,
    get_health_status
)

from assistant.project_assistent import project_assistant

from ingestion.loader import extract_text
from rag.chunking import create_chunks
from rag.embeddings import generate_embeddings
import rag.vector_store as vector_store_module

from rag.vector_store import (
    clear_collection,
    store_documents,
    search_documents,
    get_documents_for_sources,
    get_document_count,
    get_unique_document_count,
    set_active_project,
    get_project_sources
)



def retrieve_agent_context(agent_type, per_query=6):
    """Retrieve evidence for any project type from the active project's ChromaDB.

    Combines focused semantic retrieval with a lexical pass over the active project's
    stored chunks. Queries intentionally avoid assumptions about a particular domain.
    """
    query_sets = {
        "scope": [
            "project purpose problem statement objectives goals and success measures",
            "requirements features functions user needs constraints and acceptance criteria",
            "deliverables outputs scope inclusions exclusions and assumptions",
            "milestones phases schedule timeline duration and target dates",
            "team roles responsibilities stakeholders and ownership",
            "project background context and proposed solution",
        ],
        "risks": [
            "explicit project risks risk register probability impact mitigation and owner",
            "project assumptions constraints dependencies and external factors that may cause problems",
            "schedule delays missed dates incomplete tasks and milestone slippage",
            "quality defects testing gaps security privacy compliance and operational concerns",
            "open questions pending approvals unresolved decisions and uncertainty",
            "resource capacity budget scope changes communication and stakeholder risks",
            "risk response contingency prevention monitoring and escalation actions",
        ],
        "blockers": [
            "explicit blockers impediments obstacles and items preventing progress",
            "open issues unanswered questions pending decisions and approvals",
            "action items tasks assigned owners due dates status and next steps",
            "dependencies waiting items missing information and required access or resources",
            "meeting notes decisions follow ups and unresolved topics",
            "task status delayed overdue not started in progress and acceptance evidence",
        ],
    }
    queries = query_sets.get(agent_type, query_sets["scope"])
    semantic_chunks = []
    seen = set()

    def add_chunk(chunk, target):
        text = str(chunk or "").strip()
        normalized = " ".join(text.split()).casefold()
        if normalized and normalized not in seen:
            seen.add(normalized)
            target.append(text)

    for query in queries:
        try:
            embedding = generate_embeddings([query])[0]
            for chunk in (search_documents(embedding, n_results=per_query) or []):
                add_chunk(chunk, semantic_chunks)
        except Exception:
            # Continue with other queries and lexical retrieval if one query fails.
            continue

    try:
        # The previous 120-chunk limit could silently exclude later documents.
        all_chunks = get_documents_for_sources(max_chunks=2000)
    except Exception:
        all_chunks = []

    keyword_sets = {
        "scope": ["goal", "objective", "purpose", "requirement", "deliverable", "milestone", "scope", "acceptance", "role", "timeline", "phase", "stakeholder", "constraint", "assumption"],
        "risks": ["risk", "delay", "dependency", "mitigation", "impact", "likelihood", "probability", "contingency", "overdue", "quality", "security", "defect", "approval", "unresolved", "assumption", "constraint", "uncertain", "issue", "change"],
        "blockers": ["blocker", "blocked", "impediment", "obstacle", "open question", "pending", "decision", "action", "owner", "due date", "dependency", "await", "confirm", "follow-up", "assigned", "not started", "in progress", "approval", "issue", "waiting", "unresolved"],
    }
    keywords = keyword_sets.get(agent_type, keyword_sets["scope"])
    lexical_scored = []
    for index, chunk in enumerate(all_chunks):
        text = str(chunk or "").strip()
        lower = text.casefold()
        score = sum(1 for word in keywords if word in lower)
        # Small boost for useful structured evidence such as dates, task IDs, and rows.
        if re.search(r"\b(20\d{2}[-/]\d{1,2}[-/]\d{1,2}|week\s*\d+|[A-Z]{1,5}-\d+|owner|status|due)\b", text, re.I):
            score += 2
        if score:
            lexical_scored.append((score, index, text))

    combined = []
    combined_seen = set()
    def append_unique(text):
        key = " ".join(str(text).split()).casefold()
        if key and key not in combined_seen:
            combined_seen.add(key)
            combined.append(str(text).strip())

    # Keep semantically relevant evidence, then add lexical hits from across all files.
    for chunk in semantic_chunks:
        append_unique(chunk)
    for _, _, chunk in sorted(lexical_scored, key=lambda item: (item[0], -item[1]), reverse=True):
        append_unique(chunk)

    # Bound context size for the model, without truncating the retrieval pool first.
    return combined[:40]

st.set_page_config(
    page_title="AI Project Intelligence & Risk Advisor",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(ellipse at 18% 0%, rgba(35, 83, 210, .22), transparent 36%),
            radial-gradient(ellipse at 92% 10%, rgba(0, 174, 255, .14), transparent 32%),
            linear-gradient(145deg, #070B16 0%, #0A1023 48%, #080D1B 100%);
        color: #F5F8FF;
    }
    /* Full-width canvas: no empty sidebar column or divider. */
    section[data-testid="stSidebar"] { display: none !important; }
    div[data-testid="stAppViewContainer"] > .main { margin-left: 0 !important; }
    div[data-testid="stAppViewContainer"] > .main > div { max-width: 100%; }
    .projectiq-brand {
        display: flex; align-items: center; gap: 12px;
        width: fit-content; margin: 2px 0 38px 2px; padding: 0;
    }
    .projectiq-logo {
        position: relative; display: flex; align-items: center; justify-content: center;
        width: 48px; height: 48px; border-radius: 15px;
        color: white; font-size: 27px; font-weight: 850; letter-spacing: -1px;
        background: linear-gradient(140deg, #18C8FF 0%, #087BEA 54%, #5946E8 100%);
        border: 1px solid rgba(117, 224, 255, .9);
        box-shadow: 0 0 18px rgba(0, 163, 255, .35), inset 0 1px 0 rgba(255,255,255,.28);
    }
    .projectiq-name {
        font-size: 27px; font-weight: 800; letter-spacing: -.8px;
        color: #F8FAFF; line-height: 1;
    }
    .projectiq-name span {
        background: linear-gradient(90deg, #FFFFFF 5%, #9EDFFF 60%, #8D9BFF 100%);
        -webkit-background-clip: text; background-clip: text; color: transparent;
    }
    .projectiq-tagline { color: #8295B8; font-size: 12px; margin-top: 5px; letter-spacing: 1.6px; text-transform: uppercase; }
    .main .block-container {
        max-width: 1500px;
        padding-top: 1.2rem;
        padding-bottom: 3rem;
        padding-left: 2.5rem;
        padding-right: 2.5rem;
    }
    section[data-testid="stSidebar"] {
        background-color: #0B1224;
        border-right: 1px solid #243555;
    }
    h1 { color: #F8FAFC; font-weight: 700; }
    h2 { color: #EAF2FF; }
    h3 { color: #D9E7F7; }
    p, label { color: #B8C9DE; }
    .stButton > button {
        width: 100%;
        min-height: 2.65rem;
        border-radius: 10px;
        font-weight: 650;
        background: linear-gradient(135deg, #09A8F4, #1769E0);
        color: #FFFFFF;
        border: 1px solid #18C8FF;
        box-shadow: 0 6px 18px rgba(9,168,244,.16);
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #18C8FF, #09A8F4);
        color: #FFFFFF;
        border-color: #18C8FF;
        box-shadow: 0 8px 22px rgba(24,200,255,.22);
    }
    div[data-testid="stMetric"] {
        background-color: #111C35;
        border: 1px solid #243555;
        border-radius: 12px;
        padding: 0.9rem;
    }
    div[data-testid="stMetricLabel"] { color: #7F93AD; }
    div[data-testid="stMetricValue"] {
        color: #F8FAFC;
        font-weight: 700;
    }
    div[data-baseweb="input"] > div,
    div[data-baseweb="textarea"] > div {
        background-color: #111C35;
        border-color: #30466A;
        border-radius: 9px;
    }
    input, textarea { color: #F8FAFC !important; }
    div[data-baseweb="select"] > div {
        background-color: #111C35;
        border-color: #30466A;
        border-radius: 9px;
    }
    section[data-testid="stFileUploaderDropzone"] {
        background-color: #111C35;
        border: 1px dashed #3B5D82;
        border-radius: 12px;
    }
    div[role="radiogroup"] label[data-baseweb="radio"] {
        color:#B8C9DE;
    }
    div[role="radiogroup"] label[data-baseweb="radio"]:has(input:checked) {
        background:linear-gradient(90deg,rgba(9,168,244,.18),rgba(23,105,224,.08));
        border-radius:9px;
    }
    div[data-testid="stAlert"] { border-radius: 10px; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header[data-testid="stHeader"] { background: transparent; }

    /* Hide the sidebar hamburger/collapse controls; keep the sidebar branding visible. */
    [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapseButton"],
    button[kind="header"] {
        display: none !important;
        visibility: hidden !important;
    }
    
    /* =========================================================
       PROJECTIQ NEON BLUE DASHBOARD THEME
       Inspired by the reference: black canvas, electric-blue glow,
       luminous gradients and premium glass-like panels.
       ========================================================= */
    :root {
        --pi-bg: #03050B;
        --pi-panel: rgba(8, 15, 34, .92);
        --pi-panel-2: rgba(10, 22, 53, .88);
        --pi-blue: #126BFF;
        --pi-cyan: #12CFFF;
        --pi-purple: #4A42FF;
        --pi-text: #F6F8FF;
        --pi-muted: #A2B2D0;
    }
    html, body, [data-testid="stAppViewContainer"], .stApp {
        background-color: var(--pi-bg) !important;
        color: var(--pi-text) !important;
    }
    .stApp {
        background-image:
            radial-gradient(ellipse at 18% 0%, rgba(33, 88, 255, .32) 0%, transparent 35%),
            radial-gradient(ellipse at 82% 12%, rgba(0, 177, 255, .20) 0%, transparent 31%),
            radial-gradient(ellipse at 50% 100%, rgba(39, 58, 205, .20) 0%, transparent 42%),
            linear-gradient(135deg, #03050A 0%, #070D20 48%, #020409 100%) !important;
        background-attachment: fixed !important;
    }
    .main .block-container {
        max-width: 1540px !important;
        padding-top: 1.4rem !important;
        padding-bottom: 4rem !important;
    }
    /* Bright but restrained glow under the brand and key headings. */
    .projectiq-brand { margin-bottom: 34px !important; }
    .projectiq-logo {
        background: linear-gradient(135deg, #14D9FF 0%, #087BFF 48%, #5143FF 100%) !important;
        border: 1px solid rgba(115, 226, 255, .95) !important;
        box-shadow: 0 0 22px rgba(0, 119, 255, .52), 0 0 48px rgba(57, 65, 255, .22), inset 0 1px 0 rgba(255,255,255,.38) !important;
    }
    h1, h2, h3, h4, [data-testid="stHeader"] { color: #F8FAFF !important; }
    h1 { text-shadow: 0 0 24px rgba(44, 113, 255, .18); }
    p, label, .stMarkdown, [data-testid="stCaptionContainer"] { color: #C0CEE7; }
    /* Dashboard sections/cards: deep glass panels with blue-violet edges. */
    div[data-testid="stVerticalBlockBorderWrapper"],
    div[data-testid="stExpander"],
    div[data-testid="stMetric"],
    div[data-testid="stForm"],
    div[data-testid="stDataFrame"],
    div[data-testid="stTable"] {
        background: linear-gradient(145deg, rgba(10, 21, 49, .94), rgba(4, 9, 22, .96)) !important;
        border: 1px solid rgba(67, 116, 220, .34) !important;
        border-radius: 16px !important;
        box-shadow: 0 8px 28px rgba(0, 0, 0, .22), inset 0 1px 0 rgba(122, 177, 255, .06) !important;
    }
    div[data-testid="stMetric"] {
        position: relative;
        overflow: hidden;
        padding: 1rem 1.1rem !important;
        box-shadow: 0 8px 26px rgba(0, 67, 255, .10), inset 0 1px 0 rgba(130, 184, 255, .09) !important;
    }
    div[data-testid="stMetric"]:hover,
    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        border-color: rgba(24, 176, 255, .62) !important;
        box-shadow: 0 0 24px rgba(0, 111, 255, .12), 0 10px 30px rgba(0,0,0,.24) !important;
        transition: all .22s ease;
    }
    div[data-testid="stMetricLabel"] { color: #91A9D2 !important; }
    div[data-testid="stMetricValue"] { color: #F7FAFF !important; text-shadow: 0 0 18px rgba(24, 147, 255, .15); }
    /* Controls use the same electric blue gradient throughout the app. */
    .stButton > button, button[kind="primary"], [data-testid="stDownloadButton"] button {
        background: linear-gradient(105deg, #08BDEB 0%, #087BFF 52%, #393FF0 100%) !important;
        color: #FFFFFF !important;
        border: 1px solid rgba(42, 207, 255, .92) !important;
        border-radius: 11px !important;
        box-shadow: 0 5px 20px rgba(0, 111, 255, .24), inset 0 1px 0 rgba(255,255,255,.17) !important;
        transition: transform .18s ease, box-shadow .18s ease, filter .18s ease !important;
    }
    .stButton > button:hover, button[kind="primary"]:hover,
    [data-testid="stDownloadButton"] button:hover {
        filter: brightness(1.12) !important;
        transform: translateY(-1px);
        box-shadow: 0 0 25px rgba(0, 160, 255, .34) !important;
    }
    div[data-baseweb="input"] > div,
    div[data-baseweb="textarea"] > div,
    div[data-baseweb="select"] > div {
        background: rgba(7, 15, 34, .96) !important;
        border: 1px solid rgba(74, 117, 192, .58) !important;
        border-radius: 10px !important;
        color: #F7FAFF !important;
    }
    input, textarea, [data-baseweb="select"] * { color: #F7FAFF !important; }
    div[data-baseweb="input"] > div:focus-within,
    div[data-baseweb="textarea"] > div:focus-within,
    div[data-baseweb="select"] > div:focus-within {
        border-color: #13CFFF !important;
        box-shadow: 0 0 0 1px rgba(19, 207, 255, .35), 0 0 18px rgba(0, 118, 255, .16) !important;
    }
    section[data-testid="stFileUploaderDropzone"] {
        background: linear-gradient(120deg, rgba(8, 26, 62, .95), rgba(10, 18, 43, .94)) !important;
        border: 1px dashed rgba(28, 172, 255, .68) !important;
        border-radius: 15px !important;
        box-shadow: inset 0 0 24px rgba(0, 102, 255, .08) !important;
    }
    section[data-testid="stFileUploaderDropzone"] button {
        background: linear-gradient(110deg, #09BDEB, #126BFF) !important;
        border: 1px solid #22D1FF !important;
        color: white !important;
    }
    div[role="radiogroup"] label[data-baseweb="radio"] { color: #C1D1EA !important; }
    div[role="radiogroup"] label[data-baseweb="radio"]:has(input:checked) {
        background: linear-gradient(100deg, rgba(0, 190, 255, .18), rgba(64, 65, 255, .20)) !important;
        border: 1px solid rgba(23, 190, 255, .38) !important;
        border-radius: 10px !important;
    }
    div[data-testid="stAlert"] {
        background: rgba(8, 24, 53, .92) !important;
        border: 1px solid rgba(42, 139, 255, .42) !important;
        border-radius: 12px !important;
    }
    hr { border-color: rgba(81, 125, 206, .25) !important; }
    [data-testid="stDataFrame"] *, [data-testid="stTable"] * { color: #EAF1FF; }
    /* Keep the page full width and free from Streamlit's sidebar chrome. */
    section[data-testid="stSidebar"] { display: none !important; }
    [data-testid="collapsedControl"], [data-testid="stSidebarCollapseButton"], button[kind="header"] { display: none !important; }
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# PROJECTIQ BRANDING — compact header, full-width layout
# ============================================================
st.markdown(
    """
    <div class="projectiq-brand">
        <div class="projectiq-logo">P<span style="font-size:0">.</span></div>
        <div>
            <div class="projectiq-name"><span>ProjectIQ</span></div>
            <div class="projectiq-tagline">Project intelligence workspace</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# ============================================================
# MULTI-PROJECT WORKSPACE — start with project creation only
# ============================================================
if "projects" not in st.session_state:
    # Do not automatically open the old/default workspace on startup.
    # Existing ChromaDB records remain intact; the user explicitly creates
    # or selects a workspace before seeing its dashboard.
    st.session_state.projects = {}
if "active_project_id" not in st.session_state:
    st.session_state.active_project_id = None


def _delete_project_source_records(filename, project_id):
    """Delete one uploaded source and its chunks from the active project only."""
    collection = vector_store_module.collection
    rows = collection.get(
        where={
            "$and": [
                {"project_id": str(project_id)},
                {"source": str(filename)},
            ]
        },
        include=[],
    )
    ids = rows.get("ids", [])
    if ids:
        collection.delete(ids=ids)
    return len(ids)


def _create_project(project_name):
    import uuid
    clean_name = project_name.strip()
    if not clean_name:
        st.warning("Enter a project name first.")
        return False
    existing_project = next(
        (pid for pid, name in st.session_state.projects.items()
         if name.casefold() == clean_name.casefold()),
        None,
    )
    if existing_project is not None:
        # Reopen the matching workspace instead of leaving the user on another project.
        st.session_state.active_project_id = existing_project
        st.session_state.active_page = "📊 Overview"
        st.session_state["reset_project_workspace_selector"] = True
        return True
    new_id = uuid.uuid4().hex
    st.session_state.projects[new_id] = clean_name
    st.session_state.active_project_id = new_id
    # Clear the old selector value on the next run BEFORE selectbox() is created.
    # This keeps the newly created project active and reveals its empty upload panel.
    st.session_state["reset_project_workspace_selector"] = True
    st.session_state["active_page"] = "📊 Overview"
    st.session_state.documentation = ""
    st.session_state.dashboard_agent_result = None
    st.session_state.dashboard_agent_result_type = None
    st.session_state.data_processed = False
    st.session_state.current_project_sources = []
    st.session_state.processed_source = []
    st.session_state.stored_chunks = 0
    return True


# Initialize navigation before deciding where workspace controls belong.
if "active_page" not in st.session_state:
    st.session_state.active_page = "📊 Overview"

if not st.session_state.projects or not st.session_state.active_project_id:
    with st.container(border=True):
        st.markdown("## ✨ Create your first project")
        st.write("Start by creating a workspace. You can upload its documents on the next screen.")
        first_project_name = st.text_input(
            "Project name", placeholder="e.g. AgriDoctor", key="first_project_name"
        )
        if st.button("＋ Create Project", use_container_width=True, key="first_create_project_button"):
            if _create_project(first_project_name):
                st.rerun()
    st.stop()

# Reset the workspace selector on the next rerun after project creation/switching.
# This must happen BEFORE selectbox() is instantiated; Streamlit forbids changing
# a widget's session-state value after it has been rendered in the same run.
if st.session_state.pop("reset_project_workspace_selector", False):
    st.session_state.pop("project_workspace_selector", None)

# Workspace selection and project creation are dashboard-only controls.
# Other feature pages retain the active project context without showing this panel.
if st.session_state.active_page == "📊 Overview":
    with st.container(border=True):
        pcol1, pcol2, pcol3 = st.columns([2.2, 2, 1.2])
        with pcol1:
            project_ids = list(st.session_state.projects.keys())
            current_active = st.session_state.active_project_id
            if current_active not in project_ids:
                current_active = project_ids[0]
            selected_id = st.selectbox(
                "Current project workspace",
                options=project_ids,
                index=project_ids.index(current_active),
                format_func=lambda pid: st.session_state.projects[pid],
                # Give each active workspace its own selector state so the
                # selector immediately reflects a newly created project.
                key=f"project_workspace_selector_{current_active}"
            )
        with pcol2:
            new_project_name = st.text_input(
                "Create another project", placeholder="e.g. AgriDoctor", key="new_project_name"
            )
        with pcol3:
            st.markdown("<div style='height:1.75rem'></div>", unsafe_allow_html=True)
            create_project_clicked = st.button("＋ New Project", use_container_width=True, key="create_project_button")

        if create_project_clicked:
            if _create_project(new_project_name):
                # _create_project sets the new workspace as active. Rerun immediately
                # so Overview displays the upload panel for that empty project.
                st.rerun()
        else:
            # Change workspace only when the selector, not New Project, was used.
            if selected_id != st.session_state.active_project_id:
                st.session_state.active_project_id = selected_id
                st.session_state.active_page = "📊 Overview"
                st.rerun()

# Keep the selected project's ChromaDB context synchronized on every page.
active_id = st.session_state.active_project_id
set_active_project(active_id)
current_sources = get_project_sources(active_id)
st.session_state.current_project_sources = current_sources
st.session_state.processed_source = list(current_sources)
st.session_state.data_processed = bool(current_sources)
if st.session_state.active_page == "📊 Overview":
    st.caption(
        f"Active workspace: **{st.session_state.projects[active_id]}** · "
        f"{len(current_sources)} document(s)"
    )

if "data_processed" not in st.session_state:
    st.session_state.data_processed = False

if "processed_source" not in st.session_state:
    st.session_state.processed_source = []

if "stored_chunks" not in st.session_state:
    st.session_state.stored_chunks = 0

if "current_project_sources" not in st.session_state:
    st.session_state.current_project_sources = []

if "documentation" not in st.session_state:
    st.session_state.documentation = ""


if "assistant_popup_question" not in st.session_state:
    st.session_state.assistant_popup_question = ""

if "assistant_popup_answer" not in st.session_state:
    st.session_state.assistant_popup_answer = ""

if "assistant_popup_open" not in st.session_state:
    st.session_state.assistant_popup_open = False

if "dashboard_agent_analysis_pending" not in st.session_state:
    st.session_state.dashboard_agent_analysis_pending = False

if "dashboard_agent_result" not in st.session_state:
    st.session_state.dashboard_agent_result = None

if "dashboard_agent_result_type" not in st.session_state:
    st.session_state.dashboard_agent_result_type = None


def go_to_page(target_page):
    st.session_state.active_page = target_page


page = st.session_state.active_page


def require_project_data():

    if not st.session_state.data_processed:

        st.warning(
            "⚠️ Please upload and process project documents "
            "first from the Overview dashboard."
        )

        return False

    return True


def show_project_documents():

    st.title("📄 Project Documents")
    st.caption("Upload project documents or enter project information to build the project knowledge base.")
    st.divider()
    st.subheader("📂 Upload Project Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX, CSV or TXT",
        type=["pdf", "docx", "csv", "txt"],
        accept_multiple_files=True,
        key="project_document_uploader"
    )

    st.markdown("### Or enter project information manually")

    plain_text = st.text_area(
        "Project Information",
        placeholder=(
            "Enter or paste project requirements, meeting notes, "
            "deadlines, tasks, risks, etc."
        ),
        height=180,
        key="project_plain_text"
    )

    if st.session_state.data_processed:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Documents", get_unique_document_count())
        with col2:
            st.metric("Chunks", get_document_count())

    if st.button(
        "🚀 Process Project Data",
        use_container_width=True,
        key="process_project_data"
    ):

        if not uploaded_files and not plain_text.strip():

            st.warning(
                "⚠️ Please upload a document or enter "
                "project information."
            )

            st.stop()

        try:

            print("\n")
            print("=" * 60)
            print("NEW PROJECT INGESTION STARTED")
            print("=" * 60)

            # Keep other projects' vectors intact; new uploads are scoped to the active project.

            st.session_state.documentation = ""

            st.session_state.current_project_sources = get_project_sources()
            st.session_state.processed_source = list(st.session_state.current_project_sources)
            st.session_state.stored_chunks = get_document_count()

            total_processed_chunks = 0

            processed_filenames = []

            if uploaded_files:

                print(
                    f"✓ {len(uploaded_files)} new project "
                    "documents received"
                )

                for uploaded_file in uploaded_files:

                    print("\n")
                    print("-" * 60)
                    print(
                        f"PROCESSING NEW FILE: "
                        f"{uploaded_file.name}"
                    )
                    print("-" * 60)

                    filename = uploaded_file.name

                    text = extract_text(
                        uploaded_file
                    )

                    print(
                        f"✓ File received: {filename}"
                    )

                    if not text or not text.strip():

                        print(
                            f"⚠️ No readable text found "
                            f"in {filename}"
                        )

                        continue

                    print(
                        "✓ Document text extracted successfully"
                    )

                    with st.spinner(
                        f"✂️ Creating document chunks "
                        f"for {filename}..."
                    ):

                        chunks = create_chunks(
                            text
                        )

                    if not chunks:

                        print(
                            f"⚠️ No chunks created for "
                            f"{filename}"
                        )

                        continue

                    print(
                        f"✓ {len(chunks)} "
                        "document chunks created"
                    )

                    with st.spinner(
                        f"🔢 Generating embeddings "
                        f"for {filename}..."
                    ):

                        embeddings = generate_embeddings(
                            chunks
                        )

                    print(
                        f"✓ Embeddings created for "
                        f"{len(embeddings)} chunks"
                    )

                    with st.spinner(
                        f"🗄️ Storing {filename} "
                        "in ChromaDB..."
                    ):

                        count = store_documents(
                            chunks,
                            embeddings,
                            filename,
                            project_id=st.session_state.active_project_id,
                        )

                    total_processed_chunks += count

                    processed_filenames.append(
                        filename
                    )

                    print(
                        f"✓ Successfully stored "
                        f"{count} chunks from "
                        f"{filename}"
                    )

            if plain_text.strip():

                print("\n")
                print("-" * 60)
                print(
                    "PROCESSING MANUAL PROJECT INFORMATION"
                )
                print("-" * 60)

                filename = (
                    "manual_project_information.txt"
                )

                text = plain_text

                chunks = create_chunks(
                    text
                )

                print(
                    f"✓ {len(chunks)} "
                    "document chunks created"
                )

                if chunks:

                    embeddings = generate_embeddings(
                        chunks
                    )

                    print(
                        f"✓ Embeddings created for "
                        f"{len(embeddings)} chunks"
                    )

                    count = store_documents(
                        chunks,
                        embeddings,
                        filename,
                        project_id=st.session_state.active_project_id,
                    )

                    total_processed_chunks += count

                    processed_filenames.append(
                        filename
                    )

                    print(
                        f"✓ Successfully stored "
                        f"{count} chunks from "
                        "manual project information"
                    )

            if not processed_filenames:

                st.warning(
                    "⚠️ No valid project documents "
                    "could be processed."
                )

                st.session_state.data_processed = False

                return

            st.session_state.data_processed = True

            st.session_state.processed_source = (
                processed_filenames
            )

            st.session_state.current_project_sources = (
                processed_filenames
            )

            st.session_state.stored_chunks = (
                total_processed_chunks
            )

            total_documents = (
                get_unique_document_count()
            )

            total_chunks = (
                get_document_count()
            )

            print("\n")
            print("=" * 60)
            print("CURRENT PROJECT PROCESSING SUMMARY")
            print("=" * 60)

            print(
                "✓ Previous project data removed"
            )

            print(
                "✓ Current project documents:"
            )

            for filename in processed_filenames:

                print(
                    f"   - {filename}"
                )

            print(
                f"✓ Total current project documents: "
                f"{total_documents}"
            )

            print(
                f"✓ Total current project chunks: "
                f"{total_chunks}"
            )

            print(
                "✓ ChromaDB now contains ONLY "
                "the current project"
            )

            print("=" * 60)

            st.success(
                "✅ Current project information "
                "successfully loaded."
            )

            st.caption(
                "Current project documents: "
                + ", ".join(
                    processed_filenames
                )
            )

            st.info(
                f"📚 {total_processed_chunks} "
                "chunks are available for "
                "the current project."
            )

        except Exception as e:

            print("\n")
            print("=" * 60)
            print(
                "ERROR DURING PROJECT PROCESSING"
            )
            print("=" * 60)

            print(e)

            print("=" * 60)

            st.error(
                f"❌ Processing failed: {e}"
            )


def show_project_agents():

    if not require_project_data():
        return

    if "agent_type" not in st.session_state:
        st.session_state.agent_type = (
            "Scope & Deliverables"
        )

    if "agent_analysis_pending" not in st.session_state:
        st.session_state.agent_analysis_pending = False

    if "agent_result" not in st.session_state:
        st.session_state.agent_result = None

    if "agent_result_type" not in st.session_state:
        st.session_state.agent_result_type = None

    def select_scope():

        st.session_state.agent_type = (
            "Scope & Deliverables"
        )

        st.session_state.agent_analysis_pending = True

    def select_risks():

        st.session_state.agent_type = (
            "Risks & Delivery Forecast"
        )

        st.session_state.agent_analysis_pending = True

    def select_blockers():

        st.session_state.agent_type = (
            "Blockers & Action Items"
        )

        st.session_state.agent_analysis_pending = True

    def select_from_dropdown():

        st.session_state.agent_analysis_pending = True

    st.title(
        "🤖 Project Intelligence"
    )

    st.caption(
        "Select an analysis type to retrieve and analyze "
        "relevant information from the project knowledge base."
    )

    st.divider()

    st.subheader(
        "Select Analysis Type"
    )

    st.selectbox(
        "Analysis Type",
        [
            "Scope & Deliverables",
            "Risks & Delivery Forecast",
            "Blockers & Action Items"
        ],
        key="agent_type",
        on_change=select_from_dropdown,
        label_visibility="collapsed"
    )

    st.divider()

    st.subheader(
        "🚀 Project Intelligence"
    )

    st.caption(
        "Click an analysis card to automatically run the "
        "corresponding AI agent."
    )

    st.markdown(
        """
        <style>

        div[data-testid="stHorizontalBlock"] {
            gap: 1rem;
        }

        .agent-description {
            color: #94a3b8;
            font-size: 0.95rem;
            line-height: 1.5;
            margin-top: 0.4rem;
            margin-bottom: 0.8rem;
        }

        .agent-tag {
            color: #64748b;
            font-size: 0.78rem;
            margin-bottom: 0.8rem;
        }

        </style>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(
        3,
        gap="medium"
    )

    with col1:

        with st.container(
            border=True
        ):

            st.markdown(
                "### 🎯 Scope"
            )

            st.markdown(
                """
                <div class="agent-description">
                    Analyze project goals, requirements,
                    deliverables, milestones and
                    responsibilities.
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                """
                <div class="agent-tag">
                    Goals • Requirements • Deliverables
                </div>
                """,
                unsafe_allow_html=True
            )

            st.button(
                "🎯 Analyze Scope",
                use_container_width=True,
                key="scope_analysis_button",
                on_click=select_scope
            )

    with col2:

        with st.container(
            border=True
        ):

            st.markdown(
                "### ⚠️ Risks"
            )

            st.markdown(
                """
                <div class="agent-description">
                    Identify project risks, dependencies,
                    schedule delays and delivery
                    challenges.
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                """
                <div class="agent-tag">
                    Risks • Dependencies • Forecast
                </div>
                """,
                unsafe_allow_html=True
            )

            st.button(
                "⚠️ Analyze Risks",
                use_container_width=True,
                key="risk_analysis_button",
                on_click=select_risks
            )

    with col3:

        with st.container(
            border=True
        ):

            st.markdown(
                "### 🚧 Blockers"
            )

            st.markdown(
                """
                <div class="agent-description">
                    Identify blockers, unresolved issues,
                    pending decisions and project
                    action items.
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                """
                <div class="agent-tag">
                    Blockers • Decisions • Actions
                </div>
                """,
                unsafe_allow_html=True
            )

            st.button(
                "🚧 Analyze Blockers",
                use_container_width=True,
                key="blocker_analysis_button",
                on_click=select_blockers
            )

    if st.session_state.agent_analysis_pending:

        st.session_state.agent_analysis_pending = False

        selected_agent = (
            st.session_state.agent_type
        )

        try:

            with st.spinner(
                f"🤖 Running {selected_agent}..."
            ):

                if selected_agent == (
                    "Scope & Deliverables"
                ):

                    retrieval_query = (
                        "project goal requirements "
                        "deliverables milestones "
                        "timeline responsibilities "
                        "project scope objectives "
                        "functional requirements "
                        "non functional requirements"
                    )

                elif selected_agent == (
                    "Risks & Delivery Forecast"
                ):

                    retrieval_query = (
                        "project risks schedule delays "
                        "dependencies delivery challenges "
                        "potential delays risk factors "
                        "timeline risks mitigation"
                    )

                else:

                    retrieval_query = (
                        "project blockers unresolved issues "
                        "pending decisions action items "
                        "assigned persons due dates "
                        "tasks dependencies obstacles"
                    )

                retrieval_key = (
                    "scope" if selected_agent == "Scope & Deliverables"
                    else "risks" if selected_agent == "Risks & Delivery Forecast"
                    else "blockers"
                )
                results = retrieve_agent_context(retrieval_key, per_query=4)

                if not results:

                    st.session_state.agent_result = None

                    st.session_state.agent_result_type = (
                        selected_agent
                    )

                    st.warning(
                        "⚠️ No relevant project "
                        "information was found."
                    )

                else:

                    context = (
                        "\n\n".join(
                            results
                        )
                    )

                    if selected_agent == (
                        "Scope & Deliverables"
                    ):

                        result = scope_agent(
                            context,
                            retrieval_query + "\nUse all supplied evidence. Extract concrete facts even when found in tables or meeting notes. Mark only genuinely absent details as not specified; distinguish sample/template assumptions from approved facts. Cite the document/chunk evidence in plain language, combine related facts across files, and do not treat generic section headings as the actual answer."
                        )

                    elif selected_agent == (
                        "Risks & Delivery Forecast"
                    ):

                        result = risk_agent(
                            context,
                            retrieval_query + "\nAnalyse the supplied project evidence, including meeting notes, task lists, progress updates, dependencies, due dates, open questions, and assumptions. Report explicit risks separately from inferred potential risks. For each risk provide description, source evidence, likelihood and impact only when supported (otherwise label them as estimated), consequence, mitigation or recommended next action, owner and due date only if present. Treat unanswered questions, unconfirmed integrations/approvals, dependencies, overdue or incomplete tasks, and missing readiness evidence as potential risks when relevant; do not conclude that there are no risks merely because no section is titled Risk Register. Clearly label recommendations as recommendations, and sample/template entries as SAMPLE/TEMPLATE. Never invent project facts or claim successful mitigation without evidence. If no evidence supports a risk, explain that limitation precisely."
                        )

                    else:

                        result = blocker_agent(
                            context,
                            retrieval_query + "\nAnalyze the supplied project evidence, not just explicitly labelled 'blocker' fields. Extract concrete open questions, pending decisions, action items, owners, due dates, dependencies, and status from meeting notes and task tables. A question or action can be unresolved even if no document labels it a blocker. Use IDs and names exactly as written. Since these documents are marked sample/template data, label their entries as SAMPLE/TEMPLATE rather than confirmed real-world status. Do not say 'Not specified in the provided context' for a category if any relevant evidence appears anywhere in the supplied context. If no explicit blocker is recorded, say 'No explicit blocker recorded'; still list related open decisions and actions separately. Never invent owners, dates, or statuses. Include the evidence text or document topic supporting each finding; if a category has related evidence but no explicit blocker, describe it as an open item or potential blocker rather than discarding it."
                        )

                    st.session_state.agent_result = (
                        result
                    )

                    st.session_state.agent_result_type = (
                        selected_agent
                    )

        except Exception as e:

            st.session_state.agent_result = (
                f"Agent analysis failed: {e}"
            )

            st.session_state.agent_result_type = (
                selected_agent
            )

    if st.session_state.agent_result:

        st.divider()

        result_type = (
            st.session_state.agent_result_type
        )

        if result_type == (
            "Scope & Deliverables"
        ):

            st.subheader(
                "🎯 Scope & Deliverables Analysis"
            )

            st.success(
                "✅ Scope analysis completed successfully."
            )

        elif result_type == (
            "Risks & Delivery Forecast"
        ):

            st.subheader(
                "⚠️ Risk & Delivery Forecast"
            )

            st.warning(
                "⚠️ Risk analysis completed successfully."
            )

        else:

            st.subheader(
                "🚧 Blocker & Action Item Analysis"
            )

            st.warning(
                "🚧 Blocker analysis completed successfully."
            )

        with st.container(
            border=True
        ):

            st.markdown(
                f"### {result_type}"
            )

            agent_output = str(
                st.session_state.agent_result
            )

            # Normalize HTML line-break tags produced by the LLM.
            # Handle both real tags and escaped tags such as &lt;br&gt;.
            agent_output = html.unescape(agent_output)
            agent_output = re.sub(
                r"<\s*br\s*/?\s*>",
                " • ",
                agent_output,
                flags=re.IGNORECASE
            )
            agent_output = re.sub(
                r"&lt;\s*br\s*/?\s*&gt;",
                " • ",
                agent_output,
                flags=re.IGNORECASE
            )

            if result_type == "Risks & Delivery Forecast":

                severity_colors = {
                    "High": "#EF4444",
                    "Medium": "#FACC15",
                    "Low": "#22C55E"
                }

                lines = [
                    line.strip()
                    for line in agent_output.splitlines()
                    if line.strip()
                ]

                table_lines = [
                    line for line in lines
                    if line.startswith("|") and line.endswith("|")
                ]

                if len(table_lines) >= 3:

                    headers = [
                        cell.strip()
                        for cell in table_lines[0].strip("|").split("|")
                    ]

                    html_rows = []

                    for row in table_lines[2:]:

                        cells = [
                            cell.strip()
                            for cell in row.strip("|").split("|")
                        ]

                        if len(cells) != len(headers):
                            continue

                        row_html = []

                        for index, cell in enumerate(cells):

                            cell = re.sub(
                                r"\*\*(.*?)\*\*",
                                r"<strong>\1</strong>",
                                cell
                            )

                            header_name = re.sub(
                                r"\s+",
                                " ",
                                headers[index].strip().lower()
                            )

                            if header_name == "severity":

                                severity_match = re.search(
                                    r"(?:^|\s)(high|medium|low)(?:\s|$)",
                                    re.sub(r"[*_`]", "", cell),
                                    flags=re.IGNORECASE
                                )

                                if severity_match:
                                    severity = severity_match.group(1).capitalize()
                                    color = severity_colors[severity]
                                    cell = (
                                        f'<span style="display:inline-flex;'
                                        f'align-items:center;gap:7px;'
                                        f'padding:5px 11px;border-radius:999px;'
                                        f'background:{color}22;color:{color};'
                                        f'border:1px solid {color}66;'
                                        f'font-weight:800;white-space:nowrap;">'
                                        f'● {severity}'
                                        f'</span>'
                                    )

                            # Remove any remaining HTML break tags from individual cells.
                            cell = html.unescape(cell)
                            cell = re.sub(
                                r"<\s*br\s*/?\s*>",
                                " • ",
                                cell,
                                flags=re.IGNORECASE
                            )
                            cell = re.sub(
                                r"&lt;\s*br\s*/?\s*&gt;",
                                " • ",
                                cell,
                                flags=re.IGNORECASE
                            )
                            cell = cell.replace("\n", " ")

                            row_html.append(
                                f'<td style="padding:16px 14px;'
                                f'border-bottom:1px solid #24324A;'
                                f'vertical-align:top;'
                                f'line-height:1.65;">{cell}</td>'
                            )

                        html_rows.append(
                            "<tr>" + "".join(row_html) + "</tr>"
                        )

                    header_html = "".join(
                        f'<th style="padding:14px;'
                        f'border-bottom:1px solid #334155;'
                        f'text-align:left;">{header}</th>'
                        for header in headers
                    )

                    st.markdown(
                        f"""
                        <div style="overflow-x:auto; width:100%;">
                            <table style="width:100%; border-collapse:collapse;
                                          color:#E5E7EB; font-size:0.98rem;">
                                <thead>
                                    <tr>{header_html}</tr>
                                </thead>
                                <tbody>
                                    {''.join(html_rows)}
                                </tbody>
                            </table>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                else:
                    st.markdown(
                        agent_output,
                        unsafe_allow_html=True
                    )

            else:
                st.markdown(
                    agent_output,
                    unsafe_allow_html=True
                )


def create_section_pdf(
    title,
    content,
    filename
):

    buffer = io.BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=52,
        leftMargin=52,
        topMargin=72,
        bottomMargin=58,
        title=title,
        author="AI Project Intelligence & Risk Advisor"
    )

    styles = getSampleStyleSheet()

    navy = colors.HexColor("#0B1730")
    blue = colors.HexColor("#1769E0")
    text_color = colors.HexColor("#1F2937")
    muted = colors.HexColor("#64748B")
    border = colors.HexColor("#D7E1EE")

    title_style = ParagraphStyle(
        "PDFTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=22,
        leading=27,
        textColor=colors.white,
        alignment=TA_LEFT
    )

    subtitle_style = ParagraphStyle(
        "PDFSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11,
        textColor=muted
    )

    section_heading_style = ParagraphStyle(
        "PDFSectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        textColor=navy,
        spaceBefore=10,
        spaceAfter=8,
        keepWithNext=True
    )

    item_heading_style = ParagraphStyle(
        "PDFItemHeading",
        parent=styles["Heading3"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=15,
        textColor=blue,
        spaceBefore=9,
        spaceAfter=4,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        "PDFBody",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=10.2,
        leading=15,
        textColor=text_color,
        spaceAfter=7,
        alignment=TA_LEFT
    )

    bullet_style = ParagraphStyle(
        "PDFBullet",
        parent=body_style,
        leftIndent=14,
        firstLineIndent=-8,
        spaceAfter=5
    )

    story = []

    title_table = Table(
        [
            [
                Paragraph(
                    title,
                    title_style
                )
            ]
        ],
        colWidths=[A4[0] - 104]
    )

    title_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, -1),
                    navy
                ),
                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    18
                ),
                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    18
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    14
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    14
                )
            ]
        )
    )

    story.append(title_table)

    story.append(
        Spacer(
            1,
            8
        )
    )

    story.append(
        Paragraph(
            "AI Project Intelligence & Risk Advisor",
            subtitle_style
        )
    )

    story.append(
        Spacer(
            1,
            6
        )
    )

    story.append(
        Table(
            [[""]],
            colWidths=[A4[0] - 104],
            rowHeights=[2],
            style=TableStyle(
                [
                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, -1),
                        blue
                    )
                ]
            )
        )
    )

    story.append(
        Spacer(
            1,
            14
        )
    )

    def clean_for_pdf(value):
        value = html.unescape(str(value))

        value = re.sub(
            r"<\s*br\s*/?\s*>",
            " • ",
            value,
            flags=re.IGNORECASE
        )

        value = re.sub(
            r"&lt;\s*br\s*/?\s*&gt;",
            " • ",
            value,
            flags=re.IGNORECASE
        )

        value = value.replace("\u2018", "'")
        value = value.replace("\u2019", "'")
        value = value.replace("\u201c", '"')
        value = value.replace("\u201d", '"')
        value = value.replace("\u2013", "-")
        value = value.replace("\u2014", "-")
        value = value.replace("\u2212", "-")
        value = value.replace("\u00a0", " ")

        return value.strip()

    def markdown_to_paragraph(value):
        from xml.sax.saxutils import escape

        value = clean_for_pdf(value)
        value = escape(value)

        value = re.sub(
            r"\*\*(.+?)\*\*",
            r"<b>\1</b>",
            value
        )

        value = re.sub(
            r"__(.+?)__",
            r"<b>\1</b>",
            value
        )

        value = re.sub(
            r"(?<!\*)\*([^*\n]+)\*(?!\*)",
            r"<i>\1</i>",
            value
        )

        value = value.replace(
            "`",
            ""
        )

        value = re.sub(
            r"\s+",
            " ",
            value
        )

        return value

    lines = content.splitlines()

    for line in lines:

        line = line.strip()

        if not line:
            story.append(
                Spacer(
                    1,
                    5
                )
            )
            continue

        clean_line = clean_for_pdf(line)

        if clean_line in {
            "USER STORIES",
            "RISK REGISTER",
            "ACTION ITEMS"
        }:
            story.append(
                Paragraph(
                    clean_line.title(),
                    section_heading_style
                )
            )
            continue

        if re.match(
            r"^\**\s*(User Story\s+\d+|Risk\s+\d+|Action Item\s+\d+)\s*:?\s*\**$",
            clean_line,
            flags=re.IGNORECASE
        ):
            story.append(
                Paragraph(
                    markdown_to_paragraph(clean_line),
                    item_heading_style
                )
            )
            continue

        if clean_line.startswith(
            (
                "Risk:",
                "Reason:",
                "Severity:",
                "Impact:",
                "Delivery Impact:",
                "Suggested Mitigation:",
                "Suggested Action:",
                "Action Item:",
                "Assigned To:",
                "Due Date:",
                "Status:"
            )
        ):
            story.append(
                Paragraph(
                    markdown_to_paragraph(clean_line),
                    item_heading_style
                )
            )
            continue

        if clean_line.endswith(":") and len(clean_line) < 100:
            story.append(
                Paragraph(
                    markdown_to_paragraph(clean_line),
                    item_heading_style
                )
            )
            continue

        if clean_line.startswith(("-", "•", "*")):
            bullet_text = clean_line.lstrip("-•*").strip()
            story.append(
                Paragraph(
                    "• " + markdown_to_paragraph(bullet_text),
                    bullet_style
                )
            )
            continue

        story.append(
            Paragraph(
                markdown_to_paragraph(clean_line),
                body_style
            )
        )

    def add_page_decoration(canvas, doc):
        canvas.saveState()

        width, height = A4

        canvas.setStrokeColor(border)
        canvas.setLineWidth(0.6)
        canvas.line(
            52,
            42,
            width - 52,
            42
        )

        canvas.setFont(
            "Helvetica",
            7.5
        )
        canvas.setFillColor(muted)

        canvas.drawString(
            52,
            28,
            "AI Project Intelligence & Risk Advisor"
        )

        canvas.drawRightString(
            width - 52,
            28,
            f"Page {doc.page}"
        )

        canvas.restoreState()

    document.build(
        story,
        onFirstPage=add_page_decoration,
        onLaterPages=add_page_decoration
    )

    buffer.seek(0)

    return buffer.getvalue()


def split_documentation_sections(
    documentation
):

    documentation = str(
        documentation
    )

    documentation = (
        documentation
        .replace(
            "<br>",
            "\n"
        )
        .replace(
            "<br/>",
            "\n"
        )
        .replace(
            "<br />",
            "\n"
        )
    )

    user_story_match = re.search(
        r"(?:1\.\s*)?USER STORIES(.*?)(?=(?:2\.\s*)?RISK REGISTER)",
        documentation,
        re.IGNORECASE | re.DOTALL
    )

    risk_match = re.search(
        r"(?:2\.\s*)?RISK REGISTER(.*?)(?=(?:3\.\s*)?ACTION ITEMS)",
        documentation,
        re.IGNORECASE | re.DOTALL
    )

    action_match = re.search(
        r"(?:3\.\s*)?ACTION ITEMS(.*)",
        documentation,
        re.IGNORECASE | re.DOTALL
    )

    user_stories = (
        user_story_match.group(1).strip()
        if user_story_match
        else "No user stories were generated."
    )

    risk_register = (
        risk_match.group(1).strip()
        if risk_match
        else "No risk register was generated."
    )

    action_items = (
        action_match.group(1).strip()
        if action_match
        else "No action items were generated."
    )

    return (
        user_stories,
        risk_register,
        action_items
    )

# ============================================================
# DOCUMENTATION GENERATION
# ============================================================

def render_page_close_button():

    st.markdown(
        """
        <style>
        div.st-key-close_page button {
            width: 42px !important;
            height: 42px !important;
            min-height: 42px !important;
            padding: 0 !important;
            border-radius: 50% !important;
            background: #DC2626 !important;
            border: 1px solid #EF4444 !important;
            color: #FFFFFF !important;
            font-size: 1.15rem !important;
            box-shadow: 0 5px 16px rgba(220,38,38,.28) !important;
        }
        div.st-key-close_page {
            margin-bottom: 0.4rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    if st.button(
        "✕",
        key="close_page",
        help="Back to dashboard"
    ):
        st.session_state.active_page = "📊 Overview"
        st.rerun()


def render_floating_assistant():

    st.markdown(
        """
        <style>
        div.st-key-assistant_floating_button {
            position: fixed;
            right: 28px;
            bottom: 28px;
            z-index: 99999;
        }
        div.st-key-assistant_floating_button button {
            width: 58px !important;
            height: 58px !important;
            min-height: 58px !important;
            padding: 0 !important;
            border-radius: 50% !important;
            background: linear-gradient(135deg,#09A8F4,#1769E0) !important;
            border: 2px solid #18C8FF !important;
            color: #FFFFFF !important;
            font-size: 1.45rem !important;
            box-shadow: 0 8px 28px rgba(9,168,244,.38) !important;
        }
        div.st-key-assistant_floating_button button:hover {
            background: linear-gradient(135deg,#18C8FF,#1769E0) !important;
            transform: translateY(-2px);
        }
        div.st-key-assistant_popup {
            position: fixed;
            right: 28px;
            bottom: 90px;
            width: 680px;
            max-width: calc(100vw - 36px);
            height: min(680px, calc(100vh - 160px));
            max-height: calc(100vh - 160px);
            min-height: 420px;
            z-index: 99998;
            background: #0D1524;
            border: 1px solid #30466A;
            border-radius: 18px;
            padding: 1.25rem;
            box-shadow: 0 18px 50px rgba(0,0,0,.55);
            overflow-y: auto !important;
            overflow-x: hidden !important;
            overscroll-behavior: contain;
            scrollbar-width: thin;
            scrollbar-color: #238BEA #172238;
        }
        div.st-key-assistant_popup::-webkit-scrollbar {
            width: 8px;
        }
        div.st-key-assistant_popup::-webkit-scrollbar-track {
            background: #172238;
            border-radius: 10px;
        }
        div.st-key-assistant_popup::-webkit-scrollbar-thumb {
            background: #238BEA;
            border-radius: 10px;
        }
        div.st-key-assistant_popup::-webkit-scrollbar-thumb:hover {
            background: #18C8FF;
        }
        div.st-key-assistant_close button {
            width: 30px !important;
            height: 30px !important;
            min-height: 30px !important;
            padding: 0 !important;
            border-radius: 50% !important;
            background: #DC2626 !important;
            border: 1px solid #EF4444 !important;
            color: #FFFFFF !important;
            font-size: .9rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    if st.session_state.assistant_popup_open:
        with st.container(key="assistant_popup", border=True):
            close_col, title_col = st.columns([0.12, 0.88])
            with close_col:
                if st.button("✕", key="assistant_close", help="Close assistant"):
                    st.session_state.assistant_popup_open = False
                    st.rerun()
            with title_col:
                st.markdown("### 💬 Project Assistant")
                st.caption("Ask about the current project documents.")

            question = st.text_input(
                "Question",
                placeholder="What are the major project risks?",
                key="assistant_popup_question_input",
                label_visibility="collapsed"
            )

            if st.button(
                "Ask Assistant",
                use_container_width=True,
                key="assistant_floating_popup_ask"
            ):
                if not question.strip():
                    st.warning("Please enter a question.")
                else:
                    try:
                        with st.spinner("Searching project knowledge..."):
                            question_embedding = generate_embeddings([question])[0]
                            results = search_documents(question_embedding, n_results=5)

                        if not results:
                            st.warning("No relevant information was found.")
                        else:
                            context = "\n\n".join(results)
                            with st.spinner("Generating answer..."):
                                answer = project_assistant(context, question)
                            st.markdown("### Assistant")
                            st.write(answer)
                    except Exception as e:
                        st.error(f"Assistant failed: {e}")

    if st.button(
        "💬",
        key="assistant_floating_button",
        help="Open Project Assistant"
    ):
        st.session_state.assistant_popup_open = (
            not st.session_state.assistant_popup_open
        )
        st.rerun()


def show_documentation_generation():

    if not require_project_data():
        return

    render_page_close_button()

    st.title("📝 Documentation Generation")
    st.caption(
        "Generate structured project documentation from "
        "the currently uploaded project documents."
    )

    st.divider()

    st.info(
        "The documentation agent will generate "
        "User Stories, Risk Register, and Action Items "
        "using only the currently loaded project documents."
    )

    if st.session_state.current_project_sources:

        st.caption(
            "📄 Current project documents: "
            + ", ".join(
                st.session_state.current_project_sources
            )
        )

    if st.button(
        "📄 Generate Project Documentation",
        use_container_width=True,
        key="generate_documentation"
    ):

        try:

            with st.spinner(
                "🔎 Retrieving current project information..."
            ):

                documentation_query = (
                    "project requirements "
                    "project objectives "
                    "project scope "
                    "functional requirements "
                    "non functional requirements "
                    "features "
                    "users "
                    "roles "
                    "stakeholders "
                    "deliverables "
                    "milestones "
                    "tasks "
                    "responsibilities "
                    "risks "
                    "risk register "
                    "blockers "
                    "unresolved issues "
                    "action items "
                    "pending work "
                    "deadlines "
                    "due dates "
                    "status "
                    "progress updates "
                    "decisions"
                )

                documentation_embedding = (
                    generate_embeddings(
                        [documentation_query]
                    )[0]
                )

                documentation_results = (
                    search_documents(
                        documentation_embedding,
                        n_results=15
                    )
                )

            if not documentation_results:

                st.warning(
                    "⚠️ No information was found "
                    "for the current project."
                )

                return

            documentation_context = (
                "\n\n".join(
                    documentation_results
                )
            )

            if not documentation_context.strip():

                st.warning(
                    "⚠️ The current project documents "
                    "did not contain usable information."
                )

                return

            current_sources = (
                st.session_state.current_project_sources
            )

            source_information = (
                "\n".join(
                    f"- {source}"
                    for source in current_sources
                )
            )

            final_context = f"""
CURRENT PROJECT DOCUMENTS:

{source_information}

IMPORTANT:
The following information was retrieved from
the CURRENTLY LOADED PROJECT.

Do not use information from any previous project.

CURRENT PROJECT INFORMATION:

{documentation_context}
"""

            with st.spinner(
                "🤖 Generating documentation "
                "from the current project..."
            ):

                documentation = (
                    documentation_agent(
                        final_context
                    )
                )

            if documentation is None:

                st.error(
                    "❌ Documentation generation failed: "
                    "The documentation agent returned "
                    "no response."
                )

                return

            documentation = str(
                documentation
            ).strip()

            if not documentation:

                st.error(
                    "❌ Documentation generation failed: "
                    "The LLM returned an empty response."
                )

                return

            st.session_state.documentation = (
                documentation
            )

            st.success(
                "✅ Project documentation "
                "generated successfully."
            )

        except Exception as e:

            st.error(
                f"❌ Documentation generation failed: {e}"
            )

    if st.session_state.documentation:

        st.divider()

        st.subheader(
            "📋 Generated Project Documentation"
        )

        with st.container(
            border=True
        ):

            st.markdown(
                st.session_state.documentation
            )

        documentation_text = (
            st.session_state.documentation
        )

        st.divider()

        st.subheader(
            "⬇️ Download Documentation"
        )

        st.caption(
            "Download each generated documentation "
            "component as a separate PDF file."
        )

        (
            user_stories,
            risk_register,
            action_items
        ) = split_documentation_sections(
            documentation_text
        )

        user_stories_pdf = create_section_pdf(
            "User Stories",
            user_stories,
            "user_stories.pdf"
        )

        risk_register_pdf = create_section_pdf(
            "Risk Register",
            risk_register,
            "risk_register.pdf"
        )

        action_items_pdf = create_section_pdf(
            "Action Items",
            action_items,
            "action_items.pdf"
        )

        col1, col2, col3 = st.columns(
            3
        )

        with col1:

            with st.container(
                border=True
            ):

                st.markdown(
                    "### 📄 User Stories"
                )

                st.caption(
                    "Generated user stories based "
                    "on project requirements."
                )

                st.download_button(
                    label="⬇️ Download PDF",
                    data=user_stories_pdf,
                    file_name="user_stories.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    key="download_user_stories_pdf"
                )

        with col2:

            with st.container(
                border=True
            ):

                st.markdown(
                    "### ⚠️ Risk Register"
                )

                st.caption(
                    "Risks, severity, impact "
                    "and suggested mitigation."
                )

                st.download_button(
                    label="⬇️ Download PDF",
                    data=risk_register_pdf,
                    file_name="risk_register.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    key="download_risk_register_pdf"
                )

        with col3:

            with st.container(
                border=True
            ):

                st.markdown(
                    "### ✅ Action Items"
                )

                st.caption(
                    "Tasks, responsibilities, "
                    "due dates and status."
                )

                st.download_button(
                    label="⬇️ Download PDF",
                    data=action_items_pdf,
                    file_name="action_items.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    key="download_action_items_pdf"
                )
                
# ============================================================
# PROJECT HEALTH
# ============================================================

def show_project_health():

    if not require_project_data():
        return

    render_page_close_button()

    st.title(
        "❤️ Project Health Dashboard"
    )

    st.caption(
        "AI-powered assessment of project scope, "
        "timeline risk and blockers."
    )

    st.divider()

    should_analyze = st.session_state.pop(
        "health_auto_analyze",
        False
    )

    if should_analyze:

        try:

            with st.spinner(
                "🔎 Retrieving project health information..."
            ):

                scope_query = (
                    "project scope project goals requirements "
                    "functional requirements non functional "
                    "requirements deliverables milestones "
                    "responsibilities"
                )

                timeline_query = (
                    "project timeline milestones deadlines "
                    "schedule delayed tasks pending tasks "
                    "dependencies delivery dates"
                )

                blocker_query = (
                    "project blockers unresolved issues "
                    "obstacles pending decisions problems "
                    "dependencies risks preventing progress"
                )

                scope_embedding = (
                    generate_embeddings(
                        [scope_query]
                    )[0]
                )

                timeline_embedding = (
                    generate_embeddings(
                        [timeline_query]
                    )[0]
                )

                blocker_embedding = (
                    generate_embeddings(
                        [blocker_query]
                    )[0]
                )

                scope_results = search_documents(
                    scope_embedding,
                    n_results=5
                )

                timeline_results = search_documents(
                    timeline_embedding,
                    n_results=5
                )

                blocker_results = search_documents(
                    blocker_embedding,
                    n_results=5
                )


            all_results = (
                scope_results
                + timeline_results
                + blocker_results
            )


            unique_results = []

            for result in all_results:

                if result not in unique_results:

                    unique_results.append(
                        result
                    )


            health_context = (
                "\n\n".join(
                    unique_results
                )
            )


            if not health_context.strip():

                st.warning(
                    "⚠️ No relevant project information "
                    "was found for health analysis."
                )

                return


            with st.spinner(
                "🤖 AI is analyzing project health..."
            ):

                health_analysis = (
                    analyze_project_health(
                        health_context
                    )
                )


            health_result = (
                calculate_health_score(
                    health_analysis[
                        "scope_clarity"
                    ],
                    health_analysis[
                        "timeline_risk"
                    ],
                    health_analysis[
                        "blocker_count"
                    ]
                )
            )


            health_status = (
                get_health_status(
                    health_result[
                        "overall_score"
                    ]
                )
            )


            overall_score = (
                health_result[
                    "overall_score"
                ]
            )

            scope_score = (
                health_result[
                    "scope_clarity"
                ]
            )

            timeline_score = (
                health_result[
                    "timeline_score"
                ]
            )

            blocker_score = (
                health_result[
                    "blocker_score"
                ]
            )

            blocker_count = (
                health_result[
                    "blocker_count"
                ]
            )


            if health_status == "Healthy":

                risk_label = "SAFE"

                risk_message = (
                    "The project is currently "
                    "in a healthy state based on "
                    "the available project information."
                )

            elif health_status == "Moderate":

                risk_label = "AT RISK"

                risk_message = (
                    "The project requires attention. "
                    "Some areas may affect delivery "
                    "if they are not addressed."
                )

            elif health_status == "At Risk":

                risk_label = "AT RISK"

                risk_message = (
                    "The project has significant "
                    "risk indicators that require "
                    "attention."
                )

            else:

                risk_label = "CRITICAL"

                risk_message = (
                    "The project shows serious "
                    "risk indicators and requires "
                    "immediate attention."
                )


            st.divider()


            # ====================================================
            # OVERALL HEALTH
            # ====================================================

            st.subheader(
                "🎯 Overall Project Health"
            )


            col1, col2 = st.columns(
                [1, 2]
            )


            with col1:

                st.metric(
                    "Health Score",
                    f"{overall_score}/100"
                )


            with col2:

                if risk_label == "SAFE":

                    st.success(
                        f"🟢 {risk_label}"
                    )

                elif risk_label == "AT RISK":

                    st.warning(
                        f"🟡 {risk_label}"
                    )

                else:

                    st.error(
                        f"🔴 {risk_label}"
                    )


                st.write(
                    risk_message
                )


            st.progress(
                min(
                    max(
                        overall_score / 100,
                        0.0
                    ),
                    1.0
                )
            )


            st.caption(
                f"Current health level: {health_status}"
            )


            st.divider()


            # ====================================================
            # HEALTH DIMENSIONS
            # ====================================================

            st.subheader(
                "📊 Health Dimension Analysis"
            )


            col1, col2, col3 = st.columns(3)


            with col1:

                st.metric(
                    "🎯 Scope Clarity",
                    f"{scope_score}/100"
                )

                st.progress(
                    scope_score / 100
                )


            with col2:

                st.metric(
                    "⏱️ Timeline Score",
                    f"{timeline_score}/100"
                )

                st.progress(
                    timeline_score / 100
                )


            with col3:

                st.metric(
                    "🚧 Blocker Score",
                    f"{blocker_score}/100"
                )

                st.progress(
                    blocker_score / 100
                )


            st.divider()


            # ====================================================
            # GRAPHICAL REPRESENTATION
            # ====================================================

            st.subheader(
                "📈 Project Health Visualization"
            )

            graph_col1, graph_col2 = st.columns(
                [1, 1.25],
                gap="large"
            )

            with graph_col1:

                if go is not None:

                    if overall_score < 40:
                        gauge_color = "#EF4444"
                    elif overall_score < 60:
                        gauge_color = "#F97316"
                    elif overall_score < 80:
                        gauge_color = "#FACC15"
                    else:
                        gauge_color = "#22C55E"

                    fig = go.Figure(
                        go.Indicator(
                            mode="gauge+number",
                            value=overall_score,
                            number={
                                "suffix": "/100",
                                "font": {
                                    "size": 38,
                                    "color": "#F8FAFC"
                                }
                            },
                            title={
                                "text": f"Overall Health — {risk_label}",
                                "font": {
                                    "size": 14,
                                    "color": "#B8C9DE"
                                }
                            },
                            gauge={
                                "axis": {
                                    "range": [0, 100],
                                    "tickcolor": "#7F93AD",
                                    "tickfont": {
                                        "color": "#7F93AD"
                                    }
                                },
                                "bar": {
                                    "color": gauge_color,
                                    "thickness": 0.55
                                },
                                "bgcolor": "#0B1224",
                                "borderwidth": 0,
                                "steps": [
                                    {
                                        "range": [0, 40],
                                        "color": "#4A1820"
                                    },
                                    {
                                        "range": [40, 60],
                                        "color": "#4A2A16"
                                    },
                                    {
                                        "range": [60, 80],
                                        "color": "#4A4214"
                                    },
                                    {
                                        "range": [80, 100],
                                        "color": "#164A2A"
                                    }
                                ]
                            }
                        )
                    )

                    fig.update_layout(
                        height=350,
                        margin={
                            "l": 20,
                            "r": 20,
                            "t": 70,
                            "b": 20
                        },
                        paper_bgcolor="#080D1C",
                        font={
                            "color": "#F8FAFC"
                        }
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                        config={
                            "displayModeBar": False
                        }
                    )

                else:

                    st.metric(
                        "Overall Health",
                        f"{overall_score}/100"
                    )

                    st.progress(
                        overall_score / 100
                    )

            with graph_col2:

                st.markdown("### Health Indicators")

                indicator_data = [
                    (
                        "Scope Clarity",
                        scope_score
                    ),
                    (
                        "Timeline Health",
                        timeline_score
                    ),
                    (
                        "Blocker Health",
                        blocker_score
                    )
                ]

                for label, score in indicator_data:

                    st.markdown(
                        f"**{label}**"
                    )

                    st.progress(
                        min(
                            max(
                                score / 100,
                                0.0
                            ),
                            1.0
                        )
                    )

                    st.caption(
                        f"{score}/100"
                    )

            st.caption(
                "Higher scores indicate stronger project health. "
                "The overall score combines scope clarity, timeline health and blocker health."
            )

            st.divider()


            # ====================================================
            # PROJECT RISK SUMMARY
            # ====================================================

            st.subheader(
                "⚠️ Project Risk Summary"
            )


            col1, col2 = st.columns(2)


            with col1:

                st.markdown(
                    "### Current Status"
                )

                if risk_label == "SAFE":

                    st.success(
                        "🟢 Project appears to be in "
                        "a safe state."
                    )

                elif risk_label == "AT RISK":

                    st.warning(
                        "🟡 Project requires attention."
                    )

                else:

                    st.error(
                        "🔴 Project requires immediate attention."
                    )


            with col2:

                st.markdown(
                    "### Blocker Overview"
                )

                if blocker_count == 0:

                    st.success(
                        "✅ No blockers identified."
                    )

                elif blocker_count <= 2:

                    st.warning(
                        f"⚠️ {blocker_count} blocker(s) identified."
                    )

                else:

                    st.error(
                        f"🚧 {blocker_count} blockers identified."
                    )


            # ====================================================
            # PROJECT HEALTH RISK MATRIX
            # ====================================================

            st.subheader(
                "📌 Project Health Risk Matrix"
            )

            st.caption(
                "Risks are positioned using the health analysis "
                "for scope, timeline and blockers."
            )

            def classify_risk_level(score):
                if score < 60:
                    return "High"
                elif score < 80:
                    return "Medium"
                return "Low"

            def classify_blocker_level(count):
                if count > 2:
                    return "High"
                elif count > 0:
                    return "Medium"
                return "Low"

            risk_matrix_data = [
                {
                    "Risk": "Scope Risk",
                    "Probability": classify_risk_level(
                        scope_score
                    ),
                    "Impact": (
                        "High"
                        if scope_score < 60
                        else (
                            "Medium"
                            if scope_score < 80
                            else "Low"
                        )
                    ),
                    "Reason": health_analysis.get(
                        "scope_reason",
                        "Scope clarity assessment."
                    )
                },
                {
                    "Risk": "Timeline Risk",
                    "Probability": classify_risk_level(
                        100 - health_result.get(
                            "timeline_risk",
                            100 - timeline_score
                        )
                    ),
                    "Impact": (
                        "High"
                        if timeline_score < 60
                        else (
                            "Medium"
                            if timeline_score < 80
                            else "Low"
                        )
                    ),
                    "Reason": health_analysis.get(
                        "timeline_reason",
                        "Timeline health assessment."
                    )
                },
                {
                    "Risk": "Blocker Risk",
                    "Probability": classify_blocker_level(
                        blocker_count
                    ),
                    "Impact": (
                        "High"
                        if blocker_count > 2
                        else (
                            "Medium"
                            if blocker_count > 0
                            else "Low"
                        )
                    ),
                    "Reason": health_analysis.get(
                        "blocker_reason",
                        "Blocker assessment."
                    )
                }
            ]

            probability_order = [
                "Low",
                "Medium",
                "High"
            ]

            impact_order = [
                "Low",
                "Medium",
                "High"
            ]

            risk_colors = {
                "Low": "#22C55E",
                "Medium": "#FACC15",
                "High": "#F97316",
                "Critical": "#EF4444"
            }

            for item in risk_matrix_data:

                if (
                    item["Probability"] == "High"
                    and item["Impact"] == "High"
                ):
                    item["Severity"] = "Critical"

                elif (
                    item["Probability"] == "High"
                    or item["Impact"] == "High"
                ):
                    item["Severity"] = "High"

                elif (
                    item["Probability"] == "Medium"
                    and item["Impact"] == "Medium"
                ):
                    item["Severity"] = "Medium"

                else:
                    item["Severity"] = "Low"

            if go is not None:

                fig = go.Figure()

                for item in risk_matrix_data:

                    x_value = impact_order.index(
                        item["Impact"]
                    )

                    y_value = probability_order.index(
                        item["Probability"]
                    )

                    fig.add_trace(
                        go.Scatter(
                            x=[x_value],
                            y=[y_value],
                            mode="markers+text",
                            text=[item["Risk"]],
                            textposition="top center",
                            marker={
                                "size": 30,
                                "color": risk_colors[
                                    item["Severity"]
                                ],
                                "line": {
                                    "width": 2,
                                    "color": "#F8FAFC"
                                }
                            },
                            name=item["Risk"],
                            hovertemplate=(
                                "<b>%{text}</b><br>"
                                "Probability: "
                                + item["Probability"]
                                + "<br>"
                                "Impact: "
                                + item["Impact"]
                                + "<br>"
                                "Severity: "
                                + item["Severity"]
                                + "<extra></extra>"
                            ),
                            showlegend=False
                        )
                    )

                fig.update_xaxes(
                    tickmode="array",
                    tickvals=[0, 1, 2],
                    ticktext=[
                        "Low",
                        "Medium",
                        "High"
                    ],
                    title="Impact",
                    range=[-0.5, 2.5],
                    gridcolor="#26344F",
                    zeroline=False
                )

                fig.update_yaxes(
                    tickmode="array",
                    tickvals=[0, 1, 2],
                    ticktext=[
                        "Low",
                        "Medium",
                        "High"
                    ],
                    title="Probability",
                    range=[-0.5, 2.5],
                    gridcolor="#26344F",
                    zeroline=False
                )

                fig.update_layout(
                    height=430,
                    margin={
                        "l": 70,
                        "r": 30,
                        "t": 35,
                        "b": 70
                    },
                    paper_bgcolor="#080D1C",
                    plot_bgcolor="#0B1224",
                    font={
                        "color": "#EAF2FF"
                    },
                    xaxis={
                        "title_font": {
                            "size": 15
                        },
                        "tickfont": {
                            "size": 13
                        }
                    },
                    yaxis={
                        "title_font": {
                            "size": 15
                        },
                        "tickfont": {
                            "size": 13
                        }
                    }
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    config={
                        "displayModeBar": False
                    }
                )

            else:

                st.warning(
                    "Plotly is not installed. "
                    "Install it with: pip install plotly"
                )

            st.markdown(
                """
                <div style="
                    display:flex;
                    gap:18px;
                    flex-wrap:wrap;
                    margin:10px 0 20px 0;
                    padding:12px 16px;
                    border:1px solid #26344F;
                    border-radius:10px;
                    background:#0B1224;
                ">
                    <span style="color:#22C55E;">
                        🟢 Low
                    </span>
                    <span style="color:#FACC15;">
                        🟡 Medium
                    </span>
                    <span style="color:#F97316;">
                        🟠 High
                    </span>
                    <span style="color:#EF4444;">
                        🔴 Critical
                    </span>
                </div>
                """,
                unsafe_allow_html=True
            )

            risk_table = pd.DataFrame(
                [
                    {
                        "Risk": item["Risk"],
                        "Probability": item["Probability"],
                        "Impact": item["Impact"],
                        "Severity": item["Severity"]
                    }
                    for item in risk_matrix_data
                ]
            )

            st.dataframe(
                risk_table,
                use_container_width=True,
                hide_index=True
            )

            with st.expander(
                "🔎 View Risk Matrix Evidence"
            ):

                for item in risk_matrix_data:

                    st.markdown(
                        f"### {item['Risk']}"
                    )

                    st.write(
                        str(item["Reason"])
                        .replace("<br>", " ")
                        .replace("<br/>", " ")
                        .replace("<br />", " ")
                    )

                    st.caption(
                        f"Probability: {item['Probability']}  •  "
                        f"Impact: {item['Impact']}  •  "
                        f"Severity: {item['Severity']}"
                    )

                    st.divider()


            st.divider()


            # ====================================================
            # AI ANALYSIS
            # ====================================================


            st.subheader(
                "🤖 AI Health Analysis"
            )


            with st.container(
                border=True
            ):

                st.markdown(
                    "### 🎯 Scope Clarity"
                )

                st.write(
                    health_analysis[
                        "scope_reason"
                    ]
                )


            with st.container(
                border=True
            ):

                st.markdown(
                    "### ⏱️ Timeline Risk"
                )

                st.write(
                    health_analysis[
                        "timeline_reason"
                    ]
                )


            with st.container(
                border=True
            ):

                st.markdown(
                    "### 🚧 Blocker Analysis"
                )

                st.write(
                    health_analysis[
                        "blocker_reason"
                    ]
                )


            st.divider()


            # ====================================================
            # SCORE CALCULATION
            # ====================================================

            with st.expander(
                "📐 View Health Score Calculation"
            ):

                st.write(
                    f"Scope Score: {scope_score}"
                )

                st.write(
                    f"Timeline Score: {timeline_score}"
                )

                st.write(
                    f"Blocker Score: {blocker_score}"
                )

                st.markdown(
                    f"""
                    **Overall Score**

                    ({scope_score} + {timeline_score} + {blocker_score}) / 3

                    **Final Health Score: {overall_score}/100**
                    """
                )


            st.divider()


            # ====================================================
            # RECOMMENDED ATTENTION AREAS
            # ====================================================

            st.subheader(
                "🎯 Attention Areas"
            )


            attention_areas = []


            if scope_score < 60:

                attention_areas.append(
                    "🎯 Scope clarity needs attention."
                )

            elif scope_score < 80:

                attention_areas.append(
                    "🎯 Scope clarity could be improved."
                )


            if timeline_score < 60:

                attention_areas.append(
                    "⏱️ Timeline risk requires attention."
                )

            elif timeline_score < 80:

                attention_areas.append(
                    "⏱️ Timeline should be monitored closely."
                )


            if blocker_count > 2:

                attention_areas.append(
                    "🚧 Multiple blockers require resolution."
                )

            elif blocker_count > 0:

                attention_areas.append(
                    "🚧 Existing blockers should be addressed."
                )


            if not attention_areas:

                st.success(
                    "✅ No major attention areas were "
                    "identified by the health analysis."
                )

            else:

                for area in attention_areas:

                    if "Scope" in area:
                        border_color = "#FACC15"
                        bg_color = "rgba(250, 204, 21, 0.14)"
                    elif "Timeline" in area:
                        border_color = "#FACC15"
                        bg_color = "rgba(250, 204, 21, 0.14)"
                    else:
                        border_color = "#EF4444"
                        bg_color = "rgba(239, 68, 68, 0.14)"

                    st.markdown(
                        f"""
                        <div style="
                            background:{bg_color};
                            border-left:5px solid {border_color};
                            border-radius:10px;
                            padding:18px 20px;
                            margin-bottom:12px;
                            color:#dbeafe;
                            font-size:1.05rem;
                        ">
                            {area}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )


        except Exception as e:

            st.error(
                f"❌ Project health analysis failed: {e}"
            )

def show_project_assistant():

    if not require_project_data():
        return

    st.title("💬 Project Intelligence Assistant")
    st.caption("Ask questions about your project documents. The assistant retrieves relevant information from ChromaDB and generates a grounded answer.")
    st.divider()

    assistant_question = st.text_input(
        "Ask your project question",
        placeholder="Example: What are the current project risks?",
        key="assistant_question"
    )

    if st.button("🤖 Ask Project Assistant", use_container_width=True, key="ask_project_assistant"):
        if not assistant_question.strip():
            st.warning("⚠️ Please enter a question.")
        else:
            try:
                with st.spinner("🔎 Searching project knowledge base..."):
                    question_embedding = generate_embeddings([assistant_question])[0]
                    results = search_documents(question_embedding, n_results=3)

                if not results:
                    st.warning("⚠️ No relevant information was found in the project knowledge base.")
                else:
                    context = "\n\n".join(results)
                    with st.spinner("🤖 Generating assistant answer..."):
                        answer = project_assistant(context, assistant_question)
                    st.success("✅ Answer generated from project documents.")
                    st.subheader("💡 Assistant Answer")
                    with st.container(border=True):
                        st.write(answer)
            except Exception as e:
                st.error(f"❌ Assistant failed: {e}")



def add_more_project_documents(uploaded_files):

    if not uploaded_files:
        st.warning("Please select at least one document.")
        return False

    try:

        existing_sources = set(
            st.session_state.current_project_sources
        )

        added_files = []
        duplicate_files = []
        total_chunks = 0

        for uploaded_file in uploaded_files:

            filename = uploaded_file.name

            if filename in existing_sources:
                duplicate_files.append(filename)
                continue

            with st.spinner(
                f"Processing {filename}..."
            ):

                text = extract_text(uploaded_file)

                if not text or not text.strip():
                    st.warning(
                        f"No readable text was found in {filename}."
                    )
                    continue

                chunks = create_chunks(text)

                if not chunks:
                    st.warning(
                        f"No usable content was found in {filename}."
                    )
                    continue

                embeddings = generate_embeddings(chunks)

                count = store_documents(
                    chunks,
                    embeddings,
                    filename,
                    project_id=st.session_state.active_project_id,
                )

            added_files.append(filename)
            total_chunks += count

        if added_files:

            st.session_state.current_project_sources.extend(
                added_files
            )

            st.session_state.processed_source.extend(
                added_files
            )

            st.session_state.stored_chunks += total_chunks

            print("\n" + "=" * 60)
            print("ADDITIONAL PROJECT DOCUMENTS")
            print("=" * 60)

            for filename in added_files:
                print(f"Added: {filename}")

            print(
                f"Total project documents: "
                f"{get_unique_document_count()}"
            )

            print(
                f"Total project chunks: "
                f"{get_document_count()}"
            )

            print("=" * 60)

            if duplicate_files:
                st.info(
                    "Already present: "
                    + ", ".join(duplicate_files)
                )

            return True

        if duplicate_files:
            st.info(
                "The selected documents are already part "
                "of the current project."
            )
        else:
            st.error(
                "No additional project documents were added."
            )

        return False

    except Exception as e:

        st.error(
            f"Additional document processing failed: {e}"
        )

        return False


def process_dashboard_documents(uploaded_files):

    if not uploaded_files:
        st.warning(
            "Please upload at least one project document."
        )
        return False

    try:

        # Do not clear ChromaDB: vector_store scopes all operations to the active project.

        st.session_state.documentation = ""
        st.session_state.current_project_sources = get_project_sources()
        st.session_state.processed_source = list(st.session_state.current_project_sources)
        st.session_state.stored_chunks = get_document_count()
        st.session_state.agent_result = None
        st.session_state.agent_result_type = None
        st.session_state.agent_analysis_pending = False

        total_processed_chunks = 0
        processed_filenames = []

        for uploaded_file in uploaded_files:

            filename = uploaded_file.name

            with st.spinner(
                f"Processing {filename}..."
            ):

                text = extract_text(
                    uploaded_file
                )

            if not text or not text.strip():

                st.warning(
                    f"No readable text was found in {filename}."
                )

                continue

            with st.spinner(
                f"Creating knowledge chunks for {filename}..."
            ):

                chunks = create_chunks(text)

            if not chunks:

                st.warning(
                    f"No usable chunks were created for {filename}."
                )

                continue

            with st.spinner(
                f"Generating embeddings for {filename}..."
            ):

                embeddings = generate_embeddings(
                    chunks
                )

            with st.spinner(
                f"Storing {filename} in ChromaDB..."
            ):

                count = store_documents(
                    chunks,
                    embeddings,
                    filename,
                    project_id=st.session_state.active_project_id,
                )

            total_processed_chunks += count
            processed_filenames.append(filename)

        if not processed_filenames:

            st.error(
                "No valid project documents could be processed."
            )

            return False

        st.session_state.data_processed = True

        st.session_state.current_project_sources = get_project_sources()
        st.session_state.processed_source = list(st.session_state.current_project_sources)

        st.session_state.stored_chunks = (
            total_processed_chunks
        )

        print("\n" + "=" * 60)
        print("CURRENT PROJECT PROCESSING SUMMARY")
        print("=" * 60)

        for filename in processed_filenames:

            print(
                f"- {filename}"
            )

        print(
            f"Total current project documents: "
            f"{get_unique_document_count()}"
        )

        print(
            f"Total current project chunks: "
            f"{get_document_count()}"
        )

        print(
            "ChromaDB contains ONLY the current project"
        )

        print("=" * 60)

        return True

    except Exception as e:

        st.error(
            f"Project document processing failed: {str(e)}"
        )

        return False

def dashboard_navigate(target_page):

    st.session_state.active_page = target_page

    if target_page == "❤️ Project Health":
        st.session_state.health_auto_analyze = True


def dashboard_agent_navigate(agent_type):

    st.session_state.dashboard_agent_analysis_pending = True
    st.session_state.dashboard_agent_result = None
    st.session_state.dashboard_agent_result_type = agent_type


def run_dashboard_agent_analysis():

    agent_type = st.session_state.dashboard_agent_result_type

    if not agent_type:
        return

    st.session_state.dashboard_agent_analysis_pending = False

    try:

        if agent_type == "scope":

            retrieval_query = (
                "project goals requirements deliverables "
                "milestones timeline responsibilities "
                "project scope objectives functional "
                "requirements non functional requirements"
            )

        elif agent_type == "risks":

            retrieval_query = (
                "project risks schedule delays dependencies "
                "delivery challenges potential delays risk "
                "factors timeline risks mitigation"
            )

        else:

            retrieval_query = (
                "project blockers unresolved issues pending "
                "decisions action items assigned persons "
                "due dates tasks dependencies obstacles"
            )

        with st.spinner(
            f"🤖 Analyzing {agent_type} from the current project..."
        ):

            results = retrieve_agent_context(agent_type, per_query=4)

            if not results:
                st.session_state.dashboard_agent_result = (
                    "No relevant project information was found."
                )
                return

            context = "\n\n".join(results)

            if agent_type == "scope":
                result = scope_agent(
                    context,
                    retrieval_query + "\nUse all supplied evidence. Extract concrete facts even when found in tables or meeting notes. Mark only genuinely absent details as not specified; distinguish sample/template assumptions from approved facts. Cite the document/chunk evidence in plain language, combine related facts across files, and do not treat generic section headings as the actual answer."
                )
            elif agent_type == "risks":
                result = risk_agent(
                    context,
                    retrieval_query + "\nAnalyse the supplied project evidence, including meeting notes, task lists, progress updates, dependencies, due dates, open questions, and assumptions. Report explicit risks separately from inferred potential risks. For each risk provide description, source evidence, likelihood and impact only when supported (otherwise label them as estimated), consequence, mitigation or recommended next action, owner and due date only if present. Treat unanswered questions, unconfirmed integrations/approvals, dependencies, overdue or incomplete tasks, and missing readiness evidence as potential risks when relevant; do not conclude that there are no risks merely because no section is titled Risk Register. Clearly label recommendations as recommendations, and sample/template entries as SAMPLE/TEMPLATE. Never invent project facts or claim successful mitigation without evidence. If no evidence supports a risk, explain that limitation precisely."
                )
            else:
                result = blocker_agent(
                    context,
                    retrieval_query + "\nAnalyze the supplied project evidence, not just explicitly labelled 'blocker' fields. Extract concrete open questions, pending decisions, action items, owners, due dates, dependencies, and status from meeting notes and task tables. A question or action can be unresolved even if no document labels it a blocker. Use IDs and names exactly as written. Since these documents are marked sample/template data, label their entries as SAMPLE/TEMPLATE rather than confirmed real-world status. Do not say 'Not specified in the provided context' for a category if any relevant evidence appears anywhere in the supplied context. If no explicit blocker is recorded, say 'No explicit blocker recorded'; still list related open decisions and actions separately. Never invent owners, dates, or statuses. Include the evidence text or document topic supporting each finding; if a category has related evidence but no explicit blocker, describe it as an open item or potential blocker rather than discarding it."
                )

            st.session_state.dashboard_agent_result = str(
                result
            )

    except Exception as e:

        st.session_state.dashboard_agent_result = (
            f"Agent analysis failed: {e}"
        )


def close_dashboard_agent_result():

    st.session_state.dashboard_agent_result = None
    st.session_state.dashboard_agent_result_type = None


# ============================================================
# PAGE NAVIGATION
# ============================================================

def show_overview():

    st.markdown(
        """
        <style>

        div.st-key-close_dashboard_agent_result button {
            width: 34px !important;
            height: 34px !important;
            min-height: 34px !important;
            padding: 0 !important;
            border-radius: 50% !important;
            background: #DC2626 !important;
            border: 1px solid #EF4444 !important;
            color: #FFFFFF !important;
            font-size: 0.9rem !important;
            font-weight: 700 !important;
            box-shadow: 0 4px 14px rgba(220, 38, 38, 0.28) !important;
        }

        div.st-key-close_dashboard_agent_result button:hover {
            background: #EF4444 !important;
        }

        .stApp {
            background:
                radial-gradient(
                    circle at 15% 10%,
                    rgba(55, 51, 143, 0.18),
                    transparent 30%
                ),
                radial-gradient(
                    circle at 85% 5%,
                    rgba(9, 168, 244, 0.10),
                    transparent 28%
                ),
                #080D1C;
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: linear-gradient(145deg, rgba(19, 34, 67, .92), rgba(8, 16, 34, .96));
            border: 1px solid rgba(62, 112, 178, .55);
            border-radius: 18px;
            box-shadow: 0 14px 42px rgba(0, 0, 0, .20), inset 0 1px 0 rgba(255,255,255,.025);
        }

        div[data-testid="stVerticalBlockBorderWrapper"]:hover {
            border-color: #35618F;
        }

        div.stButton > button {
            background: linear-gradient(
                135deg,
                #09A8F4,
                #1769E0
            );
            color: #FFFFFF;
            border: 1px solid #18C8FF;
            border-radius: 10px;
            min-height: 44px;
            font-weight: 600;
        }

        div.stButton > button:hover {
            background: linear-gradient(
                135deg,
                #18C8FF,
                #1769E0
            );
            border-color: #18C8FF;
            color: #FFFFFF;
            box-shadow:
                0 0 18px
                rgba(9, 168, 244, 0.25);
        }

        div[data-testid="stMetric"] {
            background: linear-gradient(
                145deg,
                #162344,
                #0E172D
            );
            border: 1px solid #243A61;
            border-radius: 14px;
            padding: 0.8rem;
        }

        div[data-testid="stMetricLabel"] {
            color: #7FA6D5 !important;
        }

        div[data-testid="stMetricValue"] {
            color: #F8FAFC !important;
        }

        section[data-testid="stFileUploaderDropzone"] {
            background: #111C35;
            border: 1px dashed #3B5D82;
            border-radius: 12px;
        }

        section[data-testid="stFileUploaderDropzone"] button {
            background: linear-gradient(
                135deg,
                #09A8F4,
                #1769E0
            ) !important;
            border: 1px solid #18C8FF !important;
            color: #FFFFFF !important;
        }

        div[data-testid="stPopover"] > button {
            width: 58px !important;
            height: 58px !important;
            min-height: 58px !important;
            border-radius: 50% !important;
            background: linear-gradient(
                135deg,
                #09A8F4,
                #1769E0
            ) !important;
            border: 2px solid #18C8FF !important;
            color: #FFFFFF !important;
            font-size: 1.3rem !important;
            box-shadow:
                0 8px 28px
                rgba(9, 168, 244, 0.30);
        }

        </style>
        """,
        unsafe_allow_html=True
    )


    with st.container(border=True):

        st.markdown(
            "## 📂 Upload Project Documents"
        )

        st.write(
            "Upload your project proposal, SRS, meeting "
            "notes, task lists or progress documents "
            "to activate project intelligence."
        )

        st.caption(
            "Supported formats: PDF • DOCX • CSV • TXT"
        )

        uploaded_files = st.file_uploader(
            "Choose project documents",
            type=[
                "pdf",
                "docx",
                "csv",
                "txt"
            ],
            accept_multiple_files=True,
            # Give each project its own uploader state. A newly created project
            # therefore starts with an empty file selection instead of inheriting
            # the previous project's selected files after Streamlit reruns.
            key=f"dashboard_document_uploader_{active_id}"
        )

        if uploaded_files:

            st.info(
                f"{len(uploaded_files)} document(s) selected."
            )

            for uploaded_file in uploaded_files:

                st.write(
                    f"📄 {uploaded_file.name}"
                )

        if st.button(
            "🚀 Process Project Documents",
            use_container_width=True,
            key="dashboard_process_documents"
        ):

            if not uploaded_files:

                st.warning(
                    "Please upload at least one project document."
                )

            elif process_dashboard_documents(
                uploaded_files
            ):

                st.rerun()

    document_count = (
        get_unique_document_count()
    )

    current_sources = (
        st.session_state.current_project_sources
    )

    st.success(
        "Current project loaded — AI analysis is ready."
    )

    st.subheader(
        "Project Overview"
    )

    col1, col2, col3, col4 = st.columns(
        4,
        gap="medium"
    )

    with col1:

        st.metric(
            "📄 Documents",
            document_count,
            "Current project files"
        )

    with col2:

        st.metric(
            "🤖 AI Agents",
            "3",
            "Scope • Risk • Blocker"
        )

    with col3:

        st.metric(
            "🧠 Knowledge Base",
            "READY",
            "ChromaDB retrieval"
        )

    with col4:

        st.metric(
            "● Project Status",
            "ACTIVE",
            "Ready for analysis"
        )

    st.subheader(
        "🤖 AI Project Intelligence"
    )

    st.caption(
        "Three specialized agents analyze the current "
        "project documents."
    )

    agent_columns = st.columns(
        3,
        gap="large"
    )

    agents = [
        (
            agent_columns[0],
            "🎯",
            "Scope Agent",
            (
                "Analyzes goals, requirements, "
                "deliverables, milestones and "
                "responsibilities."
            ),
            "Goals • Requirements • Deliverables",
            "scope"
        ),
        (
            agent_columns[1],
            "⚠️",
            "Risk Agent",
            (
                "Identifies risks, dependencies, "
                "schedule concerns and delivery "
                "challenges."
            ),
            "Risks • Dependencies • Forecast",
            "risks"
        ),
        (
            agent_columns[2],
            "🚧",
            "Blocker Agent",
            (
                "Finds unresolved issues, pending "
                "decisions, obstacles and action items."
            ),
            "Blockers • Decisions • Actions",
            "blockers"
        )
    ]

    for (
        column,
        icon,
        title,
        description,
        tags,
        agent_type
    ) in agents:

        with column:

            with st.container(border=True):

                st.markdown(
                    f"## {icon}"
                )

                st.markdown(
                    f"### {title}"
                )

                st.write(
                    description
                )

                st.caption(
                    tags
                )

                st.info(
                    "● READY FROM CURRENT PROJECT"
                )

            st.button(
                f"Analyze {title.replace(' Agent', '')}",
                use_container_width=True,
                key=f"dashboard_agent_{agent_type}",
                on_click=dashboard_agent_navigate,
                args=(agent_type,)
            )

    if st.session_state.dashboard_agent_analysis_pending:
        run_dashboard_agent_analysis()

    if st.session_state.dashboard_agent_result:

        result_type = (
            st.session_state.dashboard_agent_result_type
        )

        result_titles = {
            "scope": "🎯 Scope Analysis",
            "risks": "⚠️ Risk & Delivery Forecast",
            "blockers": "🚧 Blocker & Action Item Analysis"
        }

        result_messages = {
            "scope": "Scope analysis completed successfully.",
            "risks": "Risk analysis completed successfully.",
            "blockers": "Blocker analysis completed successfully."
        }

        st.divider()

        result_header, close_column = st.columns(
            [0.94, 0.06],
            vertical_alignment="center"
        )

        with result_header:
            st.subheader(
                result_titles.get(
                    result_type,
                    "🤖 Project Analysis"
                )
            )

        with close_column:
            if st.button(
                "✕",
                key="close_dashboard_agent_result",
                help="Close analysis"
            ):
                close_dashboard_agent_result()
                st.rerun()

        st.success(
            result_messages.get(
                result_type,
                "Analysis completed successfully."
            )
        )

        with st.container(border=True):
            st.write(
                st.session_state.dashboard_agent_result
            )

    st.subheader(
        "Project Controls"
    )

    col1, col2 = st.columns(
        2,
        gap="large"
    )

    with col1:

        with st.container(border=True):

            st.markdown(
                "### 📝 Documentation Generation"
            )

            st.write(
                "Generate User Stories, Risk Register "
                "and Action Items from the current "
                "project knowledge base."
            )

            st.caption(
                "User Stories • Risks • Actions"
            )

        st.button(
            "Open Documentation Generation",
            use_container_width=True,
            key="dashboard_documentation",
            on_click=dashboard_navigate,
            args=("📝 Documentation Generation",)
        )

    with col2:

        with st.container(border=True):

            st.markdown(
                "### 📊 Project Health Score"
            )

            st.write(
                "Calculate the health score and view "
                "the gauge, scope clarity, timeline "
                "health and blocker load."
            )

            st.caption(
                "Health Score • Gauge • Risk Indicators"
            )

        st.button(
            "Calculate Project Health",
            use_container_width=True,
            key="dashboard_health",
            on_click=dashboard_navigate,
            args=("❤️ Project Health",)
        )

    st.subheader(
        "📎 Add More Project Documents"
    )

    st.caption(
        "Add additional documents later without "
        "removing the documents already stored "
        "for this project."
    )

    additional_files = st.file_uploader(
        "Choose additional project documents",
        type=[
            "pdf",
            "docx",
            "csv",
            "txt"
        ],
        accept_multiple_files=True,
        key="additional_project_documents"
    )

    if additional_files:

        st.info(
            f"{len(additional_files)} additional "
            "document(s) selected."
        )

        for uploaded_file in additional_files:

            st.write(
                f"📄 {uploaded_file.name}"
            )

        if st.button(
            "➕ Add Documents to Current Project",
            use_container_width=True,
            key="add_more_project_documents"
        ):

            try:

                existing_sources = set(
                    st.session_state.current_project_sources
                )

                added_files = []
                skipped_files = []

                with st.spinner(
                    "Adding documents to the current project..."
                ):

                    for uploaded_file in additional_files:

                        filename = uploaded_file.name

                        if filename in existing_sources:

                            skipped_files.append(
                                filename
                            )

                            continue

                        text = extract_text(
                            uploaded_file
                        )

                        if not text or not text.strip():

                            skipped_files.append(
                                filename
                            )

                            continue

                        chunks = create_chunks(
                            text
                        )

                        if not chunks:

                            skipped_files.append(
                                filename
                            )

                            continue

                        embeddings = generate_embeddings(
                            chunks
                        )

                        store_documents(
                            chunks,
                            embeddings,
                            filename,
                            project_id=st.session_state.active_project_id,
                        )

                        added_files.append(
                            filename
                        )

                        existing_sources.add(
                            filename
                        )

                        st.session_state.current_project_sources.append(
                            filename
                        )

                st.session_state.processed_source = (
                    st.session_state.current_project_sources
                )

                st.session_state.stored_chunks = (
                    get_document_count()
                )

                if added_files:

                    st.success(
                        f"{len(added_files)} document(s) "
                        "added successfully."
                    )

                if skipped_files:

                    st.warning(
                        f"{len(skipped_files)} document(s) "
                        "were skipped because they were "
                        "empty or already existed."
                    )

                print()
                print("=" * 60)
                print("ADDITIONAL DOCUMENTS")
                print("=" * 60)

                for filename in added_files:

                    print(
                        f"Added: {filename}"
                    )

                print(
                    "Total Documents:",
                    get_unique_document_count()
                )

                print(
                    "Total Chunks:",
                    get_document_count()
                )

                print("=" * 60)

                st.rerun()

            except Exception as e:

                st.error(
                    f"Additional document processing failed: {e}"
                )

    st.subheader(
        "📚 Current Project Documents"
    )

    if current_sources:

        for source in current_sources:

            with st.container(border=True):

                file_col, delete_col = st.columns([0.92, 0.08], vertical_alignment="center")

                with file_col:
                    st.write(f"📄 {source}")

                with delete_col:
                    if st.button(
                        "✕",
                        key=f"delete_project_doc_{st.session_state.active_project_id}_{source}",
                        help=f"Remove {source} from this project",
                    ):
                        try:
                            removed_chunks = _delete_project_source_records(
                                source,
                                project_id=st.session_state.active_project_id,
                            )
                            st.session_state.current_project_sources = get_project_sources(
                                st.session_state.active_project_id
                            )
                            st.session_state.processed_source = list(
                                st.session_state.current_project_sources
                            )
                            st.session_state.stored_chunks = get_document_count(
                                st.session_state.active_project_id
                            )
                            st.session_state.data_processed = bool(
                                st.session_state.current_project_sources
                            )
                            st.success(
                                f"Removed {source} and {removed_chunks} stored chunk(s) from this project."
                            )
                            st.rerun()
                        except Exception as e:
                            st.error(f"Could not remove {source}: {e}")

    else:

        st.info(
            "No project documents are currently loaded."
        )

    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )



# ============================================================
# PAGE NAVIGATION
# ============================================================

page = st.session_state.get(
    "active_page",
    "📊 Overview"
)

if page == "📊 Overview":
    show_overview()

elif page == "🤖 Project Agents":
    show_project_agents()

elif page == "❤️ Project Health":
    show_project_health()

elif page == "📝 Documentation Generation":
    show_documentation_generation()

elif page == "💬 Project Assistant":
    show_project_assistant()

render_floating_assistant()
