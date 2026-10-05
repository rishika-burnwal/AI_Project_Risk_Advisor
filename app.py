
import re
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
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
from rag.vector_store import (
    clear_collection,
    store_documents,
    search_documents,
    get_documents_for_sources,
    get_document_count,
    get_unique_document_count
)


st.set_page_config(
    page_title="AI Project Intelligence & Risk Advisor",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    .stApp { background-color: #0f172a; }
    .main .block-container { max-width: 1400px; padding-top: 2rem; padding-bottom: 3rem; padding-left: 3rem; padding-right: 3rem; }
    section[data-testid="stSidebar"] { background-color: #111827; border-right: 1px solid #263244; }
    h1 { color: #f8fafc; font-weight: 700; }
    h2 { color: #f1f5f9; }
    h3 { color: #e2e8f0; }
    p { color: #cbd5e1; }
    .stButton > button { width: 100%; border-radius: 10px; font-weight: 600; min-height: 2.7rem; }
    div[data-testid="stMetric"] { background-color: #172033; border: 1px solid #263244; border-radius: 12px; padding: 1rem; }
    div[data-testid="stMetricLabel"] { color: #94a3b8; }
    div[data-testid="stMetricValue"] { color: #f8fafc; font-weight: 700; }
    div[data-baseweb="input"] > div, div[data-baseweb="textarea"] > div { background-color: #172033; border-color: #334155; border-radius: 10px; }
    input, textarea { color: #f8fafc !important; }
    div[data-baseweb="select"] > div { background-color: #172033; border-color: #334155; border-radius: 10px; }
    section[data-testid="stFileUploaderDropzone"] { background-color: #172033; border: 1px dashed #475569; border-radius: 12px; }
    div[data-testid="stAlert"] { border-radius: 10px; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header[data-testid="stHeader"] { background: transparent; }
    </style>
    """,
    unsafe_allow_html=True
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


st.sidebar.title("🤖 Project Advisor")
st.sidebar.caption("AI Project Intelligence")
st.sidebar.divider()
st.sidebar.markdown("### MAIN")

page = st.sidebar.radio(
    "Navigation",
    [
        "📊 Overview",
        "📄 Project Documents",
        "🤖 Project Agents",
        "❤️ Project Health",
        "📝 Documentation Generation",
        "💬 Project Assistant"
    ],
    key="navigation_page",
    label_visibility="collapsed"
)

st.sidebar.divider()
st.sidebar.markdown("### PROJECT STATUS")

if st.session_state.data_processed:
    st.sidebar.success("🟢 Project Active")
    st.sidebar.caption("Knowledge Base Ready")
    st.sidebar.metric("Documents", get_unique_document_count())
    st.sidebar.metric("Chunks", get_document_count())
else:
    st.sidebar.warning("🟡 No Project Loaded")
    st.sidebar.caption("Upload project documents to begin analysis.")

st.sidebar.divider()
st.sidebar.caption("RAG • ChromaDB • AI Agents • LLM")


def require_project_data():

    if not st.session_state.data_processed:

        st.warning(
            "⚠️ Please upload and process project documents "
            "first from the Project Documents section."
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

            with st.spinner(
                "🧹 Clearing previous project data..."
            ):

                clear_collection()

            print(
                "✓ Previous project data cleared from ChromaDB"
            )

            st.session_state.documentation = ""

            st.session_state.current_project_sources = []

            st.session_state.processed_source = []

            st.session_state.stored_chunks = 0

            st.session_state.data_processed = False

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
                            filename
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
                        filename
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

                question_embedding = (
                    generate_embeddings(
                        [retrieval_query]
                    )[0]
                )

                results = search_documents(
                    question_embedding,
                    n_results=5
                )

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
                            retrieval_query
                        )

                    elif selected_agent == (
                        "Risks & Delivery Forecast"
                    ):

                        result = risk_agent(
                            context,
                            retrieval_query
                        )

                    else:

                        result = blocker_agent(
                            context,
                            retrieval_query
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

            st.write(
                st.session_state.agent_result
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
        rightMargin=45,
        leftMargin=45,
        topMargin=45,
        bottomMargin=45
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "PDFTitle",
        parent=styles["Title"],
        fontSize=20,
        leading=25,
        alignment=TA_CENTER,
        spaceAfter=20
    )

    heading_style = ParagraphStyle(
        "PDFHeading",
        parent=styles["Heading2"],
        fontSize=13,
        leading=17,
        spaceBefore=12,
        spaceAfter=7
    )

    body_style = ParagraphStyle(
        "PDFBody",
        parent=styles["BodyText"],
        fontSize=10.5,
        leading=15,
        spaceAfter=7
    )

    story = []

    story.append(
        Paragraph(
            title,
            title_style
        )
    )

    story.append(
        Spacer(
            1,
            10
        )
    )

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

        clean_line = (
            line
            .replace(
                "<br>",
                ""
            )
            .replace(
                "<br/>",
                ""
            )
            .replace(
                "<br />",
                ""
            )
        )

        if clean_line.endswith(":"):

            story.append(
                Paragraph(
                    clean_line,
                    heading_style
                )
            )

        elif clean_line.startswith(
            (
                "User Story",
                "Risk:",
                "Reason:",
                "Severity:",
                "Impact:",
                "Suggested Mitigation:",
                "Action Item:",
                "Assigned To:",
                "Due Date:",
                "Status:"
            )
        ):

            story.append(
                Paragraph(
                    clean_line,
                    heading_style
                )
            )

        else:

            clean_line = (
                clean_line
                .replace(
                    "&",
                    "&amp;"
                )
            )

            story.append(
                Paragraph(
                    clean_line,
                    body_style
                )
            )

    document.build(
        story
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

def show_documentation_generation():

    if not require_project_data():
        return

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

    st.title(
        "❤️ Project Health Dashboard"
    )

    st.caption(
        "AI-powered assessment of project scope, "
        "timeline risk and blockers."
    )

    st.divider()

    if st.button(
        "🔍 Analyze Project Health",
        use_container_width=True,
        key="analyze_project_health"
    ):

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


            chart_data = pd.DataFrame(
                {
                    "Score": [
                        scope_score,
                        timeline_score,
                        blocker_score
                    ]
                },
                index=[
                    "Scope Clarity",
                    "Timeline",
                    "Blockers"
                ]
            )


            st.bar_chart(
                chart_data,
                y="Score",
                height=350
            )


            st.caption(
                "Higher scores indicate stronger project "
                "health for the corresponding dimension."
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

                    st.warning(
                        area
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


# ============================================================
# PAGE NAVIGATION
# ============================================================

def show_overview():

    st.title(
        "🤖 AI Project Intelligence & Risk Advisor"
    )

    st.caption(
        "AI-powered project intelligence, "
        "risk analysis and decision support."
    )

    st.divider()

    if not st.session_state.data_processed:

        st.info(
            "📄 Upload and process project documents "
            "to start analyzing your project."
        )

        st.subheader(
            "🚀 Get Started"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            with st.container(border=True):

                st.markdown(
                    "### 📂 1. Upload Documents"
                )

                st.write(
                    "Upload PDF, DOCX, CSV or TXT "
                    "project documents."
                )

        with col2:

            with st.container(border=True):

                st.markdown(
                    "### 🧠 2. Build Knowledge Base"
                )

                st.write(
                    "Extract text, create chunks, "
                    "generate embeddings and store "
                    "them in ChromaDB."
                )

        with col3:

            with st.container(border=True):

                st.markdown(
                    "### 🤖 3. Analyze Project"
                )

                st.write(
                    "Use AI agents, health analysis, "
                    "documentation and the assistant."
                )

        return


    document_count = (
        get_unique_document_count()
    )


    st.success(
        "🟢 Current project is ready for analysis."
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "📄 Documents",
            document_count
        )


    with col2:

        st.metric(
            "🟢 Project Status",
            "Active"
        )


    with col3:

        st.metric(
            "🤖 AI Modules",
            "4"
        )


    st.divider()


    st.subheader(
        "🚀 Project Intelligence Modules"
    )


    col1, col2 = st.columns(2)


    with col1:

        with st.container(border=True):

            st.markdown(
                "### 🤖 Project Agents"
            )

            st.write(
                "Analyze scope, risks, delivery "
                "challenges, blockers and action items."
            )

            st.caption(
                "Scope • Risks • Blockers • Actions"
            )


    with col2:

        with st.container(border=True):

            st.markdown(
                "### ❤️ Project Health"
            )

            st.write(
                "Evaluate scope clarity, timeline "
                "risk and project blockers."
            )

            st.caption(
                "Scope • Timeline • Blockers"
            )


    col1, col2 = st.columns(2)


    with col1:

        with st.container(border=True):

            st.markdown(
                "### 📝 Documentation"
            )

            st.write(
                "Generate user stories, risk register "
                "and project action items."
            )

            st.caption(
                "User Stories • Risks • Actions"
            )


    with col2:

        with st.container(border=True):

            st.markdown(
                "### 💬 Project Assistant"
            )

            st.write(
                "Ask questions and receive answers "
                "grounded in the project knowledge base."
            )

            st.caption(
                "RAG • ChromaDB • LLM"
            )


    st.divider()


    st.subheader(
        "📚 Current Project Documents"
    )


    current_sources = (
        st.session_state.current_project_sources
    )


    if current_sources:

        for source in current_sources:

            st.markdown(
                f"📄 **{source}**"
            )

    else:

        st.caption(
            "No project documents are currently loaded."
        )


# ============================================================
# PAGE NAVIGATION
# ============================================================

if page == "📊 Overview":

    show_overview()


elif page == "📄 Project Documents":

    show_project_documents()


elif page == "🤖 Project Agents":

    show_project_agents()


elif page == "❤️ Project Health":

    show_project_health()


elif page == "📝 Documentation Generation":

    show_documentation_generation()


elif page == "💬 Project Assistant":

    show_project_assistant()
