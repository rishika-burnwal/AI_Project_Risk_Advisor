import streamlit as st

from agents.scope_agent import scope_agent
from agents.risk_agent import risk_agent
from agents.blocker_agent import blocker_agent

from ingestion.loader import extract_text
from rag.chunking import create_chunks
from rag.embeddings import generate_embeddings
from rag.vector_store import (
    store_documents,
    search_documents,
    get_document_count,
    get_unique_document_count
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Project Intelligence & Risk Advisor",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# SESSION STATE
# ============================================================

if "data_processed" not in st.session_state:
    st.session_state.data_processed = False

if "processed_source" not in st.session_state:
    st.session_state.processed_source = ""

if "stored_chunks" not in st.session_state:
    st.session_state.stored_chunks = 0


# ============================================================
# HEADER
# ============================================================

st.title("📊 AI Project Intelligence & Risk Advisor")

st.write(
    "Upload project documents or enter project information "
    "to build the project knowledge base."
)

st.divider()


# ============================================================
# DOCUMENT INPUT
# ============================================================

st.subheader("📄 Project Document")

uploaded_file = st.file_uploader(
    "Upload PDF, DOCX, CSV or TXT",
    type=["pdf", "docx", "csv", "txt"],
    accept_multiple_files=True,
    key="project_document_uploader"
)


# ============================================================
# PLAIN TEXT INPUT
# ============================================================

st.markdown("**OR enter project information manually**")

plain_text = st.text_area(
    "Project Information",
    placeholder=(
        "Enter or paste project requirements, "
        "meeting notes, deadlines, tasks, risks, etc."
    ),
    height=180,
    key="project_plain_text"
)


# ============================================================
# PROCESS PROJECT DATA
# ============================================================

if st.button(
    "🚀 Process Project Data",
    use_container_width=True,
    key="process_project_data"
):

    # --------------------------------------------------------
    # CHECK INPUT
    # --------------------------------------------------------

    if not uploaded_file and not plain_text.strip():

        st.warning(
            "⚠️ Please upload a document or enter project information."
        )

        st.stop()


    try:

        # ====================================================
        # 1. DOCUMENT INGESTION
        # ====================================================

        print("\n")
        print("=" * 60)
        print("DOCUMENT PROCESSING STARTED")
        print("=" * 60)


        if uploaded_file is not None:

            text = extract_text(uploaded_file)

            filename = uploaded_file.name

            source = f"Uploaded: {filename}"

            print(f"✓ File received: {filename}")

        else:

            text = plain_text

            filename = "manual_project_information.txt"

            source = "Manual Text Input"

            print("✓ Manual project information received")


        # ----------------------------------------------------
        # CHECK EXTRACTED TEXT
        # ----------------------------------------------------

        if not text or not text.strip():

            st.error(
                "❌ No readable text was found in the provided input."
            )

            print("✗ No readable text found")

            st.stop()


        print("✓ Document text extracted successfully")


        # ====================================================
        # 2. CHUNKING
        # ====================================================

        with st.spinner(
            "✂️ Creating document chunks..."
        ):

            chunks = create_chunks(text)


        print(
            f"✓ {len(chunks)} document chunks created"
        )


        # ====================================================
        # 3. EMBEDDINGS
        # ====================================================

        with st.spinner(
            "🔢 Generating embeddings..."
        ):

            embeddings = generate_embeddings(chunks)


        print(
            f"✓ Embeddings created for {len(embeddings)} chunks"
        )


        # ====================================================
        # 4. STORE IN CHROMADB
        # ====================================================

        with st.spinner(
            "🗄️ Storing information in ChromaDB..."
        ):

            count = store_documents(
                chunks,
                embeddings,
                filename
            )
            total_documents = get_unique_document_count()
            total_chunks = get_document_count()

        print(
    f"✓ Successfully stored {count} chunks in ChromaDB"
)

        print(
    f"✓ Total documents uploaded so far: {total_documents}"
)


        print(
    f"✓ Total chunks currently in ChromaDB: {total_chunks}"
)


        print(
            f"✓ Successfully stored {count} chunks in ChromaDB"
        )

        print(
            "✓ ChromaDB is ready for retrieval"
        )

        print("=" * 60)
        print("DOCUMENT PROCESSING COMPLETED")
        print("=" * 60)
        print("\n")


        # ====================================================
        # UPDATE SESSION STATE
        # ====================================================

        st.session_state.data_processed = True

        st.session_state.processed_source = source

        st.session_state.stored_chunks = count


        # ====================================================
        # SUCCESS MESSAGE
        # ====================================================

        st.success(
            "✅ Project information successfully stored in ChromaDB!"
        )

        st.caption(source)

        st.info(
            f"📚 {count} document chunks are now available "
            "for knowledge-base search."
        )


    except Exception as e:

        print("\n")
        print("=" * 60)
        print("ERROR DURING DOCUMENT PROCESSING")
        print("=" * 60)
        print(e)
        print("=" * 60)
        print("\n")


        st.error(
            f"❌ Processing failed: {e}"
        )


# ============================================================
# ASK ABOUT PROJECT
# ============================================================

# IMPORTANT:
# This entire section appears ONLY after successful processing.

if st.session_state.data_processed:

    st.divider()

    st.subheader("🔎 Ask About Your Project")

    st.write(
        "Ask a question about the information stored in "
        "the project knowledge base."
    )


    # --------------------------------------------------------
    # QUESTION INPUT
    # --------------------------------------------------------

    question = st.text_input(
        "Enter your question",
        placeholder=(
            "Example: What are the project risks?"
        ),
        key="project_question"
    )


    # --------------------------------------------------------
    # SEARCH BUTTON
    # --------------------------------------------------------

    if st.button(
        "🔍 Search Knowledge Base",
        use_container_width=True,
        key="search_knowledge_base"
    ):


        # ----------------------------------------------------
        # CHECK QUESTION
        # ----------------------------------------------------

        if not question.strip():

            st.warning(
                "⚠️ Please enter a question."
            )

            st.stop()


        try:

            # =================================================
            # SEARCH CHROMADB
            # =================================================

            print("\n")
            print("=" * 60)
            print("KNOWLEDGE BASE SEARCH")
            print("=" * 60)

            print(
                f"Question: {question}"
            )


            with st.spinner(
                "🔍 Searching ChromaDB..."
            ):


                # ---------------------------------------------
                # CREATE QUESTION EMBEDDING
                # ---------------------------------------------

                question_embedding = generate_embeddings(
                    [question]
                )[0]


                print(
                    "✓ Question embedding generated"
                )


                # ---------------------------------------------
                # SEARCH VECTOR DATABASE
                # ---------------------------------------------

                results = search_documents(
                    question_embedding,
                    n_results=3
                )


            print(
                f"✓ Retrieved {len(results)} relevant chunks"
            )

            print(
                "✓ Search completed successfully"
            )

            print("=" * 60)
            print("\n")


            # =================================================
            # DISPLAY SEARCH RESULTS
            # =================================================

            if results:

                st.success(
                    "✅ Relevant information retrieved from ChromaDB."
                )


                st.subheader(
                    "📖 Information from Project Knowledge Base"
                )


                for i, result in enumerate(results):

                    st.markdown(
                        f"### 📌 Relevant Information {i + 1}"
                    )

                    st.write(result)

                    st.divider()


            else:

                st.warning(
                    "No relevant information was found "
                    "in the project knowledge base."
                )


        except Exception as e:

            print("\n")
            print("=" * 60)
            print("ERROR DURING KNOWLEDGE BASE SEARCH")
            print("=" * 60)
            print(e)
            print("=" * 60)
            print("\n")


            st.error(
                f"❌ Search failed: {e}"
            )


    # =========================================================
    # MILESTONE 2 — PROJECT INTELLIGENCE
    # =========================================================

    st.divider()

    st.subheader("🤖 Project Intelligence")

    st.write(
        "Analyze the project using AI agents based on "
        "information retrieved from the project knowledge base."
    )


    # ---------------------------------------------------------
    # SELECT AGENT
    # ---------------------------------------------------------

    agent_type = st.selectbox(
        "Select Analysis Type",
        [
            "Scope & Deliverables",
            "Risks & Delivery Forecast",
            "Blockers & Action Items"
        ],
        key="agent_type"
    )


    # ---------------------------------------------------------
    # QUESTION INPUT
    # ---------------------------------------------------------

    agent_question = st.text_input(
        "Enter your project question",
        placeholder=(
            "Example: What are the main project deliverables?"
        ),
        key="agent_question"
    )


    # ---------------------------------------------------------
    # ANALYZE BUTTON
    # ---------------------------------------------------------

    if st.button(
        "🤖 Analyze Project",
        use_container_width=True,
        key="analyze_project"
    ):

        # -----------------------------------------------------
        # CHECK QUESTION
        # -----------------------------------------------------

        if not agent_question.strip():

            st.warning(
                "⚠️ Please enter a project question."
            )

            st.stop()


        try:

            print("\n")
            print("=" * 60)
            print("PROJECT INTELLIGENCE ANALYSIS")
            print("=" * 60)

            print(
                f"Question: {agent_question}"
            )

            print(
                f"Agent: {agent_type}"
            )


            # =================================================
            # STEP 1 — CREATE QUESTION EMBEDDING
            # =================================================

            with st.spinner(
                "🤖 Analyzing project..."
            ):

                question_embedding = generate_embeddings(
                    [agent_question]
                )[0]

                print(
                    "✓ Question embedding generated"
                )


                # =================================================
                # STEP 2 — RETRIEVE CONTEXT FROM CHROMADB
                # =================================================

                results = search_documents(
                    question_embedding,
                    n_results=3
                )

                print(
                    f"✓ Retrieved {len(results)} relevant chunks"
                )


                # =================================================
                # STEP 3 — CREATE CONTEXT
                # =================================================

                context = "\n\n".join(results)

                print(
                    "✓ Project context prepared"
                )


                # =================================================
                # STEP 4 — SELECT AGENT
                # =================================================

                if agent_type == "Scope & Deliverables":

                    result = scope_agent(
                        context,
                        agent_question
                    )

                    print(
                        "✓ Scope Agent executed"
                    )


                elif agent_type == "Risks & Delivery Forecast":

                    result = risk_agent(
                        context,
                        agent_question
                    )

                    print(
                        "✓ Risk Agent executed"
                    )


                else:

                    result = blocker_agent(
                        context,
                        agent_question
                    )

                    print(
                        "✓ Blocker Agent executed"
                    )


            # =================================================
            # ANALYSIS COMPLETED
            # =================================================

            print(
                "✓ Agent analysis completed"
            )

            print("=" * 60)
            print(
                "PROJECT INTELLIGENCE COMPLETED"
            )
            print("=" * 60)
            print("\n")


            # =================================================
            # DISPLAY RESULT
            # =================================================

            st.success(
                "✅ Project analysis completed successfully."
            )

            st.subheader(
                "📊 Project Analysis"
            )

            st.write(result)


        except Exception as e:

            print("\n")
            print("=" * 60)
            print(
                "ERROR DURING PROJECT INTELLIGENCE ANALYSIS"
            )
            print("=" * 60)
            print(e)
            print("=" * 60)
            print("\n")

            st.error(
                f"❌ Agent analysis failed: {e}"
            )
