import streamlit as st

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

    if uploaded_file is None and not plain_text.strip():

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