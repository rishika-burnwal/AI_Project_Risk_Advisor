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


st.set_page_config(
    page_title="AI Project Intelligence & Risk Advisor",
    page_icon="📊",
    layout="wide"
)


if "data_processed" not in st.session_state:
    st.session_state.data_processed = False

if "processed_source" not in st.session_state:
    st.session_state.processed_source = []

if "stored_chunks" not in st.session_state:
    st.session_state.stored_chunks = 0


st.title("📊 AI Project Intelligence & Risk Advisor")

st.write(
    "Upload project documents or enter project information "
    "to build the project knowledge base."
)

st.divider()


st.subheader("📄 Project Document")

uploaded_files = st.file_uploader(
    "Upload PDF, DOCX, CSV or TXT",
    type=["pdf", "docx", "csv", "txt"],
    accept_multiple_files=True,
    key="project_document_uploader"
)


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


if st.button(
    "🚀 Process Project Data",
    use_container_width=True,
    key="process_project_data"
):

    if not uploaded_files and not plain_text.strip():

        st.warning(
            "⚠️ Please upload a document or enter project information."
        )

        st.stop()


    try:

        print("\n")
        print("=" * 60)
        print("DOCUMENT PROCESSING STARTED")
        print("=" * 60)


        total_processed_chunks = 0
        processed_filenames = []


        if uploaded_files:

            print(
                f"✓ {len(uploaded_files)} project documents received"
            )


            for uploaded_file in uploaded_files:

                print("\n")
                print("-" * 60)
                print(
                    f"PROCESSING FILE: {uploaded_file.name}"
                )
                print("-" * 60)


                text = extract_text(uploaded_file)

                filename = uploaded_file.name


                print(
                    f"✓ File received: {filename}"
                )


                if not text or not text.strip():

                    print(
                        f"⚠️ No readable text found in {filename}"
                    )

                    continue


                print(
                    "✓ Document text extracted successfully"
                )


                with st.spinner(
                    f"✂️ Creating document chunks for {filename}..."
                ):

                    chunks = create_chunks(text)


                print(
                    f"✓ {len(chunks)} document chunks created"
                )


                with st.spinner(
                    f"🔢 Generating embeddings for {filename}..."
                ):

                    embeddings = generate_embeddings(chunks)


                print(
                    f"✓ Embeddings created for "
                    f"{len(embeddings)} chunks"
                )


                with st.spinner(
                    f"🗄️ Storing {filename} in ChromaDB..."
                ):

                    count = store_documents(
                        chunks,
                        embeddings,
                        filename
                    )


                total_processed_chunks += count

                processed_filenames.append(filename)


                print(
                    f"✓ Successfully stored {count} chunks "
                    f"from {filename}"
                )


        if plain_text.strip():

            print("\n")
            print("-" * 60)
            print("PROCESSING MANUAL PROJECT INFORMATION")
            print("-" * 60)


            text = plain_text

            filename = "manual_project_information.txt"


            print(
                "✓ Manual project information received"
            )


            chunks = create_chunks(text)


            print(
                f"✓ {len(chunks)} document chunks created"
            )


            embeddings = generate_embeddings(chunks)


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

            processed_filenames.append(filename)


            print(
                f"✓ Successfully stored {count} chunks "
                "from manual project information"
            )


        total_documents = get_unique_document_count()

        total_chunks = get_document_count()


        print("\n")
        print("=" * 60)
        print("DOCUMENT PROCESSING SUMMARY")
        print("=" * 60)


        print(
            f"✓ Documents processed in this upload: "
            f"{len(processed_filenames)}"
        )


        print(
            f"✓ Total documents in ChromaDB: "
            f"{total_documents}"
        )


        print(
            f"✓ Total chunks currently in ChromaDB: "
            f"{total_chunks}"
        )


        print(
            "✓ ChromaDB is ready for retrieval"
        )


        print("=" * 60)
        print("DOCUMENT PROCESSING COMPLETED")
        print("=" * 60)
        print("\n")


        st.session_state.data_processed = True

        st.session_state.processed_source = processed_filenames

        st.session_state.stored_chunks = total_processed_chunks


        st.success(
            "✅ Project information successfully stored in ChromaDB!"
        )


        st.caption(
            "Uploaded: " + ", ".join(processed_filenames)
        )


        st.info(
            f"📚 {total_processed_chunks} document chunks are now "
            "available for knowledge-base search."
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




if st.session_state.data_processed:

    st.divider()

    st.subheader("🔎 Ask About Your Project")

    st.write(
        "Ask a question about the information stored in "
        "the project knowledge base."
    )


    question = st.text_input(
        "Enter your question",
        placeholder=(
            "Example: What are the project risks?"
        ),
        key="project_question"
    )


    if st.button(
        "🔍 Search Knowledge Base",
        use_container_width=True,
        key="search_knowledge_base"
    ):


        if not question.strip():

            st.warning(
                "⚠️ Please enter a question."
            )

            st.stop()


        try:


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


                question_embedding = generate_embeddings(
                    [question]
                )[0]


                print(
                    "✓ Question embedding generated"
                )


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


    st.divider()

    st.subheader("🤖 Project Intelligence")

    st.write(
        "Analyze the project using AI agents based on "
        "information retrieved from the project knowledge base."
    )


    agent_type = st.selectbox(
        "Select Analysis Type",
        [
            "Scope & Deliverables",
            "Risks & Delivery Forecast",
            "Blockers & Action Items"
        ],
        key="agent_type"
    )


    agent_question = st.text_input(
        "Enter your project question",
        placeholder=(
            "Example: What are the main project deliverables?"
        ),
        key="agent_question"
    )


    if st.button(
        "🤖 Analyze Project",
        use_container_width=True,
        key="analyze_project"
    ):


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


            with st.spinner(
                "🤖 Analyzing project..."
            ):

                question_embedding = generate_embeddings(
                    [agent_question]
                )[0]

                print(
                    "✓ Question embedding generated"
                )


                results = search_documents(
                    question_embedding,
                    n_results=3
                )

                print(
                    f"✓ Retrieved {len(results)} relevant chunks"
                )


                context = "\n\n".join(results)

                print(
                    "✓ Project context prepared"
                )


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


            print(
                "✓ Agent analysis completed"
            )

            print("=" * 60)
            print(
                "PROJECT INTELLIGENCE COMPLETED"
            )
            print("=" * 60)
            print("\n")


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
