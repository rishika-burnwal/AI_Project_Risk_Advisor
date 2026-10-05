import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

import streamlit as st

from rag.embeddings import generate_embeddings
from rag.vector_store import search_documents
from llm.groq_client import generate_response


def get_context(question):

    question_embedding = generate_embeddings(
        [question]
    )[0]

    results = search_documents(
        question_embedding,
        n_results=3
    )

    return "\n\n".join(results)


def generate_answer(question, context):

    prompt = f"""
You are a Project Intelligence Assistant.

Answer the user's question using ONLY the information
provided in the project context.

Do not invent, assume, or add information that is not
present in the project context.

If the information is not available, say:

"Information not available in the provided project documents."

PROJECT CONTEXT:

{context}

USER QUESTION:

{question}

Give a clear and concise answer.
"""

    return generate_response(prompt)


def validate_answer(question, context, answer):

    prompt = f"""
You are validating an AI Project Intelligence Assistant.

Determine whether the assistant's answer is supported
by the provided project context.

QUESTION:
{question}

PROJECT CONTEXT:
{context}

ASSISTANT ANSWER:
{answer}

Evaluate:

1. Accuracy
2. Relevance
3. Grounding in the provided project context

Return:

Accuracy: Accurate / Partially Accurate / Inaccurate

Relevance: Relevant / Partially Relevant / Irrelevant

Grounding: Grounded / Partially Grounded / Not Grounded

Explanation:
Give a short explanation based only on the project context.
"""

    return generate_response(prompt)


st.title("🧪 Conversational Assistant Validation")

st.write(
    "Enter a question about the uploaded project documents "
    "to validate the assistant's answer."
)

st.divider()


question = st.text_input(
    "Enter your project question",
    placeholder="Example: What are the current project risks?",
    key="validation_question"
)


if st.button(
    "🔍 Validate Answer",
    use_container_width=True,
    key="validate_answer"
):

    if not question.strip():

        st.warning(
            "⚠️ Please enter a project question."
        )

        st.stop()

    try:

        with st.spinner(
            "🔎 Retrieving project information..."
        ):

            context = get_context(question)


        if not context.strip():

            st.warning(
                "No relevant information was found "
                "in the project knowledge base."
            )

            st.stop()


        st.success(
            "✅ Relevant project information retrieved."
        )


        with st.spinner(
            "🤖 Generating assistant answer..."
        ):

            answer = generate_answer(
                question,
                context
            )


        st.subheader("💬 Assistant Answer")

        st.write(answer)


        with st.spinner(
            "🧪 Validating answer..."
        ):

            validation = validate_answer(
                question,
                context,
                answer
            )


        st.subheader("📊 Validation Result")

        st.write(validation)


    except Exception as e:

        st.error(
            f"❌ Validation failed: {e}"
        )