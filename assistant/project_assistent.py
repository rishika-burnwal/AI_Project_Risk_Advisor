from llm.groq_client import generate_response


def project_assistant(
    context,
    question
):

    if not context or not context.strip():

        return (
            "I could not find enough project information "
            "in the current knowledge base to answer "
            "this question."
        )

    if not question or not question.strip():

        return (
            "Please enter a project-related question."
        )

    prompt = f"""
You are the Project Intelligence Assistant.

Your job is to answer questions about the CURRENT project
using the project information retrieved from its knowledge base.

The user may ask direct questions, analytical questions,
status questions, risk questions, progress questions,
or decision-support questions.

IMPORTANT RULES:

1. Use ONLY the project information provided below.

2. Do NOT invent project facts.

3. Do NOT use outside knowledge about the project.

4. You MAY make reasonable conclusions from explicit
   evidence in the provided project information.

5. When the user asks a question such as:

   "Are we on the right track?"
   "Is the project progressing well?"
   "Are we likely to meet the deadline?"
   "Is the project at risk?"
   "What should we focus on next?"

   do NOT simply say that the information is insufficient.

   Instead, analyze the available evidence such as:

   - completed work
   - current progress
   - pending tasks
   - milestones
   - deadlines
   - risks
   - blockers
   - dependencies
   - project status
   - decisions
   - action items
   - responsibilities

   Then provide a reasoned conclusion.

6. If the evidence supports a conclusion, clearly state it.

7. If the evidence is mixed, clearly say that the project
   appears to be partially on track but has specific concerns.

8. If the evidence shows serious blockers or risks,
   clearly identify that the project is at risk.

9. If the documents genuinely do not contain enough
   information, explain exactly what information is missing.

10. Never claim certainty when the documents do not support it.

11. Do not mention ChromaDB, embeddings, vector databases,
    retrieval pipelines, or internal AI implementation
    unless the user specifically asks about them.

12. Do not answer with only:
    "Insufficient information in the project knowledge base."

    If information is incomplete, explain what is available
    and what is missing.

13. Use clear Markdown formatting.

14. For status questions, use this structure when appropriate:

    ## Assessment

    State the overall conclusion.

    ## Evidence

    - Evidence from the project documents
    - Evidence from progress
    - Evidence from risks or blockers

    ## Concerns

    - Important unresolved issues

    ## Recommended Next Steps

    - Practical actions supported by the documents

15. For simple factual questions, give a direct answer
    without unnecessary sections.

USER QUESTION:

{question}

CURRENT PROJECT INFORMATION:

{context}

Now answer the user's question using only the
project information above.
"""

    try:

        result = generate_response(
            prompt,
            max_completion_tokens=8192,
            reasoning_effort="low"
        )

        if result is None:

            return (
                "I could not generate an answer "
                "from the current project information."
            )

        result = str(
            result
        ).strip()

        if not result:

            return (
                "I could not generate an answer "
                "from the current project information."
            )

        result = (
            result
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
            .replace(
                "<BR>",
                "\n"
            )
            .replace(
                "<BR/>",
                "\n"
            )
            .replace(
                "<BR />",
                "\n"
            )
        )

        return result

    except Exception as e:

        print(
            "Project Assistant Error:",
            repr(e)
        )

        return (
            "The project assistant encountered "
            "an error while generating the answer."
        )