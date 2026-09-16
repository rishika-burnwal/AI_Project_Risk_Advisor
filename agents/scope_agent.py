from llm.groq_client import generate_response


def scope_agent(context, question):

    prompt = f"""
You are a Project Scope and Deliverable Extraction Agent.

Use ONLY the information provided in the context.
Do not invent information.

Extract the following:

1. Project Goal
2. Requirements
3. Deliverables
4. Milestones
5. Timeline
6. Responsibilities

If information is not available, write:
"Not specified in the provided context."

PROJECT CONTEXT:
{context}

USER QUESTION:
{question}

Give a clear and structured answer.
"""

    return generate_response(prompt)