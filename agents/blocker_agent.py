from llm.groq_client import generate_response


def blocker_agent(context, question):

    prompt = f"""
You are a Project Blocker and Action Item Identification Agent.

Use ONLY the information provided in the context.
Do not invent information.

Identify:

1. Blockers
2. Unresolved Issues
3. Pending Decisions
4. Action Items
5. Assigned Person
6. Due Date

For each item provide:

Blocker / Issue:
Status:
Action Item:
Assigned To:
Due Date:

If information is not available, write:

"Not specified in the provided context."

PROJECT CONTEXT:
{context}

USER QUESTION:
{question}

Give a clear and structured answer.
"""

    return generate_response(prompt)