from llm.groq_client import generate_response


def risk_agent(context, question):

    prompt = f"""
You are a Project Risk Detection and Delivery Forecasting Agent.

Use ONLY the information provided in the context.
Do not invent risks.

Identify the following:

1. Schedule Risks
2. Dependency Gaps
3. Potential Delays
4. Delivery Challenges
5. Risk Severity
6. Suggested Action

For each identified risk, provide:

Risk:
Reason:
Severity:
Delivery Impact:
Suggested Action:

Use only:
- High
- Medium
- Low

for severity.

If no specific risk is supported by the context, write:

"No specific risk was identified from the provided context."

PROJECT CONTEXT:
{context}

USER QUESTION:
{question}

Give a clear and structured answer.
"""

    return generate_response(prompt)