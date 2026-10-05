import json

from llm.groq_client import generate_response


def analyze_project_health(context):

    prompt = f"""
You are an AI Project Health Analysis Agent.

Analyze ONLY the project information provided below.

Do not invent information.
Do not use general knowledge.
Do not assume information that is not present.

Analyze the project using these three dimensions:

1. Scope Clarity
2. Timeline Risk
3. Blocker Count

SCOPE CLARITY:

Evaluate how clearly the project scope is defined based on:
- Requirements
- Functional requirements
- Non-functional requirements
- Deliverables
- Project milestones
- Responsibilities

Give a score from 0 to 100.

100 = scope is very clearly defined
80 = scope is mostly clear
60 = some important scope information is missing
40 = significant scope information is unclear
20 = very limited scope information
0 = no usable scope information

TIMELINE RISK:

Evaluate timeline risk based ONLY on:
- Project milestones
- Deadlines
- Schedule information
- Delayed tasks
- Pending tasks
- Dependencies
- Schedule-related issues

Give a risk value from 0 to 100.

0 = no timeline risk identified
20 = low timeline risk
40 = moderate timeline risk
60 = significant timeline risk
80 = high timeline risk
100 = very high timeline risk

BLOCKER COUNT:

Count only blockers, unresolved issues, or explicitly mentioned obstacles.

Do NOT count normal project tasks as blockers.

If no blockers are explicitly mentioned, return 0.

Return ONLY valid JSON in exactly this format:

{{
    "scope_clarity": 0,
    "timeline_risk": 0,
    "blocker_count": 0,
    "scope_reason": "",
    "timeline_reason": "",
    "blocker_reason": ""
}}

PROJECT INFORMATION:

{context}
"""

    response = generate_response(prompt)

    try:
        response = response.strip()

        if response.startswith("```"):
            response = response.replace("```json", "")
            response = response.replace("```", "")
            response = response.strip()

        result = json.loads(response)

        scope_clarity = max(
            0,
            min(
                100,
                int(result.get("scope_clarity", 0))
            )
        )

        timeline_risk = max(
            0,
            min(
                100,
                int(result.get("timeline_risk", 0))
            )
        )

        blocker_count = max(
            0,
            int(result.get("blocker_count", 0))
        )

        return {
            "scope_clarity": scope_clarity,
            "timeline_risk": timeline_risk,
            "blocker_count": blocker_count,
            "scope_reason": result.get(
                "scope_reason",
                "Not specified in the project documents."
            ),
            "timeline_reason": result.get(
                "timeline_reason",
                "Not specified in the project documents."
            ),
            "blocker_reason": result.get(
                "blocker_reason",
                "No blockers were identified in the project documents."
            )
        }

    except Exception:
        return {
            "scope_clarity": 0,
            "timeline_risk": 0,
            "blocker_count": 0,
            "scope_reason": "Unable to analyze the project information.",
            "timeline_reason": "Unable to analyze the project information.",
            "blocker_reason": "Unable to analyze the project information."
        }


def calculate_health_score(
    scope_clarity,
    timeline_risk,
    blocker_count
):

    scope_score = max(
        0,
        min(
            100,
            scope_clarity
        )
    )

    timeline_score = max(
        0,
        min(
            100,
            100 - timeline_risk
        )
    )

    if blocker_count == 0:
        blocker_score = 100

    elif blocker_count <= 2:
        blocker_score = 80

    elif blocker_count <= 4:
        blocker_score = 60

    elif blocker_count <= 6:
        blocker_score = 40

    else:
        blocker_score = 20

    overall_score = round(
        (
            scope_score
            + timeline_score
            + blocker_score
        ) / 3
    )

    return {
        "overall_score": overall_score,
        "scope_clarity": scope_score,
        "timeline_score": timeline_score,
        "blocker_score": blocker_score,
        "timeline_risk": timeline_risk,
        "blocker_count": blocker_count
    }


def get_health_status(score):

    if score >= 80:
        return "Healthy"

    elif score >= 60:
        return "Moderate"

    elif score >= 40:
        return "At Risk"

    else:
        return "Critical"