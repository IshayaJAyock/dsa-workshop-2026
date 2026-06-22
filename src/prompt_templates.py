"""Prompt patterns used in the workshop notebooks."""

from __future__ import annotations

import pandas as pd

RUBRIC_CRITERIA = [
    "Clarity",
    "Relevance",
    "Faithfulness",
    "Completeness",
    "Format control",
    "Usefulness",
    "Safety",
]


def build_prompt(
    role: str,
    task: str,
    context: str,
    constraints: str,
    output_format: str,
    quality_check: str,
) -> str:
    """
    Build a structured prompt.

    Pattern we use in class:
    Role + Task + Context + Constraints + Output format + Quality check
    """
    return f"""Role:
{role}

Task:
{task}

Context:
{context}

Constraints:
{constraints}

Output format:
{output_format}

Quality check:
{quality_check}
"""


def build_learning_support_prompt(user_text: str, audience: str, task_type: str) -> str:
    """Used in Notebook 2."""
    return build_prompt(
        role=f"You are a helpful teaching assistant working with {audience}.",
        task=f"Help with this task: {task_type}.",
        context=f"Source material:\n{user_text}",
        constraints=(
            "- Stay faithful to the source.\n"
            "- Use plain language.\n"
            "- Do not invent references or exam answers."
        ),
        output_format=(
            "Use markdown headings and bullets:\n"
            "1. Short answer — 2–3 sentences\n"
            "2. Key points — bullet list\n"
            "3. One example — concrete and simple\n"
            "4. One common misunderstanding — one sentence\n"
            "5. Quiz — two short questions"
        ),
        quality_check="Would this help a student study, without misleading them?",
    )


def build_travel_brief_prompt(
    base_city: str,
    base_country: str,
    destination_city: str,
    destination_country: str,
    traveller_profile: str,
    weather: dict,
    distance: dict,
    exchange: dict,
    places: list[dict],
) -> str:
    """Used in Notebook 3. Facts come from tools — the model should not guess them."""
    place_lines = "\n".join(f"- {p['name']}: {p['short_note']}" for p in places)
    return f"""You are a travel briefing assistant for a data scientist visiting Africa.

Traveller profile: {traveller_profile}
Route: {base_city}, {base_country} → {destination_city}, {destination_country}
The workshop destination is {destination_city}, {destination_country}.

Use only the tool evidence below. Do not invent weather, rates, or distances.
If a tool returned an error, say the fact is unavailable.

Weather evidence: {weather}
Distance evidence: {distance}
Exchange evidence: {exchange}
Suggested places:
{place_lines}

Write in clear markdown with these sections:
1. Short briefing — who is travelling and why
2. Weather summary — use only the weather evidence for {destination_city}
3. Distance note — straight-line km from {base_city} to {destination_city}
4. Exchange rate note — explain the sample conversion
5. Top 3 places — bullet list with a data-science angle for each
6. Practical caution — one responsible travel reminder
7. One-sentence summary

Reminder: students should verify travel details before real trips.
"""


PROMPT_PRESETS = {
    "Explain like a lecturer": {
        "task_type": "Explain simply",
        "audience": "undergraduate students",
        "system_prompt": "Explain patiently, as you would in a practical class.",
    },
    "Create study notes": {
        "task_type": "Create study notes",
        "audience": "students revising for a test",
        "system_prompt": "Write short study notes with headings and bullets.",
    },
    "Generate quiz": {
        "task_type": "Generate quiz questions",
        "audience": "students preparing for a quiz",
        "system_prompt": "Write two fair questions that test understanding.",
    },
    "Rewrite for beginners": {
        "task_type": "Rewrite for beginners",
        "audience": "first-year students",
        "system_prompt": "Rewrite in simple language with one concrete example.",
    },
    "Review my answer": {
        "task_type": "Identify key concepts",
        "audience": "students checking their work",
        "system_prompt": "Give brief, constructive feedback.",
    },
    "Create project feedback": {
        "task_type": "Summarise",
        "audience": "students submitting a draft",
        "system_prompt": "Note one strength, one gap, and one next step.",
    },
}


def score_output_rubric(scores: dict[str, int]) -> pd.DataFrame:
    """Turn your 1–5 scores into a small table."""
    return pd.DataFrame({"Criterion": RUBRIC_CRITERIA, "Score (1-5)": [scores.get(c) for c in RUBRIC_CRITERIA]})


def build_llm_judge_prompt(task: str, prompt_used: str, model_output: str) -> str:
    """Ask the model to critique an answer — then you still check it yourself."""
    joined = ", ".join(RUBRIC_CRITERIA)
    return f"""You are helping a student evaluate an AI answer.

Task: {task}
Prompt used: {prompt_used}
Model output: {model_output}

Score each criterion from 1 to 5: {joined}
Give one strength, one weakness, and remind the reader that a human must review the result.
"""
