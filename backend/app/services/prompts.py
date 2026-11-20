"""All prompt templates. User/retrieved text is always wrapped with wrap_untrusted()."""
from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import BaseMessage

from app.services.llm_service import msgs
from app.utils.prompt_guard import GUARD_RULES, wrap_untrusted


def question_messages(*, role: str, topic: str, difficulty: str, qtype: str, context: str,
                      project_hint: str, avoid: list[str], weak_note: str, extra_explain: bool,
                      focus_note: str, jd_note: str) -> list[BaseMessage]:
    system = (
        f"You are a senior technical interviewer. Generate ONE interview question. {GUARD_RULES}\n"
        "Output ONLY the question text (1-3 sentences). No preface, no answer, no numbering."
    )
    parts = [f"Target role: {role or 'Software Engineer'}", f"Topic: {topic}",
             f"Difficulty: {difficulty}", f"Question type: {qtype}"]
    if qtype == "behavioral":
        parts.append("Ask a behavioral (STAR-style) question relevant to the role; ignore technical context.")
    if qtype == "project" and project_hint:
        parts.append("Ask a probing question about this candidate project (design choices, trade-offs, challenges):\n"
                     + wrap_untrusted(project_hint, "project", 1500))
    if context:
        parts.append("Reference material (use it to ground the question; do not quote it):\n"
                     + wrap_untrusted(context, "context", 4000))
    if jd_note:
        parts.append("Job description requirements:\n" + wrap_untrusted(jd_note, "jd", 1500))
    if weak_note:
        parts.append(f"Candidate history: {weak_note}")
    if extra_explain:
        parts.append("The candidate is struggling: ask a clear, approachable question and, in one short extra "
                     "sentence, give a hint about what a good answer covers.")
    if focus_note:
        parts.append("Candidate focus request:\n" + wrap_untrusted(focus_note, "focus", 300))
    if avoid:
        parts.append("Do NOT repeat or paraphrase these earlier questions:\n- " + "\n- ".join(a[:160] for a in avoid[:10]))
    return msgs(system, "\n\n".join(parts))


def evaluation_messages(*, question: str, answer: str, topic: str, difficulty: str, context: str,
                        extra_context: str, rubric: dict[str, float]) -> tuple[str, str]:
    crit = ", ".join(f"{k} (weight {v})" for k, v in rubric.items())
    system = (
        "You are a fair, rigorous technical interview evaluator. " + GUARD_RULES + "\n"
        "Score the candidate answer from 0 to 10. Judge against the reference context AND established technical "
        "knowledge. Do NOT penalise valid alternative terminology or phrasing. Be specific and constructive.\n"
        f"Evaluation rubric criteria (score each 0-10 in the `rubric` object using exactly these keys): {crit}.\n"
        "`max_score` must be 10. `ideal_answer` is a concise model answer. `follow_up_question` is one probing follow-up."
    )
    user = (
        f"Topic: {topic}\nDifficulty: {difficulty}\nQuestion: {question}\n\n"
        f"Candidate answer:\n{wrap_untrusted(answer, 'answer', 6000)}\n\n"
        + (f"Reference context:\n{wrap_untrusted(context, 'context', 4000)}\n\n" if context else
           "No reference material available: rely on established technical knowledge.\n\n")
        + (f"Additional context from the candidate:\n{wrap_untrusted(extra_context, 'extra', 1500)}\n" if extra_context else "")
    )
    return system, user


def feedback_messages(*, question: str, answer: str, evaluation: dict[str, Any], memory_note: str,
                      difficulty: str) -> list[BaseMessage]:
    system = (
        "You write short, encouraging, personalized feedback for an interview candidate (max 130 words, plain "
        "prose, second person). Mention what was good, the most important gap, and one concrete next step. "
        + GUARD_RULES
    )
    user = (
        f"Question: {question}\nDifficulty: {difficulty}\n"
        f"Candidate answer:\n{wrap_untrusted(answer, 'answer', 3000)}\n"
        f"Evaluation JSON: {json.dumps({k: evaluation.get(k) for k in ('score','missing_points','strengths','improvements')})}\n"
        f"Candidate history: {memory_note or 'none'}"
    )
    return msgs(system, user)


def summary_messages(stats: dict[str, Any]) -> list[BaseMessage]:
    system = (
        "You write a concise interview summary (max 170 words) for the candidate: overall performance, "
        "strengths, weaknesses, and 3 concrete next steps. Use only the statistics given. Plain prose with short "
        "bullet-style sentences."
    )
    return msgs(system, "Interview statistics:\n" + json.dumps(stats, default=str)[:4000])


def resume_messages(text: str) -> tuple[str, str]:
    system = ("You are a resume parser. Extract structured data from the resume. " + GUARD_RULES +
              " Only include facts present in the text. Skills must be concise technology/skill names.")
    return system, wrap_untrusted(text, "resume", 12000)


def jd_messages(text: str, role: str | None) -> tuple[str, str]:
    system = ("You are a job description analyzer. Extract role, required skills, preferred skills, "
              "responsibilities and seniority. " + GUARD_RULES)
    return system, f"Target role hint: {role or 'n/a'}\n" + wrap_untrusted(text, "jd", 12000)


def batch_question_messages(topic: str, difficulty: str, count: int, role: str | None) -> tuple[str, str]:
    system = ("You generate a list of interview questions. Return JSON with key `questions`: a list of objects "
              "with `question`, `topic`, `difficulty`, `expected_points` (list of strings).")
    return system, f"Role: {role or 'Software Engineer'}\nTopic: {topic}\nDifficulty: {difficulty}\nCount: {count}"


def rag_messages(question: str, context: str, grounded: bool) -> list[BaseMessage]:
    if grounded:
        system = (
            "You answer using ONLY the numbered context passages. Cite passages inline as [1], [2]. "
            "Clearly separate what comes from the documents from any extra suggestions by putting suggestions "
            "in a final section titled 'Model suggestions (not from your documents):'. " + GUARD_RULES
        )
        user = f"Context:\n{wrap_untrusted(context, 'context', 6000)}\n\nQuestion: {wrap_untrusted(question, 'question', 2000)}"
    else:
        system = ("The user's documents contain nothing relevant. Say so in one sentence, then give a brief, "
                  "accurate general-knowledge answer clearly labelled 'General knowledge (not from your documents)'. "
                  + GUARD_RULES)
        user = f"Question: {wrap_untrusted(question, 'question', 2000)}"
    return msgs(system, user)


def progress_report_messages(stats: dict[str, Any]) -> list[BaseMessage]:
    system = ("You write a short progress report (max 200 words) for an interview candidate from the statistics: "
              "trend, strongest and weakest topics, and a prioritized study plan.")
    return msgs(system, json.dumps(stats, default=str)[:5000])
