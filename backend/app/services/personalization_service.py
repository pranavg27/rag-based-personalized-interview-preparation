"""Adaptive difficulty + interview plan construction."""
from __future__ import annotations

from typing import Any

from app.config import Settings, get_settings
from app.models.enums import DIFFICULTY_LADDER, ROLE_TOPICS


def _dedupe(items: list[str]) -> list[str]:
    seen, out = set(), []
    for i in items:
        k = i.strip().lower()
        if k and k not in seen:
            seen.add(k)
            out.append(i.strip())
    return out


class PersonalizationService:
    def __init__(self, settings: Settings | None = None):
        self.s = settings or get_settings()

    def adjust_difficulty(self, current: str, scores: list[float]) -> tuple[str, str, bool]:
        """Returns (new_difficulty, reason, needs_extra_explanation)."""
        idx = DIFFICULTY_LADDER.index(current) if current in DIFFICULTY_LADDER else 1
        recent = scores[-self.s.difficulty_window:]
        if len(recent) < 2:
            return DIFFICULTY_LADDER[idx], "not enough answers yet", False
        avg = sum(recent) / len(recent)
        if avg >= self.s.difficulty_up_threshold and idx < 2:
            return DIFFICULTY_LADDER[idx + 1], f"avg {avg:.1f} >= {self.s.difficulty_up_threshold}: harder questions", False
        if avg < self.s.difficulty_down_threshold:
            if idx > 0:
                return DIFFICULTY_LADDER[idx - 1], f"avg {avg:.1f} < {self.s.difficulty_down_threshold}: easier questions", True
            return DIFFICULTY_LADDER[idx], "already easiest; adding extra explanations", True
        return DIFFICULTY_LADDER[idx], "performance steady", False

    def initial_difficulty(self, requested: str | None, preferred: str | None, recent_avg: float | None) -> str:
        if requested in DIFFICULTY_LADDER:
            return requested
        if preferred in DIFFICULTY_LADDER:
            return preferred
        if recent_avg is not None:
            return "hard" if recent_avg >= self.s.difficulty_up_threshold else "easy" if recent_avg < self.s.difficulty_down_threshold else "medium"
        return "medium"

    def build_plan(self, *, mode: str, role: str | None, requested_skills: list[str], resume_skills: list[str],
                   jd_skills: list[str], weak_topics: list[str], num_questions: int,
                   has_projects: bool = False, variation: int = 0) -> list[dict[str, Any]]:
        role_topics = ROLE_TOPICS.get(role or "", ["Python", "DSA", "DBMS", "System Design"])
        if mode == "skill_based":
            pool = requested_skills
        elif mode == "resume_based":
            pool = resume_skills[:12] + requested_skills
        elif mode == "jd_based":
            pool = jd_skills + requested_skills
        elif mode == "weak_topics":
            pool = weak_topics
        else:  # mock_interview
            pool = requested_skills + resume_skills[:6] + jd_skills[:6]
        pool = _dedupe(pool) or _dedupe(role_topics)
        weak_set = {w.lower() for w in weak_topics}
        weighted = pool + [t for t in pool if t.lower() in weak_set and mode != "weak_topics"]
        if variation:
            r = variation % len(weighted)
            weighted = weighted[r:] + weighted[:r]

        def topic_at(i: int) -> str:
            return weighted[i % len(weighted)]

        plan: list[dict[str, Any]] = []
        if mode == "mock_interview":
            n = num_questions
            for i in range(n):
                if n >= 3 and i == n - 1:
                    plan.append({"type": "behavioral", "topic": "Behavioral", "difficulty": "medium"})
                elif n >= 5 and i == min(3, n - 2):
                    plan.append({"type": "followup", "topic": plan[-1]["topic"] if plan else topic_at(i), "difficulty": None})
                else:
                    frac = i / max(n - 2, 1)
                    diff = "easy" if frac < 0.34 else "medium" if frac < 0.67 else "hard"
                    plan.append({"type": "technical", "topic": topic_at(i), "difficulty": diff})
        else:
            project_slots = {1, 3} if (mode == "resume_based" and has_projects) else set()
            for i in range(num_questions):
                t = "project" if i in project_slots else "technical"
                plan.append({"type": t, "topic": "Projects" if t == "project" else topic_at(i), "difficulty": None})
        for i, p in enumerate(plan):
            p["index"] = i
        return plan
