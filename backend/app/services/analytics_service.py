from __future__ import annotations

from collections import defaultdict
from typing import Any

from app.database import mongo as m
from app.services.memory_service import MemoryService
from app.utils.ids import serialize


def _avg(xs: list[float]) -> float | None:
    return round(sum(xs) / len(xs), 2) if xs else None


class AnalyticsService:
    def __init__(self, memory: MemoryService):
        self.memory = memory

    async def _scored_answers(self, user_id: str) -> list[dict]:
        rows = await m.get_db()[m.ANSWERS].find({"user_id": user_id}).sort("created_at", 1).to_list(3000)
        return [r for r in rows if not r.get("skipped") and r.get("feedback_decision") != "reject"
                and r.get("score") is not None]

    async def performance(self, user_id: str) -> dict[str, Any]:
        rows = await self._scored_answers(user_id)
        by_topic, by_diff = defaultdict(list), defaultdict(list)
        for r in rows:
            by_topic[r["topic"]].append(r["score"])
            by_diff[r.get("difficulty") or "medium"].append(r["score"])
        return {"answers_scored": len(rows), "overall_avg": _avg([r["score"] for r in rows]),
                "by_topic": {t: {"avg": _avg(v), "attempts": len(v)} for t, v in by_topic.items()},
                "by_difficulty": {d: {"avg": _avg(v), "attempts": len(v)} for d, v in by_diff.items()}}

    async def history(self, user_id: str, limit: int = 20, skip: int = 0) -> list[dict]:
        rows = await (m.get_db()[m.INTERVIEWS].find({"user_id": user_id}).sort("created_at", -1)
                      .skip(skip).limit(limit).to_list(limit))
        keep = ("_id", "mode", "role", "status", "avg_score", "answered", "total_questions", "difficulty",
                "created_at", "updated_at")
        return [serialize({k: r.get(k) for k in keep}) for r in rows]

    async def weak_topics(self, user_id: str, limit: int = 5) -> list[dict]:
        weak = await self.memory.weak_topics(user_id, limit)
        if weak:
            return weak
        perf = (await self.performance(user_id))["by_topic"]
        return sorted(({"topic": t, "avg_score": v["avg"], "attempts": v["attempts"]}
                       for t, v in perf.items() if v["avg"] is not None and v["avg"] < self.memory.s.weak_topic_threshold),
                      key=lambda x: x["avg_score"])[:limit]

    async def recommendations(self, user_id: str) -> list[dict]:
        recs = []
        for w in await self.weak_topics(user_id, 5):
            recs.append({"topic": w["topic"], "reason": f"Average {w['avg_score']}/10 over {w['attempts']} answers",
                         "action": f"Review fundamentals of {w['topic']} and run a 'Practice weak topics' interview."})
        for mk in await self.memory.common_mistakes(user_id, 3):
            recs.append({"topic": mk["topic"], "reason": f"Recurring gap ({mk['count']}x): {mk['point'][:100]}",
                         "action": "Write a short note on this concept and answer one practice question about it."})
        return recs

    async def dashboard(self, user_id: str) -> dict[str, Any]:
        db = m.get_db()
        interviews = await db[m.INTERVIEWS].find({"user_id": user_id}).sort("created_at", -1).to_list(500)
        finished = [i for i in interviews if i.get("status") in ("completed", "ended")]
        scores = [i["avg_score"] for i in finished if i.get("avg_score") is not None]
        strong = await self.memory.strong_topics(user_id, 5)
        weak = await self.weak_topics(user_id, 5)
        trend = [{"date": serialize(i["created_at"])[:10], "avg_score": i["avg_score"]}
                 for i in reversed(finished) if i.get("avg_score") is not None][-15:]
        return {"total_interviews": len(interviews), "completed_interviews": len(finished),
                "average_score": _avg(scores), "strongest_skills": strong, "weakest_skills": weak,
                "recent_history": await self.history(user_id, 5), "score_trend": trend,
                "recommended_topics": [w["topic"] for w in weak]
                or [r["topic"] for r in await self.recommendations(user_id) if r["topic"]][:3]}

    async def progress_stats(self, user_id: str) -> dict[str, Any]:
        d = await self.dashboard(user_id)
        return {"dashboard": d, "performance": await self.performance(user_id),
                "memory_summary": await self.memory.summarize(user_id),
                "recommendations": await self.recommendations(user_id)}
