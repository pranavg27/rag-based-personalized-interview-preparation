"""Long-term, per-user memory in MongoDB (separate from LangGraph's short-term checkpoint state)."""
from __future__ import annotations

import hashlib
import re
from typing import Any

from app.config import Settings, get_settings
from app.database import mongo as m
from app.errors import NotFoundError
from app.utils.ids import new_id, utcnow

KINDS = {"preferred_role", "preferred_difficulty", "skill", "topic_stat", "mistake", "learning_pref",
         "progress", "progress_summary"}
KIND_WEIGHT = {"topic_stat": 3.0, "mistake": 2.0, "skill": 1.5, "preferred_role": 1.0,
               "preferred_difficulty": 1.0, "learning_pref": 1.0, "progress": 0.5, "progress_summary": 0.5}


def _norm(topic: str) -> str:
    return re.sub(r"\s+", " ", topic.strip().lower())[:80]


class MemoryService:
    def __init__(self, settings: Settings | None = None):
        self.s = settings or get_settings()

    @property
    def col(self):
        return m.get_db()[m.MEMORY]

    # --- CRUD ----------------------------------------------------------
    async def upsert(self, user_id: str, kind: str, key: str, value: str = "",
                     data: dict[str, Any] | None = None, weight: float = 1.0) -> dict:
        now = utcnow()
        await self.col.update_one(
            {"user_id": user_id, "kind": kind, "key": key},
            {"$set": {"value": value, "data": data or {}, "weight": weight, "updated_at": now},
             "$setOnInsert": {"_id": new_id(), "created_at": now}},
            upsert=True,
        )
        return await self.col.find_one({"user_id": user_id, "kind": kind, "key": key})

    async def get_entry(self, user_id: str, kind: str, key: str) -> dict | None:
        return await self.col.find_one({"user_id": user_id, "kind": kind, "key": key})

    async def create(self, user_id: str, kind: str, key: str, value: str = "", data: dict | None = None) -> dict:
        return await self.upsert(user_id, kind, key, value, data)

    async def list(self, user_id: str, kinds: list[str] | None = None, limit: int = 300) -> list[dict]:
        flt: dict[str, Any] = {"user_id": user_id}
        if kinds:
            flt["kind"] = {"$in": kinds}
        return await self.col.find(flt).sort("updated_at", -1).to_list(limit)

    async def update(self, user_id: str, memory_id: str, value: str | None = None,
                     data: dict | None = None) -> dict:
        upd: dict[str, Any] = {"updated_at": utcnow()}
        if value is not None:
            upd["value"] = value
        if data is not None:
            upd["data"] = data
        res = await self.col.update_one({"_id": memory_id, "user_id": user_id}, {"$set": upd})
        if res.matched_count == 0:
            raise NotFoundError("Memory not found.")
        return await self.col.find_one({"_id": memory_id, "user_id": user_id})

    async def delete(self, user_id: str, memory_id: str) -> bool:
        res = await self.col.delete_one({"_id": memory_id, "user_id": user_id})
        return res.deleted_count > 0

    # --- performance tracking ---------------------------------------------
    async def record_answer(self, user_id: str, topic: str, score: float, difficulty: str,
                            missing_points: list[str] | None = None) -> None:
        key = _norm(topic)
        if not key:
            return
        entry = await self.get_entry(user_id, "topic_stat", key)
        data = (entry or {}).get("data") or {"topic": topic.strip(), "attempts": 0, "recent": []}
        data["attempts"] = int(data.get("attempts", 0)) + 1
        data["recent"] = (list(data.get("recent", [])) + [round(float(score), 2)])[-5:]
        data["avg"] = round(sum(data["recent"]) / len(data["recent"]), 2)
        data["last_score"], data["last_difficulty"] = round(float(score), 2), difficulty
        await self.upsert(user_id, "topic_stat", key, f"avg {data['avg']}/10 over {data['attempts']} answers", data, 3.0)
        for point in (missing_points or [])[:3]:
            pk = hashlib.sha1(_norm(point).encode()).hexdigest()[:16]
            old = await self.get_entry(user_id, "mistake", pk)
            count = int(((old or {}).get("data") or {}).get("count", 0)) + 1
            await self.upsert(user_id, "mistake", pk, point[:240], {"topic": topic, "count": count}, 2.0)

    async def record_interview(self, user_id: str, interview_id: str, avg_score: float | None,
                               topic_scores: dict[str, float], role: str | None) -> None:
        if avg_score is None:
            return
        await self.upsert(user_id, "progress", interview_id, f"avg {avg_score}/10",
                          {"avg_score": avg_score, "topic_scores": topic_scores, "role": role,
                           "at": utcnow().isoformat()}, 0.5)
        await self._compact_progress(user_id)

    async def _compact_progress(self, user_id: str) -> None:
        keep = self.s.memory_progress_keep
        rows = await self.col.find({"user_id": user_id, "kind": "progress"}).sort("updated_at", 1).to_list(1000)
        if len(rows) <= keep:
            return
        old = rows[: len(rows) - keep]
        prev = await self.get_entry(user_id, "progress_summary", "history")
        pd = (prev or {}).get("data") or {"count": 0, "avg_score": 0.0}
        n_old = len(old)
        total = pd["avg_score"] * pd["count"] + sum(r["data"]["avg_score"] for r in old)
        count = pd["count"] + n_old
        await self.upsert(user_id, "progress_summary", "history", f"{count} older interviews, avg {total / count:.1f}/10",
                          {"count": count, "avg_score": round(total / count, 2)}, 0.5)
        for r in old:
            await self.col.delete_one({"_id": r["_id"], "user_id": user_id})

    # --- views used for personalisation -------------------------------------
    async def _stats(self, user_id: str) -> list[dict]:
        return [r["data"] for r in await self.list(user_id, ["topic_stat"])]

    async def weak_topics(self, user_id: str, limit: int = 5) -> list[dict]:
        out = []
        for d in await self._stats(user_id):
            ok_attempts = d["attempts"] >= self.s.min_attempts_for_topic or d["avg"] < 3
            if ok_attempts and d["avg"] < self.s.weak_topic_threshold:
                out.append({"topic": d["topic"], "avg_score": d["avg"], "attempts": d["attempts"]})
        return sorted(out, key=lambda x: x["avg_score"])[:limit]

    async def strong_topics(self, user_id: str, limit: int = 5) -> list[dict]:
        out = [{"topic": d["topic"], "avg_score": d["avg"], "attempts": d["attempts"]}
               for d in await self._stats(user_id)
               if d["attempts"] >= self.s.min_attempts_for_topic and d["avg"] >= self.s.strong_topic_threshold]
        return sorted(out, key=lambda x: -x["avg_score"])[:limit]

    async def common_mistakes(self, user_id: str, limit: int = 5) -> list[dict]:
        rows = await self.list(user_id, ["mistake"])
        rows.sort(key=lambda r: -int(r["data"].get("count", 1)))
        return [{"point": r["value"], "topic": r["data"].get("topic"), "count": r["data"].get("count", 1)}
                for r in rows[:limit]]

    async def get_preferences(self, user_id: str) -> dict[str, Any]:
        prefs: dict[str, Any] = {}
        for kind in ("preferred_role", "preferred_difficulty", "learning_pref"):
            e = await self.get_entry(user_id, kind, "default")
            if e:
                prefs[kind] = e["value"]
        doc = await m.get_db()[m.PREFERENCES].find_one({"user_id": user_id})
        if doc:
            prefs["default_questions"] = doc.get("default_questions")
        return prefs

    async def set_preferences(self, user_id: str, prefs: dict[str, Any]) -> dict[str, Any]:
        for kind, field in (("preferred_role", "preferred_role"), ("preferred_difficulty", "preferred_difficulty"),
                            ("learning_pref", "learning_preferences")):
            val = prefs.get(field)
            if val:
                await self.upsert(user_id, kind, "default", str(val))
        await m.get_db()[m.PREFERENCES].update_one(
            {"user_id": user_id},
            {"$set": {"default_questions": prefs.get("default_questions"), "updated_at": utcnow()},
             "$setOnInsert": {"_id": new_id(), "user_id": user_id}}, upsert=True)
        return await self.get_preferences(user_id)

    async def remember_skills(self, user_id: str, skills: list[str]) -> None:
        for s in skills[:40]:
            await self.upsert(user_id, "skill", _norm(s), s.strip(), {}, 1.5)

    async def relevant(self, user_id: str, terms: list[str], limit: int = 6) -> list[dict]:
        """Relevance filtering: kind weight + term overlap + recency; drops low scorers."""
        words = {w for t in terms for w in re.findall(r"[a-z0-9+#]+", t.lower()) if len(w) > 2}
        rows = await self.list(user_id, limit=300)
        now = utcnow()
        scored = []
        for r in rows:
            text = f"{r.get('key','')} {r.get('value','')} {(r.get('data') or {}).get('topic','')}".lower()
            overlap = sum(1 for w in words if w in text)
            ts = r.get("updated_at") or now
            age_days = max((now - ts).total_seconds() / 86400, 0) if getattr(ts, "tzinfo", None) else 0
            score = KIND_WEIGHT.get(r["kind"], 0.5) * (0.5 + overlap) * (1 / (1 + age_days / 30))
            if overlap or r["kind"] in ("topic_stat", "mistake"):
                scored.append((score, r))
        scored.sort(key=lambda x: -x[0])
        return [r for _, r in scored[:limit] if _ > 0.4]

    async def profile_snapshot(self, user_id: str) -> dict[str, Any]:
        prefs = await self.get_preferences(user_id)
        progress = await self.list(user_id, ["progress"], 50)
        recent = [p["data"]["avg_score"] for p in sorted(progress, key=lambda r: r["updated_at"])][-10:]
        return {
            "preferred_role": prefs.get("preferred_role"), "preferred_difficulty": prefs.get("preferred_difficulty"),
            "skills": [r["value"] for r in await self.list(user_id, ["skill"], 60)],
            "weak_topics": await self.weak_topics(user_id), "strong_topics": await self.strong_topics(user_id),
            "common_mistakes": await self.common_mistakes(user_id), "recent_interview_scores": recent,
        }

    async def summarize(self, user_id: str) -> str:
        snap = await self.profile_snapshot(user_id)
        lines = []
        if snap["preferred_role"]:
            lines.append(f"Target role: {snap['preferred_role']}.")
        if snap["weak_topics"]:
            lines.append("Weak topics: " + ", ".join(f"{w['topic']} ({w['avg_score']})" for w in snap["weak_topics"]) + ".")
        if snap["strong_topics"]:
            lines.append("Strong topics: " + ", ".join(s["topic"] for s in snap["strong_topics"]) + ".")
        if snap["common_mistakes"]:
            lines.append("Recurring gaps: " + "; ".join(c["point"][:70] for c in snap["common_mistakes"][:3]) + ".")
        if snap["recent_interview_scores"]:
            lines.append("Recent interview averages: " + ", ".join(map(str, snap["recent_interview_scores"][-5:])) + ".")
        return " ".join(lines) or "No long-term memory yet."

    async def prompt_note(self, user_id: str, topic: str) -> str:
        """One-line note injected into prompts (only entries relevant to this topic)."""
        rel = await self.relevant(user_id, [topic], 4)
        bits = [r["value"] for r in rel if r["kind"] in ("topic_stat", "mistake", "learning_pref")]
        return "; ".join(bits)[:400]
