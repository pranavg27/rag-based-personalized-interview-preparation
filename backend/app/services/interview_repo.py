"""Persistence helpers used by the graph nodes and the interview service."""
from __future__ import annotations

from typing import Any

from app.database import mongo as m
from app.utils.ids import new_id, utcnow


class InterviewRepo:
    @property
    def interviews(self):
        return m.get_db()[m.INTERVIEWS]

    @property
    def answers(self):
        return m.get_db()[m.ANSWERS]

    @property
    def analytics(self):
        return m.get_db()[m.ANALYTICS]

    async def save_answer(self, *, interview_id: str, user_id: str, index: int, question: dict, answer: str,
                          skipped: bool, evaluation: dict | None, feedback_text: str, decision: str) -> None:
        await self.answers.update_one(
            {"interview_id": interview_id, "user_id": user_id, "index": index},
            {"$set": {"question_id": question.get("id"), "topic": question.get("topic"),
                      "difficulty": question.get("difficulty"), "type": question.get("type"),
                      "question": question.get("text"), "answer": answer, "skipped": skipped,
                      "evaluation": evaluation, "score": None if skipped or not evaluation else evaluation.get("score"),
                      "feedback_text": feedback_text, "feedback_decision": decision, "created_at": utcnow()},
             "$setOnInsert": {"_id": new_id()}},
            upsert=True)

    async def list_answers(self, user_id: str, interview_id: str) -> list[dict]:
        return await self.answers.find({"interview_id": interview_id, "user_id": user_id}).sort("index", 1).to_list(200)

    async def asked_questions(self, user_id: str, topic: str, limit: int) -> list[dict]:
        rows = await self.answers.find({"user_id": user_id, "topic": topic}).sort("created_at", -1).to_list(limit)
        return [{"question": r["question"], "difficulty": r.get("difficulty"), "score": r.get("score")} for r in rows]

    async def save_summary(self, *, interview_id: str, user_id: str, summary: dict, role: str | None) -> None:
        await self.analytics.update_one(
            {"interview_id": interview_id},
            {"$set": {"user_id": user_id, "role": role, "avg_score": summary.get("avg_score"),
                      "topic_scores": summary.get("topic_scores", {}), "answered": summary.get("answered", 0),
                      "difficulty_history": summary.get("difficulty_history", []), "created_at": utcnow()},
             "$setOnInsert": {"_id": new_id()}}, upsert=True)
        await self.interviews.update_one({"_id": interview_id, "user_id": user_id},
                                         {"$set": {"summary": summary, "avg_score": summary.get("avg_score")}})
