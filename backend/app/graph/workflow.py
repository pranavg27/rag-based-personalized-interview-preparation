"""Compile the interview StateGraph."""
from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from app.graph import edges
from app.graph.nodes import GraphDeps, InterviewNodes
from app.graph.state import InterviewState


def build_workflow(deps: GraphDeps, checkpointer: Any):
    n = InterviewNodes(deps)
    g = StateGraph(InterviewState)
    for name in ("analyze_resume", "extract_skills", "review_skills", "plan_interview", "approve_plan",
                 "retrieve_knowledge", "generate_question", "interviewer", "record_skip", "evaluate_answer",
                 "feedback", "review_feedback", "update_performance", "adjust_difficulty", "summary"):
        g.add_node(name, getattr(n, name))

    g.add_edge(START, "analyze_resume")
    g.add_edge("analyze_resume", "extract_skills")
    g.add_conditional_edges("extract_skills", edges.after_extract, ["review_skills", "plan_interview"])
    g.add_conditional_edges("review_skills", edges.after_skill_review, ["plan_interview", "summary"])
    g.add_conditional_edges("plan_interview", edges.after_plan, ["approve_plan", "retrieve_knowledge"])
    g.add_conditional_edges("approve_plan", edges.after_plan_approval,
                            ["plan_interview", "retrieve_knowledge", "summary"])
    g.add_edge("retrieve_knowledge", "generate_question")
    g.add_edge("generate_question", "interviewer")
    g.add_conditional_edges("interviewer", edges.after_interviewer, ["evaluate_answer", "record_skip", "summary"])
    g.add_edge("evaluate_answer", "feedback")
    g.add_conditional_edges("feedback", edges.after_feedback, ["review_feedback", "update_performance"])
    g.add_conditional_edges("review_feedback", edges.after_feedback_review,
                            ["evaluate_answer", "update_performance", "summary"])
    g.add_edge("update_performance", "adjust_difficulty")
    g.add_conditional_edges("adjust_difficulty", edges.after_progress, ["retrieve_knowledge", "summary"])
    g.add_conditional_edges("record_skip", edges.after_progress, ["retrieve_knowledge", "summary"])
    g.add_edge("summary", END)
    return g.compile(checkpointer=checkpointer)
