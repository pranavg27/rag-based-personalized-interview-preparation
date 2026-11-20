from app.config import get_settings
from app.services.personalization_service import PersonalizationService

P = PersonalizationService(get_settings())


def test_difficulty_up_down_and_floor():
    assert P.adjust_difficulty("medium", [9, 9, 8.5])[0] == "hard"
    assert P.adjust_difficulty("hard", [9, 9, 9])[0] == "hard"
    new, _, extra = P.adjust_difficulty("medium", [3, 4, 2])
    assert new == "easy" and extra
    new, _, extra = P.adjust_difficulty("easy", [2, 3])
    assert new == "easy" and extra
    assert P.adjust_difficulty("medium", [9])[0] == "medium"  # not enough data


def test_plan_prioritises_weak_topics():
    plan = P.build_plan(mode="skill_based", role=None, requested_skills=["Python", "DBMS", "DSA"], resume_skills=[],
                        jd_skills=[], weak_topics=["DBMS"], num_questions=6)
    topics = [p["topic"] for p in plan]
    assert topics.count("DBMS") > topics.count("DSA")
    assert [p["index"] for p in plan] == list(range(6))


def test_weak_mode_and_mock_structure():
    plan = P.build_plan(mode="weak_topics", role="Python Developer", requested_skills=[], resume_skills=[],
                        jd_skills=[], weak_topics=["Normalization"], num_questions=3)
    assert {p["topic"] for p in plan} == {"Normalization"}
    mock = P.build_plan(mode="mock_interview", role="Backend Developer", requested_skills=["DBMS", "DSA"],
                        resume_skills=[], jd_skills=[], weak_topics=[], num_questions=6)
    types = [p["type"] for p in mock]
    assert types[-1] == "behavioral" and "followup" in types
    diffs = [p["difficulty"] for p in mock if p["type"] == "technical"]
    assert diffs[0] == "easy" and diffs[-1] == "hard"
