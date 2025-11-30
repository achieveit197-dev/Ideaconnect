import os
import io
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from dotenv import load_dotenv
from openai import OpenAI
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

IDEAS = []


# -----------------------------
# Normalization Helpers
# -----------------------------
def normalize_tech_stack(value):
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    return []


def find_idea_index(idea_id):
    for idx, idea in enumerate(IDEAS):
        if idea["id"] == idea_id:
            return idx
    return None


# -----------------------------
# Score Helpers
# -----------------------------
def compute_skill_match(idea_tech_stack, builder_tech_stack):
    idea_set = {t.lower() for t in normalize_tech_stack(idea_tech_stack)}
    builder_set = {t.lower() for t in normalize_tech_stack(builder_tech_stack)}

    if not idea_set:
        return 0.0

    match = idea_set.intersection(builder_set)
    if not match:
        return 0.0

    return len(match) / len(idea_set)


def compute_answer_quality(answers):
    if not answers:
        return 0.0

    bad_tokens = [
        "i dont know", "i don't know", "idk", "not sure",
        "no idea", "na", "n/a", "nothing", "nope"
    ]

    scores = []
    for ans in answers:
        txt = str(ans).strip().lower()

        if any(tok in txt for tok in bad_tokens):
            scores.append(0.0)
            continue

        if len(txt) < 20:
            scores.append(0.2)
        elif len(txt) < 50:
            scores.append(0.5)
        else:
            scores.append(0.9)

    return round(sum(scores) / len(scores), 2)


def compute_final_readiness(skill_match, answer_quality):
    base = (skill_match * 0.55) + (answer_quality * 0.45)
    score = base * 10

    if skill_match == 0:
        return round(min(score, 2.5), 1)

    if answer_quality < 0.25:
        return round(min(score, 3.0), 1)

    return round(score, 1)


# -----------------------------
# AI Personality Only (No Scoring)
# -----------------------------
def get_ai_personality(idea, builder, questions, answers):
    prompt = f"""
Profile this builder. DO NOT generate any numeric score.

Return JSON ONLY:
{{
  "personality": "<single word>",
  "team_fit": "<single sentence>",
  "summary": "<2–3 lines>"
}}

Idea: {idea}
Builder: {builder}
QnA: {[
    {"q": questions[i], "a": answers[i]}
    for i in range(len(answers))
]}
"""

    res = client.responses.create(
        model="gpt-4.1-mini",
        response_format={"type": "json_object"},
        input=prompt
    )
    return res.output[0]["json"]


# -----------------------------
# Generate Screening Questions
# -----------------------------
def generate_questions(idea):
    prompt = f"""
Generate EXACTLY 5 builder screening questions.
Return JSON ONLY:
{{ "questions": [] }}
Idea: {idea}
"""
    res = client.responses.create(
        model="gpt-4.1-mini",
        response_format={"type": "json_object"},
        input=prompt
    )
    return res.output[0]["json"]["questions"]


# -----------------------------
# Evaluate Builder (STRONG fix)
# -----------------------------
def evaluate_builder(idea, builder, questions, answers):
    skill_match = compute_skill_match(idea["tech_stack"], builder["tech_stack"])
    answer_quality = compute_answer_quality(answers)
    personality = get_ai_personality(idea, builder, questions, answers)
    readiness = compute_final_readiness(skill_match, answer_quality)

    return {
        "readiness_score": readiness,
        "skill_match_ratio": skill_match,
        "answer_quality": answer_quality,
        "personality": personality.get("personality", "unknown"),
        "team_fit": personality.get("team_fit", ""),
        "summary": personality.get("summary", "")
    }


# -----------------------------
# API ROUTES
# -----------------------------
@app.route("/api/idea/post", methods=["POST"])
def post_idea():
    data = request.get_json()
    required_fields = [
        "idea_title", "problem_statement", "solution_summary",
        "tech_stack", "team_requirements", "engagement_type",
        "required_team_size"
    ]

    missing = [f for f in required_fields if f not in data]
    if missing:
        return jsonify({"error": "Missing fields", "missing": missing}), 400

    required_team = int(data["required_team_size"])
    total_team = sum(int(i.get("count", 0)) for i in data["team_requirements"])
    if total_team != required_team:
        return jsonify({"error": "Team size mismatch"}), 400

    idea = {
        "id": len(IDEAS) + 1,
        "idea_title": data["idea_title"],
        "problem_statement": data["problem_statement"],
        "solution_summary": data["solution_summary"],
        "tech_stack": normalize_tech_stack(data["tech_stack"]),
        "team_requirements": data["team_requirements"],
        "required_team_size": required_team,
        "engagement_type": data["engagement_type"],
        "notes": data.get("notes", ""),
        "generated_questions": [],
        "interests": []
    }

    IDEAS.append(idea)
    return jsonify({"message": "Idea created", "idea": idea}), 201


@app.route("/api/idea/<int:idea_id>/generate-questions", methods=["POST"])
def gen_q(idea_id):
    idx = find_idea_index(idea_id)
    if idx is None:
        return jsonify({"error": "Idea not found"}), 404

    q = generate_questions(IDEAS[idx])
    IDEAS[idx]["generated_questions"] = q
    return jsonify({"questions": q})


@app.route("/api/idea/<int:idea_id>/interest", methods=["POST"])
def submit_interest(idea_id):
    idx = find_idea_index(idea_id)
    if idx is None:
        return jsonify({"error": "Idea not found"}), 404

    data = request.get_json()
    required = ["name", "contact", "email", "tech_stack", "years_of_experience", "answers"]

    missing = [f for f in required if f not in data]
    if missing:
        return jsonify({"error": "Missing fields", "missing": missing}), 400

    idea = IDEAS[idx]

    questions = idea["generated_questions"]
    answers = data["answers"]
    if len(questions) != len(answers):
        return jsonify({"error": "Answer count mismatch"}), 400

    builder = {
        "name": data["name"],
        "contact": data["contact"],
        "email": data["email"],
        "tech_stack": normalize_tech_stack(data["tech_stack"]),
        "years_of_experience": data["years_of_experience"],
        "comments": data.get("comments", "")
    }

    evaluation = evaluate_builder(idea, builder, questions, answers)

    interest = {
        "interest_id": len(idea["interests"]) + 1,
        **builder,
        "answers": answers,
        "evaluation": evaluation
    }

    idea["interests"].append(interest)

    # ⭐ CRITICAL FIX ⭐
    # Return FLAT fields Lovable can read directly
    return jsonify({
        "message": "Interest recorded",
        "interest": interest,
        "readiness_score": evaluation["readiness_score"],
        "personality": evaluation["personality"],
        "team_fit": evaluation["team_fit"],
        "summary": evaluation["summary"],
        "skill_match_ratio": evaluation["skill_match_ratio"],
        "answer_quality": evaluation["answer_quality"]
    }), 201


# -----------------------------
# Run App
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
