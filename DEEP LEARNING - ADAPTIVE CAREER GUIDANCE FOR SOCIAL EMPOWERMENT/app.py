"""
app.py  — Student Learning Portal
Flask backend for the AI-powered student dashboard.

Run:
    pip install flask
    python app.py
"""

from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import json, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from utils.auth_data   import (authenticate_student, authenticate_teacher,
                                get_all_students, student_scores, get_subjects)
from models.hybrid_model import HybridStudentModel

app = Flask(__name__)
app.secret_key = "studentportal_secret_2024"

# ── Cache placement data so we don't hit sites every request ──────────────────
_placement_cache = {"data": None, "loaded": False}

def _get_placement_data(force_refresh=False):
    global _placement_cache
    if not _placement_cache["loaded"] or force_refresh:
        try:
            from selenium_tools.placement_scraper import get_placement_data
            _placement_cache["data"]   = get_placement_data()
            _placement_cache["loaded"] = True
        except Exception as e:
            _placement_cache["data"]   = {"error": str(e), "companies": [], "naukri_jobs": [], "total_open": 0}
            _placement_cache["loaded"] = True
    return _placement_cache["data"]


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        stuid    = request.form.get("stuid", "").strip()
        password = request.form.get("password", "").strip()

        # Teacher login
        teacher = authenticate_teacher(stuid, password)
        if teacher:
            session["role"]     = "teacher"
            session["stuid"]    = stuid
            session["year"]     = teacher["year"]
            session["name"]     = teacher["name"]
            return redirect(url_for("teacher_dashboard"))

        # Student login
        student, year = authenticate_student(stuid, password)
        if student:
            session["role"]    = "student"
            session["stuid"]   = stuid
            session["year"]    = year
            session["name"]    = student.get("name", stuid)
            return redirect(url_for("student_dashboard"))

        error = "Invalid Student ID or Password."

    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ── Student Dashboard ─────────────────────────────────────────────────────────

@app.route("/student")
def student_dashboard():
    if session.get("role") != "student":
        return redirect(url_for("login"))

    stuid = session["stuid"]
    year  = session["year"]

    students = get_all_students(year)
    student  = next((s for s in students if s["stuid"] == stuid), None)
    if not student:
        return redirect(url_for("login"))

    scores = student_scores(student, year)
    model  = HybridStudentModel(year=year)

    performance = model.analyse_performance(scores)
    skills      = model.suggest_skills(scores)
    roadmap     = model.generate_roadmap(scores)
    placements  = _get_placement_data() if year == 4 else None

    return render_template(
        "student_dashboard.html",
        student     = student,
        year        = year,
        performance = performance,
        skills      = skills,
        roadmap     = roadmap,
        placements  = placements,
        subjects    = get_subjects(year),
        scores_json = json.dumps(scores),
    )


# ── Teacher Dashboard ─────────────────────────────────────────────────────────

@app.route("/teacher")
def teacher_dashboard():
    if session.get("role") != "teacher":
        return redirect(url_for("login"))

    year     = session["year"]
    students = get_all_students(year)
    model    = HybridStudentModel(year=year)

    # Build per-student summary
    student_summaries = []
    for s in students:
        sc  = student_scores(s, year)
        pf  = model.analyse_performance(sc)
        student_summaries.append({
            "stuid":      s["stuid"],
            "name":       s.get("name", s["stuid"]),
            "overall":    pf["overall_percentage"],
            "grade":      pf["overall_grade"],
            "scores":     sc,
        })

    class_avg = model.class_average(students)

    return render_template(
        "teacher_dashboard.html",
        year             = year,
        teacher_name     = session.get("name", "Teacher"),
        students         = student_summaries,
        class_avg        = class_avg,
        subjects         = get_subjects(year),
        class_avg_json   = json.dumps(class_avg),
    )


# ── API: Naukri IT Jobs ───────────────────────────────────────────────────────

@app.route("/api/naukri-jobs")
def api_naukri_jobs():
    from selenium_tools.placement_scraper import get_naukri_it_jobs
    return jsonify(get_naukri_it_jobs())


# ── API: Refresh Placements ───────────────────────────────────────────────────

@app.route("/api/placements/refresh")
def api_refresh_placements():
    if session.get("role") != "student" or session.get("year") != 4:
        return jsonify({"error": "Unauthorized"}), 403
    data = _get_placement_data(force_refresh=True)
    return jsonify(data)


if __name__ == "__main__":
    app.run(debug=True, port=5000)