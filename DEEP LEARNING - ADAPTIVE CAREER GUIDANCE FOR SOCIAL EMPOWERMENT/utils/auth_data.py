"""
utils/auth_data.py
Authentication and CSV data loading utilities.
"""

import csv
import os
from pathlib import Path

BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = BASE_DIR / "data"
CSV_3RD    = DATA_DIR / "3rdyearstudentsdata.csv"
CSV_4TH    = DATA_DIR / "4thyearstudentsdata.csv"

TEACHERS = {
    "3RDTEACHER": {"password": "teachpass", "year": 3, "name": "3rd Year Class Teacher"},
    "4THTEACHER": {"password": "teachpass", "year": 4, "name": "4th Year Class Teacher"},
}

SUBJECTS_3RD = [
    "python","java","c","cpp","data_structures","aptitude","os",
    "computer_networks","software_engineering","AIML",
    "dbms","graphics_multimedia","theory_computation"
]

SUBJECTS_4TH = [
    "python","java","c","cpp","data_structures","aptitude","os",
    "computer_networks","software_engineering","AIML",
    "neural_network","cryptography","mobile_application_design",
    "dbms","graphics_multimedia","theory_computation"
]


def _load_csv(path: Path) -> list:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def authenticate_student(stuid: str, password: str):
    """Returns (student_dict, year) or (None, None)."""
    for year, path in [(3, CSV_3RD), (4, CSV_4TH)]:
        for row in _load_csv(path):
            if row.get("stuid","").strip() == stuid.strip() \
               and row.get("password","").strip() == password.strip() \
               and stuid not in TEACHERS:
                return row, year
    return None, None


def authenticate_teacher(stuid: str, password: str):
    """Returns teacher_info dict or None."""
    info = TEACHERS.get(stuid.upper())
    if info and info["password"] == password:
        return info
    return None


def get_all_students(year: int) -> list:
    path = CSV_3RD if year == 3 else CSV_4TH
    rows = _load_csv(path)
    teacher_ids = set(TEACHERS.keys())
    return [r for r in rows if r.get("stuid","").upper() not in teacher_ids]


def get_subjects(year: int) -> list:
    return SUBJECTS_3RD if year == 3 else SUBJECTS_4TH


def student_scores(student_row: dict, year: int) -> dict:
    subjects = get_subjects(year)
    scores = {}
    for s in subjects:
        try:
            scores[s] = float(student_row.get(s, 0) or 0)
        except (ValueError, TypeError):
            scores[s] = 0.0
    return scores