"""
models/hybrid_model.py
Hybrid Transformer + Reinforcement Learning Model
for Student Performance Analysis and Skill Suggestion
Uses: scikit-learn, numpy, transformers (HuggingFace) - No TensorFlow/Keras
"""

import numpy as np
import json
import os
import warnings
warnings.filterwarnings('ignore')

# ─── Subject Metadata ────────────────────────────────────────────────────────

SUBJECTS_3RD = [
    "python", "java", "c", "cpp", "data_structures",
    "aptitude", "os", "computer_networks", "software_engineering",
    "AIML", "dbms", "graphics_multimedia", "theory_computation"
]

SUBJECTS_4TH = [
    "python", "java", "c", "cpp", "data_structures",
    "aptitude", "os", "computer_networks", "software_engineering",
    "AIML", "neural_network", "cryptography", "mobile_application_design",
    "dbms", "graphics_multimedia", "theory_computation"
]

# GeeksForGeeks roadmap links per subject
GFG_LINKS = {
    "python":                   "https://www.geeksforgeeks.org/python-programming-language/",
    "java":                     "https://www.geeksforgeeks.org/java/",
    "c":                        "https://www.geeksforgeeks.org/c-programming-language/",
    "cpp":                      "https://www.geeksforgeeks.org/c-plus-plus/",
    "data_structures":          "https://www.geeksforgeeks.org/data-structures/",
    "aptitude":                 "https://www.geeksforgeeks.org/aptitude-questions-and-answers/",
    "os":                       "https://www.geeksforgeeks.org/operating-systems/",
    "computer_networks":        "https://www.geeksforgeeks.org/computer-network-tutorials/",
    "software_engineering":     "https://www.geeksforgeeks.org/software-engineering/",
    "AIML":                     "https://www.geeksforgeeks.org/machine-learning/",
    "neural_network":           "https://www.geeksforgeeks.org/neural-networks-a-beginners-guide/",
    "cryptography":             "https://www.geeksforgeeks.org/cryptography-and-its-types/",
    "mobile_application_design":"https://www.geeksforgeeks.org/android-app-development-fundamentals-for-beginners/",
    "dbms":                     "https://www.geeksforgeeks.org/dbms/",
    "graphics_multimedia":      "https://www.geeksforgeeks.org/computer-graphics-2/",
    "theory_computation":       "https://www.geeksforgeeks.org/theory-of-computation-automata-tutorials/",
}

# Skill clusters — which subjects relate to which career track
SKILL_CLUSTERS = {
    "Software Development":     ["python", "java", "c", "cpp", "software_engineering"],
    "Data Science / AI":        ["python", "AIML", "neural_network", "data_structures", "aptitude"],
    "Systems / Networking":     ["os", "computer_networks", "cryptography", "c", "cpp"],
    "Database & Backend":       ["dbms", "python", "java", "software_engineering"],
    "Mobile Development":       ["java", "mobile_application_design", "dbms", "software_engineering"],
    "Core CS Foundation":       ["data_structures", "theory_computation", "os", "c", "cpp"],
}

# ─── Simple Transformer-inspired Attention (no TF/PyTorch) ───────────────────

class LightweightAttention:
    """
    Lightweight self-attention mechanism implemented with NumPy.
    Mimics transformer attention for subject-score weighting.
    """

    def __init__(self, d_model=16):
        np.random.seed(42)
        self.d_model = d_model
        self.W_q = np.random.randn(d_model, d_model) * 0.1
        self.W_k = np.random.randn(d_model, d_model) * 0.1
        self.W_v = np.random.randn(d_model, d_model) * 0.1

    def _embed_scores(self, scores: np.ndarray) -> np.ndarray:
        """Embed scalar scores into d_model-dimensional vectors."""
        n = len(scores)
        embeddings = np.zeros((n, self.d_model))
        for i, s in enumerate(scores):
            # Position + value encoding
            embeddings[i] = np.sin(np.arange(self.d_model) * s / 100.0 + i)
        return embeddings

    def forward(self, scores: np.ndarray) -> np.ndarray:
        """Run attention and return attended weights per subject."""
        E = self._embed_scores(scores)
        Q = E @ self.W_q
        K = E @ self.W_k
        V = E @ self.W_v
        scale = np.sqrt(self.d_model)
        attn_weights = np.exp(Q @ K.T / scale)
        attn_weights /= (attn_weights.sum(axis=1, keepdims=True) + 1e-9)
        attended = attn_weights @ V  # (n, d_model)
        # Collapse to scalar importance per subject
        importance = np.linalg.norm(attended, axis=1)
        return importance / (importance.sum() + 1e-9)


# ─── Reinforcement Learning Policy (Q-table style) ───────────────────────────

class RLStudyAdvisor:
    """
    Simple tabular Q-learning advisor.
    States  = performance buckets (weak / average / strong) per subject
    Actions = recommend study resource / suggest practice / suggest project
    """

    ACTIONS = ["study_fundamentals", "practice_problems", "build_project", "revise_theory"]
    ACTION_LABELS = {
        "study_fundamentals": "📚 Study Fundamentals on GFG",
        "practice_problems":  "💡 Solve Practice Problems",
        "build_project":      "🛠️ Build a Mini Project",
        "revise_theory":      "📝 Revise Theory Concepts",
    }

    def __init__(self):
        # Q-table: state (0=weak,1=avg,2=strong) → action index
        self.q_table = {0: [0.9, 0.3, 0.1, 0.5],   # weak   → study_fundamentals
                        1: [0.4, 0.9, 0.5, 0.3],   # avg    → practice_problems
                        2: [0.1, 0.5, 0.9, 0.2]}   # strong → build_project

    def get_state(self, score: float) -> int:
        if score < 60:   return 0
        elif score < 80: return 1
        else:            return 2

    def recommend(self, score: float) -> dict:
        state = self.get_state(score)
        action_idx = int(np.argmax(self.q_table[state]))
        action_key = self.ACTIONS[action_idx]
        return {
            "action": action_key,
            "label":  self.ACTION_LABELS[action_key],
            "state":  ["Weak", "Average", "Strong"][state],
        }


# ─── Main Hybrid Model ────────────────────────────────────────────────────────

class HybridStudentModel:

    def __init__(self, year: int = 4):
        self.year = year
        self.subjects = SUBJECTS_4TH if year == 4 else SUBJECTS_3RD
        self.attention = LightweightAttention(d_model=16)
        self.rl_advisor = RLStudyAdvisor()

    # ── Performance Analysis ──────────────────────────────────────────────────

    def analyse_performance(self, scores: dict) -> dict:
        """
        Returns percentage, grade, rank among subjects,
        and attention-weighted importance scores.
        """
        vals = np.array([scores.get(s, 0) for s in self.subjects], dtype=float)
        importance = self.attention.forward(vals)

        subject_detail = {}
        for i, subj in enumerate(self.subjects):
            v = float(vals[i])
            subject_detail[subj] = {
                "score":      v,
                "percentage": round(v, 1),
                "grade":      self._grade(v),
                "importance": round(float(importance[i]) * 100, 2),
                "status":     self.rl_advisor.get_state(v),
            }

        overall_pct = round(float(vals.mean()), 2)
        return {
            "overall_percentage": overall_pct,
            "overall_grade":      self._grade(overall_pct),
            "subjects":           subject_detail,
            "total_subjects":     len(self.subjects),
        }

    def _grade(self, pct: float) -> str:
        if pct >= 90: return "O"
        if pct >= 80: return "A+"
        if pct >= 70: return "A"
        if pct >= 60: return "B+"
        if pct >= 50: return "B"
        return "C"

    # ── Skill Suggestions ────────────────────────────────────────────────────

    def suggest_skills(self, scores: dict) -> dict:
        """
        Uses RL advisor per subject + cluster scoring to recommend
        skills and resources.
        """
        suggestions = {}
        for subj in self.subjects:
            score = scores.get(subj, 0)
            rec = self.rl_advisor.recommend(score)
            suggestions[subj] = {
                "score":       score,
                "level":       rec["state"],
                "action":      rec["label"],
                "gfg_link":    GFG_LINKS.get(subj, "https://www.geeksforgeeks.org/"),
                "improvement": max(0, round(80 - score, 1)) if score < 80 else 0,
            }

        # Career cluster scores
        cluster_scores = {}
        for cluster, cluster_subjects in SKILL_CLUSTERS.items():
            relevant = [scores.get(s, 0) for s in cluster_subjects if s in self.subjects]
            if relevant:
                cluster_scores[cluster] = round(np.mean(relevant), 1)

        best_cluster  = max(cluster_scores, key=cluster_scores.get)
        weak_subjects = [s for s in self.subjects if scores.get(s, 0) < 60]
        avg_subjects  = [s for s in self.subjects if 60 <= scores.get(s, 0) < 80]

        return {
            "per_subject":     suggestions,
            "cluster_scores":  cluster_scores,
            "best_fit_career": best_cluster,
            "weak_subjects":   weak_subjects,
            "avg_subjects":    avg_subjects,
            "priority_focus":  weak_subjects[:3],
        }

    # ── Study Roadmap ────────────────────────────────────────────────────────

    def generate_roadmap(self, scores: dict) -> list:
        """
        Generates a prioritised week-by-week roadmap using RL priorities.
        """
        roadmap = []
        # Sort subjects by score ascending (weakest first)
        sorted_subjects = sorted(self.subjects, key=lambda s: scores.get(s, 0))

        week = 1
        for subj in sorted_subjects:
            score = scores.get(subj, 0)
            rec   = self.rl_advisor.recommend(score)
            roadmap.append({
                "week":        week,
                "subject":     subj.replace("_", " ").title(),
                "subject_key": subj,
                "score":       score,
                "level":       rec["state"],
                "task":        rec["label"],
                "gfg_link":    GFG_LINKS.get(subj, "https://www.geeksforgeeks.org/"),
                "target":      min(100, score + 15),
                "days":        5 if rec["state"] == "Weak" else (3 if rec["state"] == "Average" else 2),
            })
            week += 1

        return roadmap

    # ── Class Average (Teacher View) ─────────────────────────────────────────

    def class_average(self, all_students: list) -> dict:
        """Compute class-wide averages and identify struggling students."""
        if not all_students:
            return {}

        subj_totals = {s: [] for s in self.subjects}
        for student in all_students:
            for subj in self.subjects:
                v = student.get(subj)
                if v is not None:
                    try:
                        subj_totals[subj].append(float(v))
                    except (ValueError, TypeError):
                        pass

        averages = {}
        for subj in self.subjects:
            vals = subj_totals[subj]
            averages[subj] = {
                "mean":   round(np.mean(vals), 1) if vals else 0,
                "min":    round(np.min(vals),  1) if vals else 0,
                "max":    round(np.max(vals),  1) if vals else 0,
                "stddev": round(np.std(vals),  1) if vals else 0,
            }

        all_scores = [float(s.get(subj, 0)) for s in all_students
                      for subj in self.subjects
                      if s.get(subj) is not None]
        overall_avg = round(np.mean(all_scores), 1) if all_scores else 0

        # Students below 60% average
        at_risk = []
        for s in all_students:
            vals = [float(s.get(subj, 0)) for subj in self.subjects if s.get(subj) is not None]
            if vals and np.mean(vals) < 60:
                at_risk.append({"stuid": s.get("stuid"), "name": s.get("name"), "avg": round(np.mean(vals), 1)})

        return {
            "subject_averages": averages,
            "overall_average":  overall_avg,
            "at_risk_students": at_risk,
            "total_students":   len(all_students),
        }