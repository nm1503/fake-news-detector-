from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import re
from urllib.parse import urlparse
import numpy as np

# ------------------ APP ------------------

app = FastAPI(title="Fake News Detection API")

# ------------------ CORS ------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ LOAD MODEL ------------------

MODEL_PATH = "model.joblib"
VECTORIZER_PATH = "vectorizer.joblib"

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

# ------------------ CONFIG ------------------

KNOWN_SOURCES = {
    "bbc.com": 0.95,
    "reuters.com": 0.98,
    "theguardian.com": 0.94,
    "cnn.com": 0.92,
    "nytimes.com": 0.96,
    "thehindu.com": 0.90,
    "indianexpress.com": 0.88
}

FAKE_THRESHOLD = 0.70
REAL_THRESHOLD = 0.40

# ------------------ REQUEST SCHEMA ------------------

class NewsInput(BaseModel):
    text: str
    source_url: str | None = ""

# ------------------ RELAXED VALIDATION ------------------

def is_news_like(text: str) -> bool:
    words = text.strip().split()
    if len(words) < 20:
        return False

    sentences = re.split(r"[.!?]", text)
    sentences = [s for s in sentences if s.strip()]
    return len(sentences) >= 2

# ------------------ SOURCE SCORING ------------------

def get_domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""

def source_score(url: str) -> float:
    if not url:
        return 0.55

    domain = get_domain(url)
    for known, score in KNOWN_SOURCES.items():
        if known in domain:
            return score

    if domain.endswith(".gov"):
        return 0.97
    if domain.endswith(".edu"):
        return 0.90

    return 0.55

# ------------------ CLICKBAIT ------------------

def clickbait_score(text: str) -> float:
    triggers = ["shocking", "breaking", "exposed", "unbelievable"]
    score = sum(t in text.lower() for t in triggers)

    caps_ratio = sum(c.isupper() for c in text) / max(len(text), 1)
    exclamations = text.count("!")

    return min(1.0, score * 0.2 + caps_ratio + exclamations * 0.05)

# ------------------ ML PROBABILITY HANDLER ------------------

def get_ml_probability(X):
    """
    Supports both LogisticRegression and LinearSVC
    """
    # LogisticRegression
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[0][1]

    # LinearSVC → convert decision score to pseudo-probability
    if hasattr(model, "decision_function"):
        decision = model.decision_function(X)[0]
        probability = 1 / (1 + np.exp(-decision))  # sigmoid
        return float(probability)

    # Fallback
    return 0.5

# ------------------ API ENDPOINT ------------------

@app.post("/detect")
def detect_news(data: NewsInput):

    if not is_news_like(data.text):
        return {
            "status": "invalid_input",
            "message": "Please provide a proper news article."
        }

    X = vectorizer.transform([data.text])
    ml_prob = get_ml_probability(X)

    clickbait = clickbait_score(data.text)
    source = source_score(data.source_url)

    final_score = (
        0.6 * ml_prob +
        0.25 * clickbait +
        0.15 * (1 - source)
    )

    if final_score >= FAKE_THRESHOLD:
        verdict = "FAKE"
    elif final_score <= REAL_THRESHOLD:
        verdict = "REAL"
    else:
        verdict = "UNCERTAIN"

    return {
        "status": "success",
        "verdict": verdict,
        "confidence": round(final_score, 2),
        "explanation": {
            "ml_probability": round(ml_prob, 2),
            "clickbait_score": round(clickbait, 2),
            "source_credibility": round(source, 2)
        }
    }
