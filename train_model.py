import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ------------------ SAMPLE TRAINING DATA ------------------

texts = [
    "Government announces new education policy after cabinet meeting",
    "According to officials, the court delivered its verdict today",
    "Breaking shocking truth government secretly bans currency",
    "Unbelievable news you won’t believe what politicians did",
    "Sources said the minister announced reforms in parliament",
    "This fake story claims aliens landed in the capital city"
]

labels = [
    0,  # REAL
    0,  # REAL
    1,  # FAKE
    1,  # FAKE
    0,  # REAL
    1   # FAKE
]

# ------------------ TRAIN MODEL ------------------

vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(texts)

model = LogisticRegression()
model.fit(X, labels)

# ------------------ SAVE FILES ------------------

joblib.dump(model, "model.joblib")
joblib.dump(vectorizer, "vectorizer.joblib")

print("Model and vectorizer saved successfully")