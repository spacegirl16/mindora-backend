from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_bcrypt import Bcrypt
from flask_jwt_extended import (
    JWTManager,
    create_access_token,
    jwt_required,
    get_jwt_identity
)
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta
from transformers import pipeline
import torch

# -------------------- APP SETUP --------------------

app = Flask(__name__)
CORS(app)

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///mental_wellness.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["JWT_SECRET_KEY"] = "super-secret-key"

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
jwt = JWTManager(app)

# -------------------- LOAD AI MODEL --------------------

print("Loading sentiment model...")
sentiment_model = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)
print("Model loaded.")

# -------------------- MODELS --------------------

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

class MoodEntry(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.Text, nullable=False)
    sentiment = db.Column(db.String(50))
    confidence = db.Column(db.Float)
    emotion = db.Column(db.String(50))
    risk_flag = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"))

# -------------------- AUTH --------------------

@app.route("/register", methods=["POST"])
def register():
    data = request.json
    hashed = bcrypt.generate_password_hash(data["password"]).decode("utf-8")

    user = User(username=data["username"], password=hashed)
    db.session.add(user)
    db.session.commit()

    return jsonify({"message": "User registered successfully"})

@app.route("/login", methods=["POST"])
def login():
    data = request.json
    user = User.query.filter_by(username=data["username"]).first()

    if user and bcrypt.check_password_hash(user.password, data["password"]):
        token = create_access_token(identity=str(user.id))
        return jsonify({"access_token": token})

    return jsonify({"error": "Invalid credentials"}), 401

# -------------------- HELPER FUNCTIONS --------------------

def detect_emotion(text):
    text = text.lower()

    if any(word in text for word in ["sad", "depressed", "lonely", "cry"]):
        return "Sadness"
    if any(word in text for word in ["angry", "frustrated", "annoyed"]):
        return "Anger"
    if any(word in text for word in ["anxious", "worried", "scared"]):
        return "Anxiety"
    if any(word in text for word in ["happy", "excited", "grateful"]):
        return "Happiness"
    
    return "Neutral"

def detect_risk(text):
    text = text.lower()
    risk_words = ["suicide", "kill myself", "end my life", "self harm"]
    return any(word in text for word in risk_words)

# -------------------- PREDICT --------------------

@app.route("/predict", methods=["POST"])
@jwt_required()
def predict():

    user_id = get_jwt_identity()
    data = request.json
    text = data["text"]

    result = sentiment_model(text)[0]

    sentiment = result["label"]
    confidence = float(result["score"])

    emotion = detect_emotion(text)
    risk_flag = detect_risk(text)

    entry = MoodEntry(
        text=text,
        sentiment=sentiment,
        confidence=confidence,
        emotion=emotion,
        risk_flag=risk_flag,
        user_id=user_id
    )

    db.session.add(entry)
    db.session.commit()

    return jsonify({
        "sentiment": sentiment,
        "confidence": round(confidence * 100, 2),
        "emotion": emotion,
        "risk_detected": risk_flag
    })

# -------------------- WEEKLY SUMMARY --------------------

@app.route("/weekly-summary", methods=["GET"])
@jwt_required()
def weekly_summary():

    user_id = get_jwt_identity()
    week_ago = datetime.utcnow() - timedelta(days=7)

    entries = MoodEntry.query.filter(
        MoodEntry.user_id == user_id,
        MoodEntry.created_at >= week_ago
    ).all()

    total = len(entries)

    if total == 0:
        return jsonify({"message": "No entries this week"})

    positive = len([e for e in entries if e.sentiment == "POSITIVE"])
    negative = len([e for e in entries if e.sentiment == "NEGATIVE"])
    risks = len([e for e in entries if e.risk_flag])

    mood_score = int((positive / total) * 100)

    summary_text = (
        f"This week you logged {total} entries. "
        f"{positive} positive and {negative} negative moods. "
        f"Overall mood health score is {mood_score}%. "
    )

    if risks > 0:
        summary_text += "Some entries showed emotional distress. Consider talking to someone you trust."

    return jsonify({
        "total_entries": total,
        "positive": positive,
        "negative": negative,
        "risk_flags": risks,
        "mood_score": mood_score,
        "ai_summary": summary_text
    })

# -------------------- RUN SERVER --------------------

if __name__ == "__main__":
    with app.app_context():
        db.create_all()

    app.run(host="0.0.0.0", port=5000, debug=True)