import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# Simple training dataset
data = {
    "text": [
        "I feel happy today",
        "I am very stressed and tired",
        "Life is beautiful",
        "I feel anxious about exams",
        "I am calm and relaxed",
        "I feel depressed and lonely",
        "I feel excited and motivated",
        "I am overwhelmed with work"
    ],
    "label": [1, 0, 1, 0, 1, 0, 1, 0]  # 1 = Positive, 0 = Negative
}

df = pd.DataFrame(data)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["text"])
y = df["label"]

model = LogisticRegression()
model.fit(X, y)

# Save model and vectorizer
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model trained and saved successfully!")