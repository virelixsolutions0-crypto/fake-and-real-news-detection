import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load dataset
fake = pd.read_csv("dataset/Fake.csv")
true = pd.read_csv("dataset/True.csv")

# Labeling
fake["label"] = 0   # Fake News
true["label"] = 1   # Real News

# Merge datasets
data = pd.concat([fake, true])

# Shuffle data
data = data.sample(frac=1, random_state=42)

# Features & Labels
X = data["text"]
y = data["label"]

# Convert text into numbers
vectorizer = TfidfVectorizer(stop_words="english", max_features=10000)
X = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Accuracy check
accuracy = model.score(X_test, y_test)
print("Model Accuracy:", accuracy)

# Save model
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Training Completed Successfully ✔")