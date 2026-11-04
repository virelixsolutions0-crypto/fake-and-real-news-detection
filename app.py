from flask import Flask, render_template, request, jsonify
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():

    data = request.get_json()
    news = data['news_text']

    vector = vectorizer.transform([news])

    prediction = model.predict(vector)[0]

    confidence = max(model.predict_proba(vector)[0]) * 100
    confidence = round(confidence, 2)

    result = "REAL" if prediction == 1 else "FAKE"

    return jsonify({
        "prediction": result,
        "confidence": confidence,
        "word_count": len(news.split()),
        "char_count": len(news)
    })

if __name__ == "__main__":
    app.run(debug=True)