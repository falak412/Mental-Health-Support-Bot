from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

model = pickle.load(open("model/emotion_model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

emotion_map = {
    0: "sadness 😢",
    1: "joy 😊",
    2: "love ❤️",
    3: "anger 😠",
    4: "fear 😨",
    5: "surprise 😲"
}

support_links = {
    "sadness": "Take deep breaths and reach out to someone you trust. 💬",
    "joy": "That's wonderful! Keep doing what brings you happiness. 🌟",
    "love": "Cherish this feeling! Spread it to those around you. 💖",
    "anger": "Try journaling or going for a walk to cool off. 🧘",
    "fear": "You're safe. Talk to someone or practice grounding techniques. 🛡️",
    "surprise": "Embrace the unexpected! Explore what this means for you. 🤔"
}

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    vector = vectorizer.transform([text])
    prediction = model.predict(vector)[0]
    emotion = emotion_map[prediction]
    support = support_links[emotion.split()[0]]
    return render_template("result.html", emotion=emotion, support=support)

if __name__ == "__main__":
    app.run(debug=True)
