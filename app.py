import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
from keras.models import load_model
import json
import random
from flask import Flask, render_template, request, jsonify

lemmatizer = WordNetLemmatizer()

# Load model + data
model = load_model('model/models.h5')
intents = json.loads(open('model/data.json').read())
words = pickle.load(open('model/texts.pkl', 'rb'))
classes = pickle.load(open('model/labels.pkl', 'rb'))


# ------------------ NLP FUNCTIONS ------------------

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)


def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25

    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)

    if not results:
        return [{"intent": "fallback", "probability": "1.0"}]

    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]


# ------------------ GET RESPONSE ------------------

def get_intent_response(ints):
    tag = ints[0]["intent"]

    for intent in intents["intents"]:
        if intent["tag"] == tag:
            response = random.choice(intent["responses"])
            url = intent.get("url", None)
            return response, url

    return "Maaf, saya tidak paham.", None


# ------------------ FLASK APP ------------------

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/get")
def get_bot_response():
    userText = request.args.get("msg")
    ints = predict_class(userText, model)

    response, url = get_intent_response(ints)

    return jsonify({
        "response": response,
        "url": url
    })


# ------------------ RUN APP ------------------

if __name__ == "__main__":
    app.run(debug=True)
