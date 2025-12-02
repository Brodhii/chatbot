import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
from keras.models import load_model
import json
import random
from flask import Flask, render_template, request, jsonify

# Spell correction
from spellchecker import SpellChecker
spell = SpellChecker()

# Fuzzy matching untuk patterns
from thefuzz import fuzz, process

lemmatizer = WordNetLemmatizer()

# Load model + data (sesuaikan path jika beda)
model = load_model('model/models.h5')
intents = json.loads(open('model/data.json').read())
words = pickle.load(open('model/texts.pkl', 'rb'))
classes = pickle.load(open('model/labels.pkl', 'rb'))

# ------------------ FUZZY MATCH PATTERNS (SAFE) ------------------
def fuzzy_match_patterns(user_input, threshold=70):
    """
    Cocokkan keseluruhan input user dengan semua pattern intents secara fuzzy.
    Kembalikan pattern terbaik jika score >= threshold, else None.
    Aman terhadap None/empty patterns.
    """
    try:
        # kumpulkan semua pattern valid
        all_patterns = []
        pattern_to_intent = {}
        for intent in intents.get("intents", []):
            for p in intent.get("patterns", []):
                if p and isinstance(p, str) and p.strip():
                    p_norm = p.strip()
                    all_patterns.append(p_norm)
                    pattern_to_intent[p_norm] = intent  # map pattern -> intent

        if not all_patterns:
            return None, None

        result = process.extractOne(user_input, all_patterns, scorer=fuzz.ratio)
        if not result:
            return None, None

        best_match, score = result
        if score >= threshold:
            return best_match, pattern_to_intent.get(best_match)
        return None, None

    except Exception as e:
        print("Fuzzy pattern error:", e)
        return None, None

# ------------------ CLEAN + WORD-LEVEL CORRECTION (keamanan) ------------------
def clean_up_sentence(sentence):
    if not sentence:
        return []

    # tokenisasi sederhana
    try:
        sentence_words = nltk.word_tokenize(sentence)
    except Exception:
        # fallback split sederhana jika tokenizer belum tersedia
        sentence_words = sentence.split()

    corrected_words = []
    for word in sentence_words:
        try:
            corrected = spell.correction(word)
            if corrected is None:
                corrected = word
        except Exception:
            corrected = word

        # lemmatize dan lower
        try:
            corrected = lemmatizer.lemmatize(corrected.lower())
        except Exception:
            corrected = str(corrected).lower()

        corrected_words.append(corrected)

    return corrected_words

# ------------------ BOW ------------------
def bow(sentence, words_list):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words_list)
    for s in sentence_words:
        for i, w in enumerate(words_list):
            if w == s:
                bag[i] = 1
    return np.array(bag)

# ------------------ PREDICT ------------------
def predict_class(sentence, model):
    p = bow(sentence, words)
    try:
        res = model.predict(np.array([p]))[0]
    except Exception as e:
        print("Model prediction error:", e)
        return [{"intent": "fallback", "probability": "1.0"}]

    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)

    if not results:
        return [{"intent": "fallback", "probability": "1.0"}]
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]

# ------------------ GET RESPONSE ------------------
def get_intent_response(ints):
    tag = ints[0]["intent"]
    for intent in intents.get("intents", []):
        if intent.get("tag") == tag:
            response = random.choice(intent.get("responses", ["Maaf, saya tidak mengerti."]))
            url = intent.get("url", None)
            return response, url
    return "Maaf, saya tidak mengerti.", None

# ------------------ FLASK APP ------------------
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get("msg", "")
    userText = userText.strip()
    if not userText:
        return jsonify({"response": "Tolong masukkan pesan.", "url": None})

    # 1) FUZZY MATCH PADA PATTERN (PRIORITAS)
    matched_pattern, matched_intent = fuzzy_match_patterns(userText.lower(), threshold=70)
    if matched_pattern and matched_intent:
        # kembalikan langsung response + url dari intent yang cocok
        resp = random.choice(matched_intent.get("responses", ["Maaf, saya tidak mengerti."]))
        return jsonify({"response": resp, "url": matched_intent.get("url", None)})

    # 2) Jika fuzzy pattern tidak cocok → gunakan model
    try:
        ints = predict_class(userText, model)
        response, url = get_intent_response(ints)
        return jsonify({"response": response, "url": url})
    except Exception as e:
        print("Runtime error:", e)
        return jsonify({"response": "Terjadi kesalahan pada sistem.", "url": None})

if __name__ == "__main__":
    app.run(debug=True)
