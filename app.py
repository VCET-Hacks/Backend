from flask import Flask, request, jsonify
from transformers import pipeline


pipe = pipeline("text-classification", model="dk3156/toxic_tweets_model")


app = Flask(__name__)


@app.route('/classify', methods=['POST'])
def classify_text():
    data = request.get_json()
    sentences = data.get('sentences', [])
    
    if not sentences:
        return jsonify({"error": "No sentences provided"}), 400
    if isinstance(sentences, str):
        sentences = [sentences]
    results = pipe(sentences)
    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True)
