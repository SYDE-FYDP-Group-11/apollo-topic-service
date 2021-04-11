from flask import Flask, jsonify, request
import spacy
import pytextrank

app = Flask(__name__)

nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("positionrank", last=True)

@app.route('/')
def index():
    text = request.args.get('text').lower()
    n = request.args.get('n')

    words = list(get_keywords(text, int(n)))
    return jsonify({'topics': words})

def get_keywords(text, n):
    doc = nlp(text)
    i = 0
    for p in doc._.phrases:
        if len(p.text.split(' ')) <= 2:
            yield p.text
            i += 1
        if i >= n:
            break

if __name__ == '__main__':
    app.run()
