from flask import Flask, jsonify, request
import pickle

app = Flask(__name__)

@app.route('/')
def index():
    text = request.args.get('text')
    n = request.args.get('n')

    words, scores = model.predict(text, n=int(n))
    return jsonify({'topics': words})

with open('model.pkl', 'rb') as model_file:
  model = pickle.load(model_file)

if __name__ == '__main__':
    app.run()
