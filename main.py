from flask import Flask, jsonify, request
import pickle

app = Flask(__name__)

@app.route('/')
def index():
    text = request.args.get('text')
    predictions = model.predict(text)
    return jsonify({'predictions': predictions})

with open('model.pkl', 'rb') as model_file:
  model = pickle.load(model_file)

if __name__ == '__main__':
    app.run()
