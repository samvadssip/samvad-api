from flask import Flask, request, jsonify
from functions import Predictor

app = Flask(__name__)

pred = Predictor('model.h5')

@app.route('/')
def home():
    return "hello world"


@app.route('/translate', methods=['POST'])
def translate():
    video_path = request.form.get('video') #video path
    # result = {'pictures': pictures}

    text = pred.predict(video_path)

    result = {
        'word': text
    }

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)