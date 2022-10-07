from flask import Flask, request, jsonify
import requests, json
from functions import Predictor

app = Flask(__name__)

# pred = Predictor('model.h5')

@app.route('/')
def home():
    return "hello world"


# @app.route('/translate', methods=['POST', 'GET'])
# def translate():
#     # video_path = request.form.get('video') #video path
#     video_path = "VN20221006_123507.mp4"
#     # result = {'pictures': pictures}
#
#     # res = json.loads(requests.get(video_path).content.decode("utf-8"))
#     # video_path += "?alt=media&token=" + res["downloadTokens"]
#
#     text = pred.predict(video_path)
#
#     result = {
#         'word': text
#     }
#
#     return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)