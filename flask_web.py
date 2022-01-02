from config import config
from predict import TextClassify
from flask import Flask, request

"""
开启网络服务
"""

app = Flask(__name__)
tc = TextClassify(config, 'cnn')


# 设置一个url服务
@app.route('/predict', methods=['GET'])
def cut():
    sentence = request.args.get('sentence')
    return tc.predict(sentence)


if __name__ == '__main__':
    app.run(port=1234, debug=True)
