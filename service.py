from flask import Flask, request
from flask.json import jsonify
import time
import tensorflow as tf
from setting import *
from concatenate import concatenate

import pickle
from keras.models import load_model
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy

import numpy as np
from keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

# loading model
model_path = BEST_MODEL
with open(TOKEN_PATH + 'token_x.pickle', 'rb') as handle_x:
    token_x = pickle.load(handle_x)
with open(TOKEN_PATH + 'token_y.pickle', 'rb') as handle_y:
    token_y = pickle.load(handle_y)

x_index = token_x.word_index
index_y = token_y.index_word

model = load_model(BEST_MODEL, custom_objects={'CRF': CRF, 'crf_loss': crf_loss,
                                               'crf_viterbi_accuracy': crf_viterbi_accuracy})
graph = tf.get_default_graph()


def get_prediction(txt, x_index_, index_y_, model_):
    char2id = [x_index_.get(i, 0) for i in txt]
    _input = pad_sequences([char2id], maxlen=None, dtype='float32')
    result = model_.predict(_input)[0]
    result = [np.argmax(i) for i in result]
    result = [index_y_.get(i, "O") for i in result]
    return result


@app.route('/ner_service/predict', methods=['GET', 'POST'])
def parse():
    t_start = time.time()
    if request.method == 'POST':
        try:
            final = {"ip": request.remote_addr}
            input_ = request.get_json()
            text = input_.get("content", "")
            # 启用多线程
            global graph
            with graph.as_default():
                result = get_prediction(text, x_index, index_y, model)
            final["result"] = result
            final["time_cost"] = "{:.2f} ms".format((time.time() - t_start) * 1000)
            final["status_code"] = 0
            return jsonify(final)
        except:
            return jsonify({"status_code": -1, "retMsg": "param error"})


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8888, debug=False, threaded=True)