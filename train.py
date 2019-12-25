import numpy as np
import pickle
from setting import *

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM
from keras import callbacks
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy


class DataSet:
    def __init__(self, data_path, format_type=0):
        """
        data_path: the path of the training dataset
        format_type: 0 (use "\r\n" as the newline symbol); others (use "\n" as the newline symbol).
        """
        self.format_type = format_type
        # load the dataset
        with open(data_path, "rb") as f:
            self.data = f.read().decode("utf-8")
        # processing the data
        self.processed_data = self.process_data()
        self.data_x, self.data_y = self.split_x_y()

    def process_data(self):
        if self.format_type == 0:
            train_data = self.data.split("\r\n\r\n")                        # get each sentence
            train_data = [token.split("\r\n") for token in train_data]      # get each character
        else:
            train_data = self.data.split("\n\n")                            # get each sentence
            train_data = [token.split("\n") for token in train_data]        # get each character or word
        # split character or word and the corresponding label
        train_data = [[j.split() if j[-1] != "S" else [" ", "S"] for j in i if j] for i in train_data]
        # delete last empty line
        if not train_data[-1]:
            train_data.pop()
        return train_data

    def split_x_y(self):
        data_x = []
        data_y = []
        for order, sen in enumerate(self.processed_data):
            sentence = ""
            label = []
            for pair in sen:
                sentence += pair[0]
                label.append(pair[1])
            data_x.append(sentence)
            data_y.append(" ".join(label))
        return data_x, data_y

    def get_x(self):
        tokenizer = Tokenizer(num_words=None, char_level=True, filters='', lower=False)   # set up tokenizer
        tokenizer.fit_on_texts(self.data_x)                     # transfer characters or words to tokens
        x_index = tokenizer.word_index
        x_index["unknown"] = 0
        sequences = tokenizer.texts_to_sequences(self.data_x)   # use tokens to represent each data
        sequences = pad_sequences(sequences, maxlen=None)       # no limitation for the length of input

        # save the token
        with open(TOKEN_PATH + 'token_x.pickle', 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return sequences, x_index

    def get_y(self):
        # set up tokenizer
        tokenizer = Tokenizer(num_words=None, char_level=False, split=" ", filters='\n', lower=False)
        tokenizer.fit_on_texts(self.data_y)                     # transfer characters or words to tokens
        sequences = tokenizer.texts_to_sequences(self.data_y)   # use tokens to represent labels for each data
        sequences = pad_sequences(sequences, maxlen=None)       # no limitation for the length of output as well

        # save the token
        with open(TOKEN_PATH + 'token_y.pickle', 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return sequences, tokenizer.index_word


class NER:
    def __init__(self, word_index_, labels_count):
        self.embedding_dim = EMBEDDING_DIMENSION
        self.word_index = word_index_
        self.labels_count = labels_count
        self.max_characters = None
        self.model = self.build_model()

    def build_model(self):
        model_ = Sequential()
        # Random embedding; mask_zero is for the recurrent networks.
        model_.add(Embedding(len(self.word_index), self.embedding_dim, mask_zero=True))
        # Bidirectional LSTMs
        model_.add(Bidirectional(LSTM(256, return_sequences=True, dropout=DROPOUT, recurrent_dropout=RECURRENT_DROPOUT)))
        # CRF layer
        crf = CRF(self.labels_count, sparse_target=True)
        model_.add(crf)
        model_.summary()
        model_.compile(OPTIMIZER, loss=crf_loss, metrics=[crf_viterbi_accuracy])
        return model_

    def train(self, data_, label, p=False):
        # custom callbacks
        callbacks_list = [
            callbacks.EarlyStopping(monitor='val_crf_viterbi_accuracy', mode="max", patience=ES_PATIENCE),
            callbacks.ModelCheckpoint(filepath=BEST_MODEL, monitor='val_crf_viterbi_accuracy',
                                      save_best_only=True),
            callbacks.ReduceLROnPlateau(monitor='val_crf_viterbi_accuracy',
                                        factor=RL_FACTOR, patience=RL_PATIENCE, min_lr=0.000000000000001)
        ]

        history = self.model.fit(data_, label, batch_size=BATCH_SIZE, epochs=1000000,
                                 callbacks=callbacks_list, validation_split=VALIDATION_SPLIT, shuffle=True)

        self.model.save(END_MODEL, overwrite=True)

        if p:
            import matplotlib.pyplot as plt
            acc = history.history['crf_viterbi_accuracy']
            val_acc = history.history['val_crf_viterbi_accuracy']
            loss = history.history['loss']
            val_loss = history.history['val_loss']

            epochs = range(1, len(acc) + 1)

            plt.plot(epochs, acc, 'bo', label='Training acc')
            plt.plot(epochs, val_acc, 'r', label='Validation acc')
            plt.title('Training and Validation Accuracy')
            plt.legend()
            plt.savefig(PLT_PATH + 'accuracy.png')

            plt.figure()
            plt.plot(epochs, loss, 'bo', label='Training loss')
            plt.plot(epochs, val_loss, 'r', label='Validation loss')
            plt.title('Training and Validation Loss')
            plt.legend()
            plt.savefig(PLT_PATH + 'loss.png')
            # plt.show()

    def predict(self, model_path, data_):
        model_ = self.model
        char2id = [self.word_index.get(i, 0) for i in data_]
        input_data = pad_sequences([char2id], maxlen=None, dtype='float32')
        model_.load_weights(model_path)
        result_ = model_.predict(input_data)[0]
        result_label = [np.argmax(i) for i in result_]
        return result_label


if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # choose GPU device
    data = DataSet(TRAIN_DATA)
    x_sequences, word_index = data.get_x()
    y_sequences, label_dic = data.get_y()
    y_sequences = y_sequences.reshape((y_sequences.shape[0], y_sequences.shape[1], 1))

    model = NER(word_index, len(label_dic))
    model.train(x_sequences, y_sequences, False)
    result = model.predict(BEST_MODEL, "")
    result = [label_dic.get(i, "O") for i in result]
    print(result)
