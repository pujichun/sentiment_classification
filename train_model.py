import pickle
from typing import List, Tuple
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, recall_score, f1_score
from utils import remove_punctuation


class SentimentClassification(object):
    def __init__(self, data_path: str, sep: str = None, model_path: str = './model.h5', n_units: int = 100,
                 text_name: str = None, label_name: str = None, test_size: float = 0.2, epochs: int = 20,
                 input_shape: int = 200) -> None:

        self.data_path = data_path
        if sep:
            self.sep = sep
        self.model_path = model_path
        self.n_units = n_units
        if text_name and label_name:
            self.usecols = [text_name, label_name]
        self.test_size = test_size
        self.epochs = epochs
        self.input_shape = input_shape

    def load_data(self) -> DataFrame:
        file_type = self.data_path.split('.')[-1]
        if file_type == 'xlsx':
            if hasattr(self, 'usecols'):
                data = pd.read_excel(self.data_path, usecols=self.usecols)
                data.columns = ['text', 'label']
                return data
            return pd.read_excel(self.data_path)
        if hasattr(self, 'usecols'):
            if hasattr(self, 'sep'):
                data = pd.read_csv(self.data_path,
                                   sep=self.sep,
                                   usecols=self.usecols)
                data.columns = ['text', 'label']
                return data
            data = pd.read_csv(self.data_path, usecols=self.usecols)
            data.columns = ['text', 'label']
            return data
        if hasattr(self, 'sep'):
            data = pd.read_csv(self.data_path, sep=self.sep)
            data.columns = ['text', 'label']
            return data
        return pd.read_csv(self.data_path)

    def vectoring(self, data: DataFrame) -> Tuple:
        s = data['text'].sum()
        vocabulary = set(s)
        char_dictionary = {
            word: i
            for i, word in enumerate(vocabulary, start=1)
        }
        with open('char_dictionary.pk', mode='wb') as f:
            pickle.dump(char_dictionary, f)
        inverse_char_dictionary = {
            i: word
            for i, word in enumerate(vocabulary, start=1)
        }
        labels = list(data['label'].unique())
        label_dictionary = {label: i for i, label in enumerate(labels)}
        with open('label_dictionary.pk', mode='wb') as f:
            pickle.dump(label_dictionary, f)
        output_dictionary = {i: label for i, label in enumerate(labels)}
        vocabulary_size = len(char_dictionary)
        label_size = len(label_dictionary)
        x = [[char_dictionary[c] for c in sen] for sen in data['text']]
        x = pad_sequences(sequences=x, maxlen=self.input_shape, padding='post', value=0)
        y = [[i] for i in data['label'].map(label_dictionary)]
        y = [to_categorical(label, num_classes=label_size) for label in y]
        y = np.array([list(i[0]) for i in y])
        return x, y, output_dictionary, vocabulary_size, label_size, inverse_char_dictionary

    def create_model(self, input_shape: Tuple, vocabulary_size: int, label_size: int, output_dim: int):
        model = Sequential()
        model.add(Embedding(input_dim=vocabulary_size + 1, output_dim=output_dim,
                            input_length=self.input_shape, mask_zero=True))
        model.add(LSTM(self.n_units, input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(Dense(self.n_units // 2, activation='relu'))
        model.add(Dense(label_size, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()
        return model

    def fit(self):
        data = self.load_data()
        data['text'] = data['text'].apply(remove_punctuation)
        data = shuffle(data)
        x, y, output_dictionary, vocabulary_size, label_size, inverse_char_dictionary = self.vectoring(data)
        model = self.create_model(input_shape=(x.shape[0], x.shape[1]), vocabulary_size=vocabulary_size,
                                  label_size=label_size, output_dim=50)
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=self.test_size)
        model.fit(train_x, train_y, epochs=self.epochs, batch_size=32, verbose=1)
        model.save(self.model_path)
        # model = load_model('./model.h5')
        N = test_x.shape[0]  # 测试的条数
        predict = []
        label = []
        for start, end in zip(range(0, N, 1), range(1, N + 1, 1)):
            sentence = [inverse_char_dictionary[i] for i in test_x[start] if i != 0]
            y_predict = model.predict(test_x[start:end])
            label_predict = output_dictionary[np.argmax(y_predict[0])]
            label_true = output_dictionary[np.argmax(test_y[start:end])]
            # print(''.join(sentence), label_true, label_predict)  # 输出预测结果
            predict.append(label_predict)
            label.append(label_true)

        acc = accuracy_score(label, predict)  # 预测准确率
        print('模型在测试集上的准确率为: %s.' % acc)
        auc = recall_score(label, predict)
        print('模型在测试集上的AUC为: %s.' % auc)
        f1 = f1_score(label, predict)
        print('模型在测试集上的F1为: %s.' % f1)


if __name__ == '__main__':
    sc = SentimentClassification(data_path='new_data.xlsx', text_name='titles', label_name='label', epochs=10)
    sc.fit()
