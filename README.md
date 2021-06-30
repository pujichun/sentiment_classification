# sentiment_classification

整个程序十分简单，提供给初学者或没有学过nlp但是急需模型的同学玩玩

### 运行程序

```shell
python run.py
```

### 配置

配置内容都放在了`config.toml`中，建议不对日志配置进行修改

#### 模型配置参数说明

- data_path：数据文件的路径
- delimiter：csv文件的分隔符，`如果是excel文件配置为字符串就行`
- text_name：文本所在列的名称
- label_name：标签所在列的名称，text_name和label_name如果在数据集中的名称就是`text`和`label`可以配置为空字符串
- test_size：测试集的占数据的比例
- batch_size：批处理的大小
- n_units：lstm中神经元的数量
- input_shape：lstm输入的张量维度
- epochs：训练轮数

#### 日志配置参数说明

- filename：日志文件名称
- fm：日志格式
- level：日志等级

### 程序说明

#### 训练

整个程序的主题都在`train_model.py`中，整个`SentimentClassification`类显得很臃肿，后面考虑将数据读取单独抽离

考虑到数据集不管是excel文件还是平面文件源都十分容易调整格式，所以只做了excel和csv的判别，因为excel和csv文件很常见，没有过多的去在文件读取上进行深究，再深究就显得像是重复造轮子了。

```python
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
```

向量化操作，先将所有字符放到set中，然后建立字典，获取标签然后也建立映射关系，将本文转换映射为向量，然后对向量进行维度填充

```python
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
    inverse_label_dictionary = {i: label for i, label in enumerate(labels)}
    vocabulary_size = len(char_dictionary)
    label_size = len(label_dictionary)
    x = [[char_dictionary[c] for c in sen] for sen in data['text']]
    x = pad_sequences(sequences=x, maxlen=self.input_shape, padding='post', value=0)
    y = [[i] for i in data['label'].map(label_dictionary)]
    y = [to_categorical(label, num_classes=label_size) for label in y]
    y = np.array([list(i[0]) for i in y])
    return x, y, inverse_label_dictionary, vocabulary_size, label_size, inverse_char_dictionary

```

构造模型，因为在`Embedding`层中处理的维度是`input_shape`，所以其实前面也可以不填充，或者不使用`Embedding`层

```python
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
```

最后就是训练了，后面有时间把测试程序抽离

```python
def fit(self):
    data = self.load_data()
    data['text'] = data['text'].apply(remove_punctuation)
    data = shuffle(data)
    x, y, inverse_label_dictionary, vocabulary_size, label_size, inverse_char_dictionary = self.vectoring(data)
    model = self.create_model(input_shape=(x.shape[0], x.shape[1]), vocabulary_size=vocabulary_size,
                              label_size=label_size, output_dim=50)
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=self.test_size)
    model.fit(train_x, train_y, epochs=self.epochs, batch_size=self.batch_size, verbose=1)
    model.save(self.model_path)
    # 测试
    N = test_x.shape[0]
    predict = []
    label = []
    for start, end in zip(range(0, N, 1), range(1, N + 1, 1)):
        sentence = [inverse_char_dictionary[i] for i in test_x[start] if i != 0]
        y_predict = model.predict(test_x[start:end])
        predict.append(np.argmax(y_predict[0]))
        label.append(np.argmax(test_y[start:end]))
        label_predict = inverse_label_dictionary[np.argmax(y_predict[0])]
        label_true = inverse_label_dictionary[np.argmax(test_y[start:end])]
        self.logger.info(f'{"".join(sentence)}    真实标签为{label_true},预测标签为{label_predict}')

    # 评价
    acc = accuracy_score(label, predict)
    self.logger.info(f'模型在测试集上的准确率为: {acc}.')
    auc = recall_score(label, predict)
    self.logger.info(f'模型在测试集上的AUC为: {auc}.')
    f1 = f1_score(label, predict)
    self.logger.info(f'模型在测试集上的F1为: {f1}.')
```

#### 预测

整个预测程序很简单，没有过多的去封装复杂的功能，考虑到更多人在预测时需要的是定制化的功能，仅仅提供了一个接口来处理，如果是文件应该可以和文件读取模块搭配使用，有时间再进行优化吧！

```python

import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 导入字典
with open('char_dictionary.pk', 'rb') as f:
    word_dictionary = pickle.load(f)
with open('label_dictionary.pk', 'rb') as f:
    output_dictionary = pickle.load(f)


def predict(text):
    try:
        # 数据预处理
        input_shape = 50
        x = [[word_dictionary[word] for word in text]]
        x = pad_sequences(maxlen=input_shape, sequences=x, padding='post', value=0)
        # 载入模型
        model_save_path = './model.h5'
        lstm_model = load_model(model_save_path)

        # 模型预测
        y_predict = lstm_model.predict(x)
        label_dict = {v: k for k, v in output_dictionary.items()}
        print('输入语句: %s' % text)
        print('情感预测结果: %s' % label_dict[np.argmax(y_predict)])
        print(y_predict)
    except KeyError as e:
        print("您输入的句子有汉字不在词汇表中，请重新输入！")
        print("不在词汇表中的单词为：%s." % e)


predict('不惨')
```

