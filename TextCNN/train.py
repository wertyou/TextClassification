import pandas as pd
import numpy as np

from keras import Model
from sklearn import metrics
from keras.utils import plot_model
from keras.layers import Input, Embedding, Conv1D, MaxPooling1D, Concatenate, Flatten, Dropout, Dense

from data_process import DataProcess


def get_data():
    df = pd.read_csv('../data/data.csv')
    texts = df['text'].values.tolist()
    labels = df['label'].values.tolist()

    return texts, labels


def textcnn(wordsize, label, embedding_matrix=None):
    input = Input(shape=(data_process.max_len,))
    if embedding_matrix is None:
        embedding = Embedding(input_dim=wordsize,
                              output_dim=32,
                              input_length=data_process.max_len,
                              trainable=True)(input)
    else:  # 使用预训练矩阵初始化Embedding
        embedding = Embedding(input_dim=wordsize,
                              output_dim=32,
                              weights=[embedding_matrix],
                              input_length=data_process.max_len,
                              trainable=False)(input)

    convs = []
    for kernel_size in [2, 3, 4]:
        conv = Conv1D(64, kernel_size, activation='relu')(embedding)
        pool = MaxPooling1D(pool_size=data_process.max_len - kernel_size + 1)(conv)
        convs.append(pool)
    concat = Concatenate()(convs)
    flattern = Flatten()(concat)
    dropout = Dropout(0.3)(flattern)
    output = Dense(len(set(label)), activation='softmax')(dropout)
    model = Model(input, output)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    print(model.summary())
    return model


if __name__ == '__main__':
    data_process = DataProcess()
    texts, labels = get_data()

    data, label, word2index = data_process.data_encoding(texts, labels)
    X_train, X_text, y_train, y_test = data_process.split_data(data, label)
    class_weight = data_process.class_weight(y_train)
    model = textcnn(len(word2index), label)
    history = model.fit(X_train, y_train, validation_split=0.05, batch_size=32, epochs=50, class_weight=class_weight,
                        verbose=2)
    plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=False)
    model.save("../model/textcnn.h5")
    predict = model.predict(X_text)
    predict_class = np.argmax(predict, axis=1)
    accuracy = metrics.accuracy_score(y_test, predict_class)
    print("模型准确率：", accuracy)
    print("混淆矩阵:", metrics.confusion_matrix(y_test, predict_class))
    print("分类报告：", metrics.classification_report(y_test, predict_class))
