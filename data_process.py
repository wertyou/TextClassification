import pandas as pd
import jieba
import re
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.utils.np_utils import to_categorical
from collections import Counter


class DataProcess():
    def __init__(self):
        self.text = []
        self.label = []
        self.words = []
        self.max_len = 25

    def load_stopwords(self):
        stopwords = []
        file = open("data/stopword.txt", 'r', encoding='UTF-8')
        while True:
            line = file.readline()
            stopwords.append(line.replace('\n', ''))
            if not line:
                break
        file.close()
        return stopwords

    def load_data(self):
        for category in ["体育", "女性", "文学", "校园"]:
            index = 0
            while True:
                index += 1
                filename = "data/" + category + '/' + str(index) + ".txt"
                try:
                    with open(filename, 'r', encoding='gbk') as fp:
                        line = fp.readline()
                        self.text.append(line)
                        self.label.append("__label__" + category)
                except:
                    break

    def text_process(self, stopwords):
        for i in range(len(self.text)):
            # 使用正则表达式过滤非中文字符或数字
            pattern = re.compile(r'[^\u4e00-\u9fa5]')
            self.text[i] = re.sub(pattern, '', self.text[i])
            # jieba 分词
            cut_result = list(jieba.cut(self.text[i]))
            # 过滤停用词
            for j in range(len(cut_result)):
                if cut_result[j] in stopwords:
                    cut_result[j] = ''
                else:
                    # 把所有单词存到集合里
                    if cut_result[j] not in self.words:
                        self.words.append(cut_result[j])

            # 数据填充
            # tmp = self.data_padding([x.strip() for x in list(cut_result) if x != '' and x != ' '])
            tmp = [x.strip() for x in list(cut_result) if x != '' and x != ' ']
            self.text[i] = ' '.join(tmp)

    def data_padding(self, sequence):
        # 序列小于最大长度填充'0'
        if len(sequence) <= self.max_len:
            sequence.extend(['0'] * (self.max_len - len(sequence)))
        else:
            # 序列大于最大长度进行截断
            sequence = sequence[:self.max_len]
        return sequence

    def data_encoding(self, texts, labels):
        with open('../data/word2index.txt') as fp:
            word2index = json.load(fp)

        # 文本编码 -- 找到每个词对应的索引
        data = []
        for text in texts:
            text = text.split(' ')
            tmp = []
            for i in range(len(text)):
                text[i] = word2index.get(text[i], 0)
                tmp.append(text[i])
            data.extend(tmp)

        # 标签编码
        label2ind = {}
        unique_label = list(set(labels))
        for index, label in enumerate(unique_label):
            label2ind[label] = index
        for i in range(len(labels)):
            labels[i] = label2ind[labels[i]]

        # one hot 编码
        # labels = to_categorical(labels, len(set(labels)), dtype=int)
        return np.array(data).reshape(-1, self.max_len), np.array(labels), word2index

    def save2csv(self):
        data = pd.Series(self.text)
        label = pd.Series(self.label)
        df = pd.concat([label, data], axis=1)
        df.columns = ["label", "text"]
        df.to_csv("data/data.csv", index=False)

        f = open("data/data.txt", mode='a',encoding='utf-8')
        for index, row in df.iterrows():
            f.write(row["label"] + " " + row["text"])
            f.write('\n')

        f.close()

    def word2index(self):
        word2ind = {'0': 0}
        for index, word in enumerate(self.words):
            # index + 1 是考虑后续填充的字符0
            word2ind[word] = index + 1

        with open('data/word2index.txt', 'w', ) as fp:
            json.dump(word2ind, fp, ensure_ascii=False)

    def index2word(self):
        ind2word = {}
        for index, word in enumerate(self.words):
            ind2word[index + 1] = word

        with open('data/index2word.txt', 'w', ) as fp:
            json.dump(ind2word, fp, ensure_ascii=False)

    def split_data(self, data, label):
        # shuffle data
        data, label = shuffle(data, label, random_state=2020)
        X_train, X_text, y_train, y_test = train_test_split(data, label, test_size=0.1, random_state=2020,
                                                            stratify=label)
        return X_train, X_text, y_train, y_test

    def class_weight(self, y_train):
        count_res = dict(Counter(y_train))
        for key in count_res.keys():
            count_res[key] = round(len(y_train) / count_res[key], 2)
        return count_res


if __name__ == '__main__':
    data_process = DataProcess()

    stopwords = data_process.load_stopwords()
    data_process.load_data()
    data_process.text_process(stopwords)
    data_process.word2index()
    data_process.index2word()
    data_process.save2csv()
