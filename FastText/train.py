import fasttext
import pandas as pd
from sklearn.utils import shuffle

"""
训练集下载地址:https://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/531810/train_set.csv.zip
测试集下载地址:https://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/531810/test_a.csv.zip
"""

class DataProcess(object):

    def load_data(self):
        df_train = pd.read_csv('train_set.csv', sep='\t')

        # 对类别加上 "__label__"前缀
        df_train['label_ft'] = '__label__' + df_train['label'].astype(str)

        df_train[['text', 'label_ft']].iloc[:195000].to_csv('train.csv', index=None, header=None, sep='\t')

        return df_train

    def split_data(self, df_train):
        # 打乱数据集
        df_train = shuffle(df_train)

        # 训练集
        train_data = df_train[['text', 'label_ft']].iloc[:195000]
        train_data.to_csv('train.csv', index=None, header=None, sep='\t')

        # 挑选5000条数据作为验证集
        validate_data = df_train[['text', 'label_ft']].iloc[-5000:]
        validate_data.to_csv('validate.csv', index=None, header=None, sep='\t')


class FastTextModel(object):

    def __init__(self, ):
        pass

    def train(self):
        model = fasttext.train_supervised(input='train.csv',
                                          label_prefix="__label__",
                                          epoch=30,
                                          dim=32,
                                          lr=0.1,
                                          loss='softmax',
                                          word_ngrams=3,
                                          min_count=2,
                                          bucket=1000000)

        return model

    def save_model(self, model):
        model.save_model("fasttext.bin")

    def load_model(self):
        model = fasttext.load_model("fasttext.bin")

        return model

    # 预测验证集结果
    def test(self):
        model = self.load_model()
        score = model.test("validate.csv")
        precision = score[1]
        recall = score[2]
        f1_score = round(2 * (precision * recall) / (precision + recall), 2)

        print("验证集评测结果： Precision:{}, Recall:{}, F1-score:{}".format(precision, recall, f1_score))

    # 预测5万条测试集A的结果,或者测试集B的结果提交
    def predict_testA(self):
        df_testA = pd.read_csv("test_a.csv")
        test_data = df_testA["text"].values.tolist()

        model = self.load_model()
        res = model.predict(test_data)

        predict_res = [y_[0].replace("__label__", "") for y_ in res[0]]
        print(predict_res)
        predict_label = pd.Series(predict_res, name="label")
        predict_label.to_csv("predict_label.csv", index=False)


if __name__ == '__main__':
    data_process = DataProcess()
    fasttext_model = FastTextModel()
    df_train = data_process.load_data()
    data_process.split_data(df_train)
    model = fasttext_model.train()
    fasttext_model.save_model(model)
    fasttext_model.test()
    fasttext_model.predict_testA()
