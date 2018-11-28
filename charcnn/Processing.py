# encoding:utf-8
from tensorflow.contrib import learn
import numpy as np
import random
import os

root_path = os.path.dirname(os.path.abspath(os.path.join(os.getcwd(), ".")))
alphabet = " abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"


class Processing(object):
    '''
        语料的预处理工作
    '''
    def loadPracticeFile(self, filename):
        '''
        :param filename: 文件名
        训练文件格式
        1/t/t全民保/t/t实体
        :return:
        '''
        with open(filename, 'r', encoding='utf-8') as fr:
            articles = []
            tags = []
            for line in fr.readlines():
                data = line.replace("\t\t", "\t").replace("\n", "").split("\t")
                if len(data) == 3:
                    articles.append(self.preprocess(data[1]))
                    tags.append(data[2])
                else:
                    print(line+"------格式错误")
        return articles, tags

    def preprocess(self, doc):
        index = []
        for x in doc:
            x = x.lower()
            if x in alphabet:
                index.append(alphabet.index(x))
            else:
                index.append(0)
        return index

    def embedding(self, articles, tags):
        length = []
        for article in articles:
            length.append(len(article))
        max_length = max(length)
        data_embedding = []
        for article in articles:
            temp = [0] * max_length
            temp[0:len(article)] = article
            data_embedding.append(temp)
        index = [x for x in range(len(articles))]
        random.shuffle(index)
        data_embedding_new = [data_embedding[x] for x in index]
        tags_new = [tags[x] for x in index]
        tags_vec = []
        for x in tags_new:
            temp = [0] * 2
            if x == '0':
                temp[0] = 1
            else:
                temp[1] = 1
            tags_vec.append(temp)

        return data_embedding_new, tags_vec


if __name__ == '__main__':
    processing = Processing()
    processing.loadPracticeFile(root_path+"/data/train.txt")
