import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import re, os
from zhon.hanzi import punctuation
from string import punctuation as english_punc
import jieba
from gensim.models.word2vec import Word2Vec 
from keras.preprocessing import sequence

# 替换一些字符 类似于 （ ） ‘ ’ _
def  rm_punc(strs):
    return re.sub(r"[{}]+".format(punctuation + english_punc)," ",strs)

# 分离标签
def label_split(data_y):
    data_y = data_y.apply(lambda x: x.replace(" ", "").replace("--", " ").replace("/", "")).tolist()
    data_y= [each.split() for each in data_y]
    return data_y

# 分词
def participle(data):
    words = []
    for i in range(len(data)):
        result=[]
        seg_list = jieba.cut(data.iloc[i])
        for w in seg_list :#读取每一行分词
            if w != " ":
                result.append(w)
        words.append(result)#将该行分词写入列表形式的总分词列表
    return words

def create_w2v_model(sentences, size=50, window=10, min_count = 1):
    """
    sentences 所有句子
    size 生成的词向量的长度
    """
    sentences = word_data
    model= Word2Vec(size=size, window=window, min_count = min_count)
    model.build_vocab(sentences)
    model.train(sentences,total_examples = model.corpus_count,epochs = model.iter)

    # 保存词向量模型
    model.save("w2v_model")
    return model

if __name__ == '__main__':
    data = pd.read_csv('../data/train/trainx.txt', sep='\t')

    # 特征
    features = data.columns.values

    print("特征如下：{} \n".format(features))
    print("数据数目: ", len(data))
    # 替换无用字符
    data[features[0]]  = data[features[0]].apply(rm_punc)

    # 训练数据的所有商品名生成的词
    word_data = participle(data[features[0]])
    if os.path.exists("w2v_model"):
        model = Word2Vec.load("w2v_model")
    else:
        model = create_w2v_model(word_data)
    # 训练词向量
    print(model.most_similar('床上用品'))

    word_vec = [model[word] for word in word_data]
    x_train = sequence.pad_sequences(word_vec, maxlen=20)
    
