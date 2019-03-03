import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import re, os, json
from zhon.hanzi import punctuation
from string import punctuation as english_punc
import jieba
from gensim.models.word2vec import Word2Vec 
from keras.preprocessing import sequence
from keras.preprocessing.text import *
from keras.preprocessing import sequence
tokenizer_dit_dir = "tok_dir"
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

def word_tokenizer(data, update_index = False):
    tok = Tokenizer()

    if os.path.exists(tokenizer_dit_dir) and not update_index:
        with open(tokenizer_dit_dir,'r') as f:
            tok.word_index = json.loads(f.read())
            data_tok = tok.texts_to_sequences(data)
    else:
        tok.fit_on_texts(data)
        with open(tokenizer_dit_dir,'w', encoding= "utf-8") as f:
            f.write(json.dumps(tok.word_index))
        data_tok = tok.texts_to_sequences(data)
    return data_tok

def process_data(data):
    data = shuffle(data)
    features = data.columns.values # 0 商品名 1商品分类
    # 数据清洗
    data[features[0]] = data[features[0]].apply(rm_punc)
    data_x = data[features[0]]
    data_y = data[features[1]]

    # 将y标签分开
    data_y = label_split(data_y)
    data_y = pd.DataFrame(data_y, columns=['label1', 'label2', 'label3'])

    # 分词
    x_word_list = participle(data_x)
    word_tokened = word_tokenizer(x_word_list, update_index=True)

    array = np.array(word_tokened)
    np.save("../data/train/processed_datax", array)
    
    data_y.to_csv("../data/train/processed_datay",columns=['label1', 'label2', 'label3'])
    




if __name__ == '__main__':
    data = pd.read_csv('../data/train/trainx.txt', sep='\t')

    # 特征
    features = data.columns.values

    print("特征如下：{} \n".format(features))
    print("数据数目: ", len(data))
    # 替换无用字符
    data[features[0]]  = data[features[0]].apply(rm_punc)
    process_data(data)
    # print(data.iloc[0:10])
    # arr3 = np.load('./array.npy')
    # print(arr3.tolist())