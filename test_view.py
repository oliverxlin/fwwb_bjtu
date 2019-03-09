import pandas as pd
import numpy as np
import sys, os
from flask import *
from predict import *
from keras.preprocessing.text import *
sys.path.append("networks/")
import inception_model.network as incmodel
import textcnn_model.network as textmodel
import jieba
import tensorflow as tf
import predict

predict = predict.Predict("networks/inception_model/model/inception_model-11960.meta",
                           "networks/inception_model/model/",
                      )   
tok = Tokenizer()

def word_tokenizer(data, update_index = False):
    with open("preprocess/tok_dir",'r') as f:
        tok.word_index = json.loads(f.read())
        data_tok = tok.texts_to_sequences(data)
    return data_tok

def participle(data):
    words = []
    for i in range(len(data)):
        result=[]
        seg_list = jieba.cut(data[i])
        for w in seg_list :#读取每一行分词
            if w != " ":
                result.append(w)
        words.append(result)#将该行分词写入列表形式的总分词列表
    return words


app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/query", methods = ["POST"])
def query():
    if request.method == "POST":
        name = request.form["name"]
        ans = get_label([name])
        # 找到返回1， 找不到返回-1， 这里只写找到的情况测试一下
        print(ans[0][0])
        return jsonify({"code" : 1 , "ans" : str(ans[0][0])})


def get_label(data):
    data = participle(data)
    data = word_tokenizer(data)
    print(data)
    data = sequence.pad_sequences(data, maxlen=20)
    print(data)
    return predict.predict(data)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug = True)


#腾讯QQ黄钻三个月QQ黄钻3个月季卡官方自动充值可查时间可续费	本地生活--游戏充值--QQ充值
#有的数据可能查不出来。需要洗， 就用上一条测试，再写个找不到的情况就行