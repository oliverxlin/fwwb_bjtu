import pandas as pd
import numpy as np
import sys, os
from flask import *

# 读取训练集数据， 实际使用中应该使用测试集合
data = pd.read_csv('../data/train/trainx.txt', sep='\t')

features = data.columns.values

print("特征如下：{} \n".format(features))
print("数据数目: ", len(data))

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("test_index.html")


@app.route("/query", methods = ["POST"])
def query():
    if request.method == "POST":
        name = request.form["name"]
        ans = get_label(name)
        # 找到返回1， 找不到返回-1， 这里只写找到的情况测试一下。
        return jsonify({"code" : 1 , "ans" : ans})


def get_label(name):
    return data[data["ITEM_NAME" ] == name]["TYPE"][0]

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug = True)

# 0	腾讯QQ黄钻三个月QQ黄钻3个月季卡官方自动充值可查时间可续费	本地生活--游戏充值--QQ充值
# 有的数据可能查不出来。需要洗， 就用上一条测试，再写个找不到的情况就行