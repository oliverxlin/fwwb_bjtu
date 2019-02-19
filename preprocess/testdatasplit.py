import pandas as pd
import os

# 文件编码不同， 使用IO更改文件
data = []
with open('../data/test/test.tsv', "rb") as f:
    with open("../data/test/test_raw.tsv", "w") as w:
        w.write(f.readline().decode("utf-8"))
        for line in f.readlines():
            w.write(line.decode("gbk", "ignore"))

data = pd.read_csv('../data/test/test_raw.tsv', sep='\t',encoding = "gbk")
features = data.columns.values
print("特征如下：{} \n".format(features))
print("数据数目: ", len(data))

# 分割测试集
datas = []
start = 0
end = 500000
for i in range(9):
    if end + 500000 * i < data.shape[0]:
        data.iloc[start + 500000 * i : end + 500000 * i ].to_csv("../data/test/Test_Raw_{}".format(i))
    else:
        data.iloc[start + 500000 * i : ].to_csv("../data/test/Test_Raw_{}".format(i))

