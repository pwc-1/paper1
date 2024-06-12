# 在版本4的基础上增加标签之间的相关性，标签的编码方式不再是One-Hot，而是带有权重的向量

from model3 import *
import numpy as np
import mindspore
from mindspore import nn
from mindspore import context
from mindspore import Tensor
from mindspore import load_checkpoint, load_param_into_net
from mindspore.ops import operations as P
from mindspore.nn.metrics import Precision
from mindspore.nn.metrics import Recall
from sklearn.metrics import precision_recall_curve, auc, roc_curve

# 设置计算设备
device = 'CPU'
context.set_context(device_target=device)

EP_list = {}
with open('data/USPT-50K/other_data/原子电负性表.txt', 'r', encoding='UTF-8') as file_object:
    for line in file_object:
        line = line.rstrip()
        Name, ep = line.split('	')
        EP_list[Name] = float(ep)

Mol_Xing = Chem.MolFromSmiles('*')
Labels_dict = {}
Labels = []
with open('data/USPT-50K/other_data/Labels.txt', 'r', encoding='UTF-8') as file_object:
    for line in file_object:
        line = line.rstrip()
        if line not in Labels:
            Labels.append(line)
count = 1
for num1 in range(len(Labels)):
    if Labels[num1] not in Labels_dict.keys():
        Labels_dict[Labels[num1]] = [0.0]*(count-1) + [1.0] + [0.0]*(len(Labels) - count)
        count += 1

Datapath = 'data/USPT-50K/test.txt'
Data = load_data(Datapath)  #  S_P_d = {}, S_R1_d = {}, S_R2_d = {}, Bond_dict = {},F_D_P = {},F_D_R1 = {} F_D_R2 = {},A_D_P = {}, A_D_R1 = {},A_D_R2 = {}

Train_list = list(range(0, 3))
Test_list = list(range(0, 3))

mini_Labels_dict, mini_Labels, Label_num, Initial_Features, Adj_Labels = GetLabelsrelationship(Labels, Labels_dict, Data)
model = RedOut(len(Data[4]['0'][0]))
model2 = Classify(len(Data[4]['0'][0]), Label_num)
model3 = Model3(Label_num, Label_num)

# 将初始特征和邻接标签转换为 Tensor
Adj_Labels = Tensor(Adj_Labels)
Initial_Features = Tensor(Initial_Features)
print(Initial_Features.shape)
print(Adj_Labels.shape)

txt = open('output/Adj_Labels.txt', 'w', encoding='UTF-8')
for i in Adj_Labels:
    for j in i:
        txt.write(str(j.item()) + '	')
    txt.write('\n')

# 设置优化器
optimizer1 = nn.SGD(model.trainable_params(), learning_rate=0.01, weight_decay=5e-4)
optimizer2 = nn.SGD(model2.trainable_params(), learning_rate=0.01, weight_decay=5e-4)
optimizer3 = nn.SGD(model3.trainable_params(), learning_rate=0.01, weight_decay=1e-4)

loss_fn = nn.BCELoss()
loss_fn_3 = nn.MSELoss()

if __name__ == "__main__":
    train(1000, Data, Train_list, Test_list, model, model2, optimizer1, optimizer2, loss_fn, loss_fn_3, mini_Labels_dict, Initial_Features, model3, AtomSymble_One_Hot, EP_list)
