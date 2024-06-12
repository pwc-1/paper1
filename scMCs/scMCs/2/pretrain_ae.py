# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2021/6/6 17:48
# @Author: LRR
# @File  : test.py
# from numpy import load
import argparse
import time

import torch
import torch.nn as nn
import numpy as np
import scipy.io as io
from scipy.io import loadmat
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn.metrics import silhouette_score as SI
from post_clustering import DI_calcu
from post_clustering import JC_calcu as JI
import umap
from matplotlib import pyplot as plt
from sklearn.cluster import SpectralClustering, KMeans


def maxminnorm(Single_cell_dataset):
    maxcols = Single_cell_dataset.max(axis=0)
    mincols = Single_cell_dataset.min(axis=0)
    data_shape = Single_cell_dataset.shape
    data_rows = data_shape[0]
    data_cols = data_shape[1]
    t = np.empty((data_rows, data_cols))
    for i in range(data_cols):
        t[:, i] = (Single_cell_dataset[:, i] - mincols[i]) / (maxcols[i] - mincols[i])
    return t


class dot_attention(nn.Module):
    """ 点积注意力机制"""

    def __init__(self, attention_dropout=0.0):
        super(dot_attention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None, attn_mask=None):
        """
        前向传播
        :param q:
        :param k:
        :param v:
        :param scale:
        :param attn_mask:
        :return: 上下文张量和attention张量。
        """
        attention = torch.bmm(q, k.transpose(1, 2))
        if scale:
            attention = attention * scale  # 是否设置缩放
        if attn_mask:
            attention = attention.masked_fill(attn_mask, -np.inf)  # 给需要mask的地方设置一个负无穷。
        # 计算softmax
        attention = self.softmax(attention)
        # 添加dropout
        attention = self.dropout(attention)
        # 和v做点积。
        context = torch.bmm(attention, v)
        return context, attention


class MultiHeadAttention(nn.Module):
    """ 多头自注意力"""

    def __init__(self, model_dim, num_heads, dropout=0.0):
        super(MultiHeadAttention, self).__init__()

        self.dim_per_head = model_dim // num_heads  # 每个头的维度
        self.num_heads = num_heads
        self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads)

        self.dot_product_attention = dot_attention(dropout)

        self.linear_final = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)  # LayerNorm 归一化
        # self.exclusivity = torch.nn.L1Loss()
        # self.clusterlayer =

    def forward(self, key, value, query, attn_mask=None):
        # 残差连接
        residual = query

        dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        batch_size = key.size(0)

        # 线性映射
        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)

        # 按照头进行分割
        key = key.view(batch_size * num_heads, -1, dim_per_head)
        value = value.view(batch_size * num_heads, -1, dim_per_head)
        query = query.view(batch_size * num_heads, -1, dim_per_head)

        if attn_mask:
            attn_mask = attn_mask.repeat(num_heads, 1, 1)

        # 缩放点击注意力机制

        scale = (key.size(-1) // num_heads) ** -0.5
        context, attention = self.dot_product_attention(query, key, value, scale, attn_mask)

        # 进行头合并 concat heads
        context = context.view(batch_size, -1, dim_per_head * num_heads)

        # 进行线性映射
        out = self.linear_final(context)

        # dropout
        out = self.dropout(out)

        # 添加残差层和正则化层。
        out = self.layer_norm(residual + out)

        context = out.view(batch_size * N, dim_per_head, -1)
        context = context.permute(2, 0, 1)
        o1 = context[0]
        o2 = context[1]

        return out, attention, o1, o2


class Encoder(nn.Module):
    def __init__(self, ):
        super(Encoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(M, Encoderlayer1),  # 样本数 x encoder_1
            nn.ReLU(),
            nn.Linear(Encoderlayer1, embed_dimension),
            nn.ReLU(),
        )

        self.MHA1 = MultiHeadAttention(embed_dimension, num_heads)

    def forward(self, x):
        #   x: 1047*16
        # encoded = self.encoder(x)   # 1047 * 8
        encoded = x
        encoded = encoded.reshape(batch_size, N, embed_dimension)   # 1 * 1047 * 8
        attn, attn_weights, o1, o2 = self.MHA1(encoded, encoded, encoded)
        #   attn: 1 * 1047 * 8
        #   attn_weights: 2 * 1047 * 1047
        attn = attn.reshape(batch_size * attn.shape[1], attn.shape[2])
        return attn, o1, o2


class Decoder(nn.Module):
    def __init__(self, ):
        super(Decoder, self).__init__()

        self.decoder = nn.Sequential(
            nn.Linear(embed_dimension, Decoderlayer1),  # encoder_3 x encoder_4
            nn.ReLU(),
            nn.Linear(Decoderlayer1, M),
            nn.Sigmoid(),  # compress to a range (0, 1)
        )

    def forward(self, x):
        decoded = self.decoder(x)
        return decoded


class Model(nn.Module):
    def __init__(self, ):
        super(Model, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        attn, o1, o2 = self.encoder(x)
        decoded = self.decoder(attn)
        return o1, o2, decoded


if __name__ == "__main__":
    # seed = 1000
    # torch.manual_seed(seed)  # 为CPU设置随机种子
    parser = argparse.ArgumentParser()
    # ______________ Eval clustering Setting _________
    parser.add_argument('--pretrain_path1', type=str,
                        default='E:/Single-Cell/multiOmics/SNARE/新建文件夹/ae/ae_mha_SF.pkl')

    params = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    print('Using device:', device)

    # load data
    # x = np.loadtxt('E:/Single-Cell/multiOmics/SNARE/新建文件夹/z_xxy.txt')
    # x = x.astype(np.float32)
    # x = torch.from_numpy(x).to(device)

    file = 'E:/PythonProjects/PycharmProjects/pythonProject/IDEC/IDEC改/iMClusts/stickfigures20'
    data = loadmat(file, mat_dtype=True)
    dataset = data['X']

    data = loadmat(file, mat_dtype=True)
    a = data['a']
    a = a.squeeze()
    a = a.astype(np.float32)

    data = loadmat(file, mat_dtype=True)
    b = data['b']
    b = b.squeeze()
    b = b.astype(np.float32)
    # 归一化数据
    dataset = maxminnorm(dataset)
    x = dataset.astype(np.float32)
    x = torch.from_numpy(x).to(device)

    N = x.shape[0]
    M = x.shape[1]
    batch_size = 1
    num_heads = 2

    Encoderlayer1 = M
    embed_dimension = M
    Decoderlayer1 = M

    EPOCH = 200
    LR = 0.01

    head_dim = int(embed_dimension / num_heads)
    ###############################################

Model = Model()
Model.to(device)
optimizer = torch.optim.Adam(Model.parameters(), lr=LR)
loss_func = nn.MSELoss()
print(Model)

for epoch in range(EPOCH):
    b_x = x.view(N, -1)  # batch x, shape (batch, N)

    o1, o2, decoded = Model(b_x)
    loss_recon = loss_func(b_x, decoded)  # mean square error
    print('Epoch: ', epoch, '| loss_recon: %.5f' % loss_recon.data.cpu().numpy())

    optimizer.zero_grad()  # clear gradients for this training step
    loss_recon.backward()  # backpropagation, compute gradients
    optimizer.step()  # apply gradients

    torch.save(Model.state_dict(), params.pretrain_path1)

print("model saved to {}.".format(params.pretrain_path1))
print('===== Finished of training_scmifc =====')

