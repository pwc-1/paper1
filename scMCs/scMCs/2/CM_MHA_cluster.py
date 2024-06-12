# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2021/6/6 17:48
# @Author: LRR
# @File  : test.py
# from numpy import load
import argparse
import time
import scipy.io as sio
import torch
import torch.nn as nn
import numpy as np
import scipy.io as io
from scipy.io import loadmat
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import torch.nn.parameter as Parameter
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


def cosine_Matrix(_matrixA, _matrixB):

    _matrixA_matrixB = torch.mm(_matrixA, torch.transpose(_matrixB, 1, 0))
    # _matrixA_matrixB = torch.mul(_matrixA, _matrixB)
    # 按行求和，生成一个列向量
    # 即各行向量的模
    _matrixA_norm = torch.sqrt(torch.mul(_matrixA, _matrixA).sum(dim=1))
    _matrixB_norm = torch.sqrt(torch.mul(_matrixB, _matrixB).sum(dim=1))
    res = torch.mean(torch.div(_matrixA_matrixB, torch.mul(_matrixA_norm, _matrixB_norm)))
    return res


def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def HSIC(attn_output, h):
    loss = 0
    n = attn_output.shape[1]
    H = torch.eye(n) - (1. / n) * torch.ones([n, n])
    H = H.to(device)
    for i in range(h - 1):
        for j in range(i + 1, h):
            matrix1 = (attn_output[i]).mm(attn_output[i].t().mm(H))
            matrix2 = (attn_output[j]).mm(attn_output[j].t().mm(H))

            loss_temp1 = torch.trace(matrix1.mm(matrix2))
            loss_temp2 = torch.trace(matrix1.mm(matrix1))
            loss_temp3 = torch.trace(matrix2.mm(matrix2))
            loss_temp = loss_temp1 / (torch.sqrt(loss_temp2 * loss_temp3))
            loss = (h * (h - 1) / 2) * (loss + loss_temp)
    return loss


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
        # print('size key2:', key.shape)
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
        # loss_HSIC = HSIC(context, num_heads)

        # loss_HSIC = self.exclusivity(o1, o2)
        # loss_HSIC = torch.mean(torch.sum(torch.abs(o1 * o2), dim=1))
        # loss_HSIC = 1-cosine_Matrix(o1, o2)
        # loss_cos = torch.cosine_similarity(o1, o2)
        # loss_cos = torch.mean(loss_cos / torch.sum(loss_cos, 0))

        return out, context


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
        encoded = encoded.reshape(batch_size, N, embed_dimension)  # 1 * 1047 * 8
        attn, context = self.MHA1(encoded, encoded, encoded)
        #   attn: 1 * 1047 * 8
        #   attn_weights: 2 * 1047 * 1047
        attn = attn.reshape(batch_size * attn.shape[1], attn.shape[2])
        return attn, context


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


class ClusteringLayer(nn.Module):
    def __init__(self, n_clusters, dim_per_head, cluster_centers=None, alpha=1.0):
        super(ClusteringLayer, self).__init__()
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.hidden = dim_per_head
        if cluster_centers is None:
            initial_cluster_centers = torch.zeros(
                self.n_clusters,
                self.hidden,
                dtype=torch.float
            )
            nn.init.xavier_uniform_(initial_cluster_centers)
        else:
            initial_cluster_centers = cluster_centers
        self.cluster_centers = nn.Parameter(initial_cluster_centers)

    def forward(self, x):
        norm_squared = torch.sum((x.unsqueeze(1) - self.cluster_centers) ** 2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        numerator = numerator ** power
        q = (numerator.t() / torch.sum(numerator, 1)).t()  # soft assignment using t-distribution
        return q


class Model(nn.Module):
    def __init__(self, ):
        super(Model, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        attn, context = self.encoder(x)
        decoded = self.decoder(attn)
        # decoded = attn
        return context, decoded


class MC(nn.Module):
    def __init__(self, n_cluster1, n_cluster2, dim_per_head, params):
        super(MC, self).__init__()
        self.Model = Model()
        self.pretrain_path = params.pretrain_path1
        self.n_cluster1 = n_cluster1
        self.n_cluster2 = n_cluster2
        self.dim_per_head = dim_per_head
        self.clusterlayer1 = ClusteringLayer(self.n_cluster1, self.dim_per_head)
        self.clusterlayer2 = ClusteringLayer(self.n_cluster2, self.dim_per_head)

        # # cluster layer
        # self.clusterlayer1 = torch.nn.Parameter(torch.Tensor(self.n_cluster1, self.dim_per_head))
        # torch.nn.init.xavier_normal_(self.clusterlayer1.data)  # torch.nn.init.xavier_normal_: 初始化为正态分布
        #
        # self.clusterlayer2 = torch.nn.Parameter(torch.Tensor(self.n_cluster2, self.dim_per_head))
        # torch.nn.init.xavier_normal_(self.clusterlayer2.data)  # torch.nn.init.xavier_normal_: 初始化为正态分布
    def pretrain(self, path=''):
        # load pretrain weights
        self.Model.load_state_dict(torch.load(self.pretrain_path))
        print('load pretrained weights from', path)

    def forward(self, x):
        context, decoded = self.Model(x)
        o1 = context[0]
        o2 = context[1]
        # clusterlayer1
        q1 = self.clusterlayer1(o1)
        # clusterlayer2
        q2 = self.clusterlayer2(o2)
        # decoded = self.decoder(attn)
        # decoded = attn

        return context, o1, o2, decoded, q1, q2


if __name__ == "__main__":
    # seed = 1000
    # torch.manual_seed(seed)  # 为CPU设置随机种子
    parser = argparse.ArgumentParser()
    # ______________ Eval clustering Setting _________
    parser.add_argument('--pretrain_path1', type=str,
                        default='../ae/ae_mha.pkl')

    params = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    print('Using device:', device)

    x = np.loadtxt('../z_xxy.txt')
    x = x.astype(np.float32)
    x = torch.from_numpy(x).to(device)
    a = np.loadtxt('../label_ground_truth.txt')
    a = a.astype(np.float32)
    b = a

    N = x.shape[0]
    M = x.shape[1]
    batch_size = 1
    num_heads = 2

    Encoderlayer1 = M
    embed_dimension = M
    Decoderlayer1 = M

    EPOCH = 100
    LR = 0.01
    alpha = 1
    beta = 1
    n_cluster1 = 4
    n_cluster2 = 2
    head_dim = int(embed_dimension / num_heads)
    ###############################################

    MC = MC(n_cluster1, n_cluster2, head_dim, params)
    MC.to(device)
    print(MC)
    MC.pretrain()

    optimizer = torch.optim.Adam(MC.parameters(), lr=LR)

    # 初始化聚类中心
    context, decoded = MC.Model(x)

    o1 = context[0]
    o2 = context[1]

    # KMeans
    kmeans1 = KMeans(n_clusters=n_cluster1).fit(o1.data.cpu().numpy())
    cluster_centers1 = kmeans1.cluster_centers_
    cluster_centers1 = torch.tensor(cluster_centers1, dtype=torch.float).to(device)
    MC.clusterlayer1.cluster_centers = torch.nn.Parameter(cluster_centers1).to(device)

    kmeans2 = KMeans(n_clusters=n_cluster2).fit(o2.data.cpu().numpy())
    cluster_centers2 = kmeans2.cluster_centers_
    cluster_centers2 = torch.tensor(cluster_centers2, dtype=torch.float).to(device)
    MC.clusterlayer2.cluster_centers = torch.nn.Parameter(cluster_centers2).to(device)

    Loss_list = []
    Loss_recon_list = []
    Loss_HSIC_list = []
    start = time.time()

    loss_func = nn.MSELoss()
    loss_KL = nn.KLDivLoss(size_average=False)

    for epoch in range(EPOCH):
        b_x = x.view(N, -1)  # batch x, shape (batch, N)

        context, o1, o2, decoded, q1, q2 = MC(b_x)

        # clustering loss
        tar1 = target_distribution(q1).detach()
        tar2 = target_distribution(q2).detach()

        q1_out = q1.argmax(1)
        q2_out = q2.argmax(1)

        loss_kl1 = loss_KL(q1.log(), tar1) / q1_out.shape[0]
        loss_kl2 = loss_KL(q2.log(), tar2) / q2_out.shape[0]
        loss_kl = loss_kl1+loss_kl2

        loss_HSIC = HSIC(context, num_heads)
        # loss_HSIC = torch.mean(torch.sum(torch.abs(o1 * o2), dim=1))
        # loss_HSIC = torch.abs(cosine_Matrix(o1, o2))
        loss_HSIC = loss_HSIC

        loss_recon = loss_func(b_x, decoded)  # mean square error
        Loss_total = loss_recon + alpha * loss_HSIC + beta * loss_kl
        print('Epoch: ', epoch, '| HSIC loss: %.4f' % loss_HSIC.data.cpu().numpy(),
              '| loss_kl: %.4f' % loss_kl.data.cpu().numpy(),
              '| loss_recon: %.4f' % loss_recon.data.cpu().numpy(),
              '| train loss: %.4f' % Loss_total.data.cpu().numpy())

        optimizer.zero_grad()  # clear gradients for this training step
        Loss_total.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients

    S1 = o1.data.cpu().numpy()
    S2 = o2.data.cpu().numpy()

    label_S1 = q1_out.data.cpu().numpy()
    label_S2 = q2_out.data.cpu().numpy()

    reducer = umap.UMAP(n_neighbors=15, n_components=2, metric="euclidean")
    S1_umap = reducer.fit_transform(S1)

    reducer = umap.UMAP(n_neighbors=15, n_components=2, metric="euclidean")
    S2_umap = reducer.fit_transform(S2)

    plt.scatter(S1_umap[:, 0], S1_umap[:, 1], c=label_S1)
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
    plt.show()

    plt.scatter(S2_umap[:, 0], S2_umap[:, 1], c=label_S2)
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
    plt.show()

    # calculate the diversity and quality of S1 and S2
    print('calculate the diversity and quality of S1 and S2')
    NMI_S1S2 = NMI(label_S1, label_S2)
    JI_S1S2 = JI(label_S1, label_S2)
    DI1 = DI_calcu(S1, label_S1)
    DI2 = DI_calcu(S2, label_S2)
    SI1 = SI(S1, label_S1)
    SI2 = SI(S2, label_S2)

    print("NMI_S1S2 = {:.4f}".format(NMI_S1S2))
    print("JI_S1S2 = {:.4f}".format(JI_S1S2))
    print("SI1 = {:.4f}".format(SI1))
    print("SI2 = {:.4f}".format(SI2))
    print("DI1 = {:.4f}".format(DI1))
    print("DI2 = {:.4f}".format(DI2))

    # calculate the diversity of label_S1, label_S2 and cell_type label
    print('calculate the diversity of label_S1, label_S2 and cell_type label')
    label_S1 = label_S1.astype(np.float32)
    label_S2 = label_S2.astype(np.float32)

    NMI1_a = NMI(a, label_S1)
    NMI2_a = NMI(b, label_S2)
    JI1 = JI(a, label_S1)
    JI2 = JI(b, label_S2)

    print("NMI1_a = {:.4f}".format(NMI1_a))
    print("NMI2_a = {:.4f}".format(NMI2_a))
    print("JI1 = {:.4f}".format(JI1))
    print("JI2 = {:.4f}".format(JI2))


    