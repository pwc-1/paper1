import argparse
import math
import os
import sys
import warnings

import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.numpy as mnp
import umap
from matplotlib import pyplot as plt
from mindspore import Tensor
import mindspore.ops as P
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, normalized_mutual_info_score
from sklearn import metrics

from ms_DAE_ZINB import DAE_ZINB
from ms_DAE_Ber import DAE_Ber
from process_data import normalize, read_dataset


def _nan2inf(x):
    return P.where(P.isnan(x), P.zeros_like(x) + np.inf, x)


def binary_cross_entropy(x_pred, x):
    return - P.sum(x * P.log(x_pred + 1e-8) + (1 - x) * P.log(1 - x_pred + 1e-8), dim=1)


def reconstruction_loss(decoded, x):
    loss_func = nn.MSELoss()
    loss_rec = loss_func(decoded, x)
    return loss_rec


def log_zinb_positive(x, mu, theta, pi, eps=1e-8):
    if theta.ndim == 1:
        theta = P.reshape(theta, (1, P.shape(theta)[0]))

    softplus_pi = P.Softplus(-pi)

    log_theta_eps = P.log(theta + eps)

    log_theta_mu_eps = P.log(theta + mu + eps)

    pi_theta_log = -pi + theta * (log_theta_eps - log_theta_mu_eps)

    case_zero = P.Softplus(pi_theta_log) - softplus_pi
    mul_case_zero = P.mul((x < eps).astype(mnp.float32), case_zero)

    case_non_zero = (-softplus_pi + pi_theta_log
                     + x * (P.log(mu + eps) - log_theta_mu_eps)
                     + P.lgamma(x + theta)
                     - P.lgamma(theta)
                     - P.lgamma(x + 1))

    mul_case_non_zero = P.mul((x > eps).astype(mnp.float32), case_non_zero)

    res = mul_case_zero + mul_case_non_zero
    result = - P.sum(res, dim=1)
    result = _nan2inf(result)

    return result


# contrastive loss
def compute_joint(view1, view2):
    """Compute the joint probability matrix P"""
    '''未作修改！！！'''
    bn, k = view1.shape
    assert (view2.shape[0] == bn and view2.shape[1] == k)

    p_i_j = view1.unsqueeze(2) * view2.unsqueeze(1)
    p_i_j = p_i_j.sum(dim=0)
    p_i_j = (p_i_j + p_i_j.T) / 2.  # symmetrise
    p_i_j = p_i_j / p_i_j.sum()  # normalise

    return p_i_j


def crossview_contrastive_Loss(view1, view2, lamb=9.0, EPS=sys.float_info.epsilon):
    """Contrastive loss for maximizng the consistency"""
    _, k = view1.shape
    p_i_j = compute_joint(view1, view2)
    assert (p_i_j.shape == (k, k))

    p_i = P.sum(p_i_j, dim=1).view(k, 1) * mnp.ones((k, k))
    p_j = P.sum(p_i_j, dim=0).view(1, k) * mnp.ones((k, k))

    p_i_j = P.where(p_i_j < EPS, Tensor([EPS], dtype=p_i_j.dtype), p_i_j)
    p_j = P.where(p_j < EPS, Tensor([EPS], dtype=p_j.dtype), p_j)
    p_i = P.where(p_i < EPS, Tensor([EPS], dtype=p_i.dtype), p_i)

    loss = - p_i_j * (P.log(p_i_j) \
                      - (lamb + 1) * P.log(p_j) \
                      - (lamb + 1) * P.log(p_i))

    loss = loss.sum()

    return loss


class SelfAttention(nn.Cell):
    """
    attention_1
    """

    def __init__(self, dropout):
        super(SelfAttention, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

    def construct(self, q, k, v):
        queries = q
        keys = k
        values = v
        n, d = queries.shape
        scores = P.mm(queries, P.t(keys)) / math.sqrt(d)
        att_weights = P.softmax(scores, axis=1)
        att_emb = P.mm(self.dropout(p=att_weights), values)
        return att_weights, att_emb


class MLP(nn.Cell):

    def __init__(self, z_emb_size1, dropout_rate):
        super(MLP, self).__init__()
        self.mlp = nn.SequentialCell(
            nn.Dense(z_emb_size1, z_emb_size1),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
        )

    def construct(self, z_x, z_y):
        q_x = self.mlp(z_x)
        q_y = self.mlp(z_y)
        return q_x, q_y


class Omics_label_Predictor(nn.Cell):
    def __init__(self, z_emb_size1):
        super(Omics_label_Predictor, self).__init__()

        # input to first hidden layer
        self.hidden1 = nn.Dense(z_emb_size1, 5)

        # second hidden layer and output
        self.hidden2 = nn.Dense(5, 2)

    def construct(self, X):
        X = P.sigmoid(self.hidden1(X))
        y_pre = P.softmax(self.hidden2(X), axis=1)
        return y_pre


class scMODF(nn.Cell):
    def __init__(self, N, in_dim1, in_dim2, hidden1_1, hidden1_2, hidden1_3, z_emb_size1, dropout_rate1,
                 hidden2_1, hidden2_2, hidden2_3, hidden2_4, z_emb_size2, dropout_rate2, params):
        super(scMODF, self).__init__()
        self.N = N
        self.params = params
        self.DAE_ZINB = DAE_ZINB(in_dim1, hidden1_1, hidden1_2, hidden1_3, z_emb_size1, dropout_rate1)
        self.DAE_Ber = DAE_Ber(in_dim2, hidden2_1, hidden2_2, hidden2_3, hidden2_4, z_emb_size2, dropout_rate2)
        self.attlayer1 = SelfAttention(dropout=0.1)
        self.attlayer2 = SelfAttention(dropout=0.1)
        self.attlayer3 = SelfAttention(dropout=0.1)
        self.olp = Omics_label_Predictor(z_emb_size1)
        self.fc = nn.Dense(z_emb_size1 + z_emb_size2, z_emb_size1)
        self.mlp = MLP(z_emb_size1, dropout_rate=0.1)

    def construct(self, x1, x2, scale_factor):
        z_x = self.DAE_ZINB.fc_encoder(x1)
        z_y = self.DAE_Ber.fc_encoder(x2)

        zx_weights, z_gx = self.attlayer1(z_x, z_x, z_x)
        zy_weights, z_gy = self.attlayer2(z_y, z_y, z_y)

        z_conxy = P.cat([z_gx, z_gy], axis=0)
        y_pre = self.olp(z_conxy)

        q_x, q_y = self.mlp(z_x, z_y)

        cl_loss = crossview_contrastive_Loss(q_x, q_y)

        emb_con = P.cat([q_x, q_y], axis=1)
        z_xy = self.fc(emb_con)

        z_I = self.params.beta * z_gx + self.params.lam * z_gy + z_xy

        latent_zinb = self.DAE_ZINB.fc_decoder(z_I)
        normalized_x_zinb = P.softmax(self.DAE_ZINB.decoder_scale(latent_zinb))

        batch_size = normalized_x_zinb.shape[0]
        scale_factor = scale_factor * mnp.ones((batch_size, 1))
        scale_factor = P.tile(scale_factor, (1, P.shape(normalized_x_zinb)[1]))

        scale_x_zinb = P.exp(scale_factor) * normalized_x_zinb

        disper_x_zinb = P.Exp(self.DAE_ZINB.decoder_r(latent_zinb))
        dropout_rate_zinb = self.DAE_ZINB.dropout(latent_zinb)

        latent_ber = self.DAE_Ber.fc_decoder(z_I)
        recon_x_ber = self.DAE_Ber.decoder_scale(latent_ber)
        Final_x_ber = P.sigmoid(recon_x_ber)

        return z_x, z_y, z_gx, z_gy, q_x, q_y, z_xy, z_I, y_pre, cl_loss, \
            normalized_x_zinb, dropout_rate_zinb, disper_x_zinb, scale_x_zinb, Final_x_ber


def runAlgorithm():
    warnings.filterwarnings('ignore')
    ms.set_seed(1000)

    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain_path1', type=str,
                        default='../pre_train/dae_scRNA.pkl')
    parser.add_argument('--pretrain_path2', type=str,
                        default='../pre_train/dae_scATAC.pkl')
    parser.add_argument('--pretrain_path3', type=str,
                        default='../pre_train/dae_scMCs')

    parser.add_argument('--epoch1', type=int, default=500, help='Number of epochs to training_model.')
    parser.add_argument('--training_dae_scRNA', type=bool, default=True, help='Training dae.')
    # parameters for multi-omics data fusion
    parser.add_argument('--lam', type=int, default=0.1, help='omics fusion for Z_{gY}')
    parser.add_argument('--beta', type=int, default=1, help='omics fusion for Z_{XY}')
    # parameters for model optimization
    parser.add_argument('--alpha1', type=int, default=0.0001, help='weight of loss_ber')
    parser.add_argument('--alpha2', type=int, default=1, help='weight of loss_dis')
    parser.add_argument('--alpha3', type=int, default=0.01, help='weight of loss_cl')

    params = parser.parse_args()
    # params.device = device

    print('===== Load scRNA-seq and scATAC data together =====')

    data_root = "..\\..\\data\\datasets\\SNARE"

    X1 = os.path.join(data_root, 'scRNA_seq_SNARE.tsv')
    X2 = os.path.join(data_root, 'scATAC_seq_SNARE.txt')
    x3 = os.path.join(data_root, 'cell_metadata.txt')

    x1, x2, train_index, test_index, label_ground_truth, _ = read_dataset(File1=X1, File2=X2,
                                                                          File3=x3, File4=None,
                                                                          transpose=True, test_size_prop=0.0,
                                                                          state=1, format_rna="table",
                                                                          formar_epi="table")
    x1 = normalize(x1, filter_min_counts=True,
                   size_factors=True, normalize_input=False,
                   logtrans_input=True)

    x2 = normalize(x2, filter_min_counts=True,
                   size_factors=False, normalize_input=False,
                   logtrans_input=False)

    print('===== Normalize =====')

    x_scRNA = x1.X
    np.savetxt('CellMix.txt', x_scRNA)

    x_scRNAraw = x1.raw.X
    x_scRNA_size_factor = x1.obs['size_factors'].values

    x_scRNA = Tensor(x_scRNA)
    x_scRNAraw = Tensor(x_scRNAraw)
    x_scRNA_size_factor = Tensor(x_scRNA_size_factor)

    x_scATAC = x2.X

    x_scATACraw = x2.raw.X
    x_scATAC_size_factor = x2.obs['size_factors'].values

    x_scATAC = Tensor(x_scATAC)
    x_scATACraw = Tensor(x_scATACraw)

    classes, label_ground_truth = np.unique(label_ground_truth, return_inverse=True)
    classes = classes.tolist()

    N1, M1 = np.shape(x_scRNA)
    N2, M2 = np.shape(x_scATAC)

    ol_x1 = P.ones((N1, 1))
    ol_x2 = P.zeros((N1, 1))
    ol_y1 = P.zeros((N2, 1))
    ol_y2 = P.ones((N2, 1))

    ol_x = P.cat([ol_x1, ol_x2], axis=1)
    ol_y = P.cat([ol_y1, ol_y2], axis=1)

    ol = P.cat([ol_x, ol_y], axis=0)
    ol1 = Tensor(ol).asnumpy()

    ce_loss = nn.CrossEntropyLoss()
    if params.training_dae_scRNA:
        print("===== Pretrain a scMODF.")
        scMCs = scMODF(N1, M1, M2,
                       hidden1_1=500, hidden1_2=300, hidden1_3=128, z_emb_size1=16, dropout_rate1=0.1,
                       hidden2_1=3000, hidden2_2=2500, hidden2_3=1000, hidden2_4=128, z_emb_size2=16, dropout_rate2=0.1,
                       params=params
                       )
        print(scMCs)
        print("===== Pretrained weights are loaded successfully.")

        optimizer = nn.Adam(params=scMCs.trainable_params(), lr=0.0001)
        train_loss_list1 = []
        ans1 = []
        ans2 = []
        for epoch in range(params.epoch1):
            total_loss = 0
            # optimizer.zero_grad()
            '''梯度清零'''
            z_x, z_y, z_gx, z_gy, q_x, q_y, z_xy, z_I, y_pre, cl_loss, \
                normalized_x_zinb, dropout_rate_zinb, disper_x_zinb, scale_x_zinb, Final_x_ber = scMCs(x_scRNA,
                                                                                                       x_scATAC,
                                                                                                       x_scRNA_size_factor)
            loss_zinb = P.mean(log_zinb_positive(x_scRNA, scale_x_zinb, disper_x_zinb, dropout_rate_zinb, eps=1e-8))
            loss_Ber = P.mean(binary_cross_entropy(Final_x_ber, x_scATAC))
            loss_ce = ce_loss(y_pre, ol)
            y_pre = Tensor(y_pre).asnumpy()

            loss_cl = cl_loss

            loss = loss_zinb + params.alpha1 * loss_Ber + params.alpha2 * loss_ce + params.alpha3 * loss_cl
            # loss.backward()
            # optimizer.step()
            '''反向传播'''
            train_loss_list1.append(loss.item())
            print("epoch {} => loss_zinb={:.4f} loss_Ber={:.4f} loss_ce={:.4f} loss_cl={:.4f} loss={:.4f}".format(epoch,loss_zinb,loss_Ber,loss_ce,loss_cl,loss))
            print("===== save as .mat(txt) and visualization on scRNA-seq")

            z_x = Tensor(z_gx).asnumpy()
            z_y = Tensor(z_gy).asnumpy()
            z_xxy = Tensor(z_I).asnumpy()

            reducer = umap.UMAP(n_neighbors=15, n_components=2, metric="euclidean")
            z_I_umap = reducer.fit_transform(z_xxy)

            kmeans = KMeans(n_clusters=4, random_state=100)
            label_pred_z_I_umap = kmeans.fit_predict(z_I_umap)

            scatter = plt.scatter(z_I_umap[:, 0], z_I_umap[:, 1], c=label_ground_truth, s=10)
            plt.xlabel('UMAP1')
            plt.ylabel('UMAP2')
            plt.legend(handles=scatter.legend_elements()[0], labels=classes)
            plt.show()

            SI_z_I = silhouette_score(z_I_umap, label_pred_z_I_umap)
            NMI_z_I = normalized_mutual_info_score(label_ground_truth, label_pred_z_I_umap, average_method='max')
            ARI_z_I = metrics.adjusted_rand_score(label_ground_truth, label_pred_z_I_umap)

            print('NMI_z_xxy = {:.4f}'.format(NMI_z_I))
            print('ARI_z_xxy = {:.4f}'.format(ARI_z_I))
            print('SI_z_xxy = {:.4f}'.format(SI_z_I))

        # print("model saved to {}.".format(params.pretrain_path3 + "scMCs" + "_alpha1_" + str(params.alpha1)
        #                                   + "_alpha2_" + str(params.alpha2) + "_alpha3_" + str(
        #     params.alpha3) + "_lambda_" + str(params.lam)
        #                                   + "_beta_" + str(params.beta) + "_epoch_" + str(params.epoch1) + ".pkl"))
        print('===== Finished of training_scmifc =====')
