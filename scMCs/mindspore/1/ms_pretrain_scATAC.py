import argparse
import os
import warnings

import numpy as np
import scipy.io as sio
import umap
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, normalized_mutual_info_score
from sklearn import metrics

from ms_DAE_Ber import DAE_Ber
from ms_DAE_ZINB import DAE_ZINB
from process_data import read_dataset, normalize
import mindspore.nn as nn
import mindspore.numpy as mnp
from mindspore import ops as P, Tensor


def _nan2inf(x):
    return P.where(P.isnan(x), P.zeros_like(x) + mnp.inf, x)


def binary_cross_entropy(x_pred, x):
    return - P.sum(x * P.log(x_pred + 1e-8) + (1 - x) * P.log(1 - x_pred + 1e-8), axis=1)


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
                     + P.mvlgamma(P.Abs(x + theta), p=1)
                     - P.mvlgamma(P.Abs(theta), p=1)
                     - P.mvlgamma(P.Abs(x + 1), p=1))

    mul_case_non_zero = P.mul((x > eps).astype(mnp.float32), case_non_zero)

    res = mul_case_zero + mul_case_non_zero
    result = - P.sum(res, dim=1)
    result = _nan2inf(result)

    return result


class Eucli_dis(nn.Cell):
    """like what you like: knowledge distill via neuron selectivity transfer"""

    def __init__(self):
        super(Eucli_dis, self).__init__()
        pass

    def construct(self, g_s, g_t):
        g_s = g_s.astype(mnp.float32)
        g_t = g_t.astype(mnp.float32)
        ret = P.pow((g_s - g_t), 2)

        return P.sum(ret, dim=1)


class scMODF(nn.Cell):
    def __init__(self, x1, x2, hidden1_1, hidden1_2, hidden1_3, z_emb_size1, dropout_rate1,
                 hidden2_1, hidden2_2, hidden2_3, hidden2_4, z_emb_size2, dropout_rate2
                 ):
        super(scMODF, self).__init__()
        self.N = P.shape(x1)[0]
        self.DAE_ZINB = DAE_ZINB(x1, hidden1_1, hidden1_2, hidden1_3, z_emb_size1, dropout_rate1)
        self.DAE_Ber = DAE_Ber(x2, hidden2_1, hidden2_2, hidden2_3, hidden2_4, z_emb_size2, dropout_rate2)

    def construct(self, x1, x2, scale_factor):
        # encoder and intergation
        '''
        是不是不能放到device中？
        '''
        emb_zinb = self.DAE_ZINB.fc_encoder(x1)
        emb_ber = self.DAE_Ber.fc_encoder(x2)
        emb_i = 0.8 * emb_zinb + 0.2 * emb_ber

        # decoder for DAE_ZINB
        latent_zinb = self.DAE_ZINB.fc_decoder(emb_i)

        normalized_x_zinb = P.softmax(self.DAE_ZINB.decoder_scale(latent_zinb), axis=1)

        batch_size = P.shape(normalized_x_zinb)[0]
        scale_factor = scale_factor * mnp.ones((batch_size, 1))
        scale_factor = P.tile(scale_factor, (1, P.shape(normalized_x_zinb)[1]))

        scale_x_zinb = P.exp(scale_factor) * normalized_x_zinb

        disper_x_zinb = P.exp(self.DAE_ZINB.decoder_r(latent_zinb))
        dropout_rate_zinb = self.DAE_ZINB.dropout(latent_zinb)

        # decoder for DAE_Ber
        latent_ber = self.DAE_Ber.fc_decoder(emb_i)
        recon_x_ber = self.DAE_Ber.decoder_scale(latent_ber)
        Final_x_ber = P.sigmoid(recon_x_ber)

        return emb_zinb, emb_ber, emb_i, normalized_x_zinb, dropout_rate_zinb, disper_x_zinb, scale_x_zinb, Final_x_ber


def runAlgorithm():
    warnings.filterwarnings('ignore')

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch2', type=int, default=500, help='Number of epochs to training_dae_scATAC.')

    parser.add_argument('--training_dae_scATAC', type=bool, default=True, help='Training dae.')

    parser.add_argument('--pretrain_path2', type=str,
                        default='/home/lrren/Single_cell_expermients/DSC-Net-master-pytorch/dae_zinb/SNARE/pre_train/dae_scATAC.pkl')

    params = parser.parse_args()
    data_root = "..\\..\\data\\datasets\\SNARE"
    X1 = os.path.join(data_root, 'scRNA_seq_SNARE.tsv')
    X2 = os.path.join(data_root, 'scATAC_seq_SNARE.txt')
    x3 = os.path.join(data_root, 'cell_metadata.txt')

    x1, x2, train_index, test_index, label_ground_truth, _ = read_dataset(File1=X1, File2=X2,
                                                                          File3=x3, File4=None,
                                                                          transpose=True, test_size_prop=0.0,
                                                                          state=1, format_rna="table",
                                                                          formar_epi="table")

    x2 = normalize(x2, filter_min_counts=True,
                   size_factors=False, normalize_input=False,
                   logtrans_input=False)

    x_scATAC = x2.X
    x_scATACraw = x2.raw.X
    x_scATAC_size_factor = x2.obs['size_factors'].values

    x_scATAC = Tensor(x_scATAC)
    x_scATACraw = Tensor(x_scATACraw)
    x_scATAC_size_factor = Tensor(x_scATAC_size_factor)

    N2, M2 = x_scATAC.shape

    result = []

    if params.training_dae_scATAC:
        print("===== Pretrain a DAE_scATAC.")
        dae_scATAC = DAE_Ber(M2, hidden1=3000, hidden2=2500, hidden3=1000, hidden4=128, z_emb_size=16, dropout_rate=0.1)
        print(dae_scATAC)
        optimizer = nn.Adam(params=dae_scATAC.trainable_params(), lr=0.00001)
        train_loss_list2 = []

        for epoch in range(params.epoch2):
            '''我不确定这个epoch2是什么东西'''
            total_loss = 0
            # optimizer.zero_grad()
            # 梯度清零
            emb_scATAC, recon_x = dae_scATAC(x_scATAC)
            loss_ber = P.mean(binary_cross_entropy(recon_x, x_scATAC))
            loss_dae_scATAC = loss_ber
            # loss_dae_scATAC.backward()
            # optimizer.step()
            # 反向传播
            train_loss_list2.append(loss_dae_scATAC.item())
            print("epoch {} loss={:.4f} ".format(epoch, loss_dae_scATAC))

            emb_scATAC = emb_scATAC.asnumpy()  # 需要保证 emb_scATAC 为 ms.Tensor
            np.savetxt('emb_scATAC.txt', emb_scATAC)
            sio.savemat('emb_scATAC.mat', {'emb_scATAC': emb_scATAC})

            reducer = umap.UMAP(n_neighbors=15, n_components=2, metric="euclidean")
            emb_scATAC_umap = reducer.fit_transform(emb_scATAC)

            np.savetxt('emb_scATAC_umap.txt', emb_scATAC_umap)
            sio.savemat('emb_scATAC_umap.mat', {'emb_scATAC_umap': emb_scATAC_umap})

            kmeans = KMeans(n_clusters=4)
            label_pred_emb_scATAC_umap = kmeans.fit_predict(emb_scATAC_umap)

            result = label_pred_emb_scATAC_umap

            plt.scatter(emb_scATAC_umap[:, 0], emb_scATAC_umap[:, 1], c=label_pred_emb_scATAC_umap)
            plt.gca().set_aspect('equal', 'datalim')
            plt.colorbar(boundaries=np.arange(5) - 0.5).set_ticks(np.arange(4))
            plt.xlabel('UMAP1')
            plt.ylabel('UMAP2')
            plt.show()

            SI_dae_scATAC = silhouette_score(emb_scATAC_umap, label_pred_emb_scATAC_umap)
            NMI_dae_scATAC = round(
                normalized_mutual_info_score(label_ground_truth, label_pred_emb_scATAC_umap, average_method='max'),
                3)
            ARI_dae_scATAC = round(metrics.adjusted_rand_score(label_ground_truth, label_pred_emb_scATAC_umap), 3)
            print('SI_dae_scATAC = {:.4f}'.format(SI_dae_scATAC))
            print('NMI_dae_scATAC = {:.4f}'.format(NMI_dae_scATAC))
            print('ARI_dae_scATAC = {:.4f}'.format(ARI_dae_scATAC))
        print('===== Finished of training_dae_scATAC =====')
