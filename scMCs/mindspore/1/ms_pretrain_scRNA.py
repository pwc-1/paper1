import argparse
import os
import warnings

import umap
from matplotlib import pyplot as plt
from mindspore import Tensor

from ms_DAE_Ber import DAE_Ber
from ms_DAE_ZINB import DAE_ZINB
import mindspore as ms
import numpy as np
import mindspore.numpy as mnp
import mindspore.nn as nn
from mindspore.common.initializer import initializer
import mindspore.ops as P
import scipy.io as sio
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, normalized_mutual_info_score
from sklearn import metrics
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
                 hidden2_1, hidden2_2, hidden2_3, hidden2_4, z_emb_size2, dropout_rate2):
        super(scMODF, self).__init__()
        self.N = x1.shape[0]
        self.DAE_ZINB = DAE_ZINB(x1, hidden1_1, hidden1_2, hidden1_3, z_emb_size1, dropout_rate1)
        self.DAE_Ber = DAE_Ber(x2, hidden2_1, hidden2_2, hidden2_3, hidden2_4, z_emb_size2, dropout_rate2)
        self.a = ms.Parameter(initializer('constant', [self.N, z_emb_size1], 0.5), name='a', requires_grad=True)
        self.b = 1 - self.a


    def construct(self, x1, x2, scale_factor):
        emb_zinb = self.DAE_ZINB.fc_encoder(x1)
        emb_ber  = self.DAE_Ber.fc_encoder(x2)

        emb_i = 0.8 * emb_zinb + 0.2 * emb_ber

        latent_zinb = self.DAE_ZINB.fc_decoder(emb_i)

        normalized_x_zinb = P.softmax(self.DAE_ZINB.decoder_scale(latent_zinb), axis=1)

        batch_size = normalized_x_zinb.shape[0]
        scale_factor = scale_factor * mnp.ones((batch_size, 1))
        scale_factor = P.tile(scale_factor, (1, P.shape(normalized_x_zinb)[1]))

        scale_x_zinb = P.exp(scale_factor) * normalized_x_zinb

        disper_x_zinb = P.Exp(self.DAE_ZINB.decoder_r(latent_zinb))
        dropout_rate_zinb = self.DAE_ZINB.dropout(latent_zinb)

        latent_ber = self.DAE_Ber.fc_decoder(emb_i)
        recon_x_ber = self.DAE_Ber.decoder_scale(latent_ber)
        Final_x_ber = P.sigmoid(recon_x_ber)

        return emb_zinb, emb_ber, emb_i, normalized_x_zinb, dropout_rate_zinb, disper_x_zinb, scale_x_zinb, Final_x_ber

def runAlgorithm():
    warnings.filterwarnings('ignore')

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch1', type=int, default=500, help='Number of epochs to training_dae_scRNA.')

    parser.add_argument('--training_dae_scRNA', type=bool, default=True, help='Training dae.')

    # ______________ Eval clustering Setting _________
    parser.add_argument('--pretrain_path1', type=str,
                        default='/home/lrren/Single_cell_expermients/DSC-Net-master-pytorch/dae_zinb/SNARE/pre_train/dae_scRNA.pkl')

    params = parser.parse_args()

    print('===== Load scRNA-seq and scATAC data together =====')
    data_root = "..\\..\\data\\datasets\\SNARE"

    X1 = os.path.join(data_root, 'scRNA_seq_SNARE.tsv')
    X2 = os.path.join(data_root, 'scATAC_seq_SNARE.txt')
    x3 = os.path.join(data_root, 'cell_metadata.txt')  # cell type information

    # # adata: scRNA-seq with samples x genes
    x1, x2, train_index, test_index, label_ground_truth, _ = read_dataset(File1=X1, File2=X2,
                                                                          File3=x3, File4=None,
                                                                          transpose=True, test_size_prop=0.0,
                                                                          state=1, format_rna="table",
                                                                          formar_epi="table")
    x1 = normalize(x1, filter_min_counts=True,
                   size_factors=True, normalize_input=False,
                   logtrans_input=True)

    print('===== Normalize =====')

    x_scRNA = x1.X
    # pca = PCA(n_components=300)
    # x_scRNA = pca.fit_transform(x_scRNA)

    x_scRNAraw = x1.raw.X
    x_scRNA_size_factor = x1.obs['size_factors'].values

    x_scRNA = Tensor(x_scRNA)
    x_scRNAraw = Tensor(x_scRNAraw)
    x_scRNA_size_factor = Tensor(x_scRNA_size_factor)

    N1, M1 = x_scRNA.shape

    classes, label_ground_truth = np.unique(label_ground_truth, return_inverse=True)
    classes = classes.tolist()

    if params.training_dae_scRNA:
        print("===== Pretrain a DAE_scRNA.")
        dae_scRNA = DAE_ZINB(M1, hidden1=500, hidden2=300, hidden3=128, z_emb_size=16, dropout_rate=0.1)
        print(dae_scRNA)
        optimizer = nn.Adam(dae_scRNA.parameters(), lr=0.0001)
        train_loss_list2 = []

    for epoch in range(params.epoch1):
        total_loss = 0
        # optimizer.zero_grad()
        '''梯度清零'''
        emb_scRNA, normalized_scRNA, pi_scRNA, disp_scRNA, mean_scRNA = dae_scRNA(x_scRNA, x_scRNA_size_factor)
        loss = P.mean(log_zinb_positive(x_scRNAraw, mean_scRNA, disp_scRNA, pi_scRNA))
        # loss.backward()
        # optimizer.step()
        '''反向传播'''
        train_loss_list2.append(loss.item())
        print("epoch {} loss={:.4f}".format(epoch, loss))

        emb_scRNA = Tensor(emb_scRNA).asnumpy()
        np.savetxt('emb_scRNA.txt', emb_scRNA)
        sio.savemat('emb_scRNA.mat', {'emb_scRNA': emb_scRNA})

        reducer = umap.UMAP(n_neighbors=15, n_components=2, metric="euclidean")
        emb_scRNA_umap = reducer.fit_transform(emb_scRNA)

        np.savetxt('emb_scRNA_umap.txt', emb_scRNA_umap)
        sio.savemat('emb_scRNA_umap.mat', {'emb_scRNA_umap': emb_scRNA_umap})

        kmeans = KMeans(n_clusters=4)
        label_pred_emb_scRNA_umap = kmeans.fit_predict(emb_scRNA_umap)

        plt.scatter(emb_scRNA_umap[:, 0], emb_scRNA_umap[:, 1], c=label_ground_truth)
        plt.xlabel('UMAP1')
        plt.ylabel('UMAP2')
        plt.show()

        SI_dae_scRNA = silhouette_score(emb_scRNA, label_pred_emb_scRNA_umap)
        NMI_dae_scRNA = round(
            normalized_mutual_info_score(label_ground_truth, label_pred_emb_scRNA_umap, average_method='max'),
            3)
        ARI_dae_scRNA = round(metrics.adjusted_rand_score(label_ground_truth, label_pred_emb_scRNA_umap), 3)

        print('NMI_dae_scRNA = {:.4f}'.format(NMI_dae_scRNA))
        print('ARI_dae_scRNA = {:.4f}'.format(ARI_dae_scRNA))
        print('SI_dae_scRNA = {:.4f}'.format(SI_dae_scRNA))

        '''保存参数'''
        ms.save_checkpoint(dae_scRNA, params.pretrain_path1)

    print("model saved to {}.".format(params.pretrain_path1))
    print('===== Finished of training_dae_scRNA =====')





