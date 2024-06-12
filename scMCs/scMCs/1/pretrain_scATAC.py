_author_ = 'lrren'
# coding: utf-8

import math

_author_ = 'lrren'
# coding: utf-8

import argparse
import os
import warnings
import seaborn as sns
import anndata
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import umap
from sklearn.decomposition import PCA
from DAE_ZINB import DAE_ZINB
from DAE_Ber import DAE_Ber
from torch.optim import Adam
from sklearn import metrics
import scipy.io as sio
from sklearn.metrics.cluster import normalized_mutual_info_score
from torch.autograd import Variable
from torch.distributions import Normal, kl_divergence as kl
import scanpy as sc
import torch.utils.data as data_utils

# import umap.plot
from matplotlib import pyplot as plt
from scipy.io import loadmat
from process_data import read_dataset, normalize, normalize2
from process_data import dopca
from sklearn.cluster import KMeans
from post_clustering import spectral_clustering, acc, nmi, DI_calcu, JC_calcu
from sklearn.metrics import silhouette_score, normalized_mutual_info_score


def _nan2inf(x):
    return torch.where(torch.isnan(x), torch.zeros_like(x) + np.inf, x)


def binary_cross_entropy(x_pred, x):
    #mask = torch.sign(x)
    return - torch.sum(x * torch.log(x_pred + 1e-8) + (1 - x) * torch.log(1 - x_pred + 1e-8), dim=1)


def reconstruction_loss(decoded, x):
    loss_func = torch.nn.MSELoss()
    # x = x.to_dense()
    loss_rec = loss_func(decoded, x)
    return loss_rec


def log_zinb_positive(x, mu, theta, pi, eps=1e-8):

    # x = x.float()

    if theta.ndimension() == 1:

        theta = theta.view(1, theta.size(0))

    softplus_pi = F.softplus(-pi)

    log_theta_eps = torch.log( theta + eps )

    log_theta_mu_eps = torch.log( theta + mu + eps )

    pi_theta_log = -pi + theta * (log_theta_eps - log_theta_mu_eps)

    case_zero = F.softplus(pi_theta_log) - softplus_pi
    mul_case_zero = torch.mul((x < eps).type(torch.float32), case_zero)

    case_non_zero = (-softplus_pi + pi_theta_log
        + x * (torch.log(mu + eps) - log_theta_mu_eps)
        + torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1))

    mul_case_non_zero = torch.mul((x > eps).type(torch.float32), case_non_zero)

    res = mul_case_zero + mul_case_non_zero
    result = - torch.sum(res, dim=1)
    result = _nan2inf(result)

    return result


class Eucli_dis(nn.Module):
    """like what you like: knowledge distill via neuron selectivity transfer"""
    def __init__(self):
        super(Eucli_dis, self).__init__()
        pass

    def forward(self, g_s, g_t):
        g_s = g_s.float()
        g_t = g_t.float()
        ret = torch.pow( (g_s - g_t) , 2)

        return torch.sum( ret, dim = 1 )


class scMODF(nn.Module):
    def __init__(self, x1, x2, hidden1_1, hidden1_2, hidden1_3, z_emb_size1, dropout_rate1,
                 hidden2_1, hidden2_2, hidden2_3, hidden2_4, z_emb_size2, dropout_rate2
                 ):
        super(scMODF, self).__init__()
        self.N = x1.shape[0]
        self.DAE_ZINB = DAE_ZINB(x1, hidden1_1, hidden1_2, hidden1_3, z_emb_size1, dropout_rate1)
        self.DAE_Ber = DAE_Ber(x2, hidden2_1, hidden2_2, hidden2_3, hidden2_4, z_emb_size2, dropout_rate2)

    def forward(self, x1, x2, scale_factor):
        # encoder and intergation
        emb_zinb = self.DAE_ZINB.fc_encoder(x1).to(device)
        emb_ber  = self.DAE_Ber.fc_encoder(x2).to(device)
        # emb_i = self.a * emb_zinb + self.b * emb_ber
        emb_i = 0.8 * emb_zinb + 0.2 * emb_ber

        # decoder for DAE_ZINB
        latent_zinb = self.DAE_ZINB.fc_decoder(emb_i)

        normalized_x_zinb = F.softmax(self.DAE_ZINB.decoder_scale(latent_zinb), dim=1)

        batch_size = normalized_x_zinb.size(0)
        scale_factor.resize_(batch_size, 1)
        scale_factor.repeat(1, normalized_x_zinb.size(1))

        scale_x_zinb = torch.exp(scale_factor) * normalized_x_zinb  # recon_x
        # scale_x = normalized_x  # recon_x

        disper_x_zinb = torch.exp(self.DAE_ZINB.decoder_r(latent_zinb))  # theta
        dropout_rate_zinb = self.DAE_ZINB.dropout(latent_zinb) # pi

        # decoder for DAE_Ber
        latent_ber = self.DAE_Ber.fc_decoder(emb_i)
        recon_x_ber = self.DAE_Ber.decoder_scale(latent_ber)
        Final_x_ber = torch.sigmoid(recon_x_ber)

        return emb_zinb, emb_ber, emb_i, normalized_x_zinb, dropout_rate_zinb, disper_x_zinb, scale_x_zinb, Final_x_ber


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    torch.cuda.cudnn_enabled = False
    # np.random.seed(0)
    # torch.manual_seed(0)
    # torch.cuda.manual_seed(0)
    device = 'cuda:4' if torch.cuda.is_available() else 'cpu'
    print('===== Using device: ' + device)

    # ################ Parameter setting
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch2', type=int, default=500, help='Number of epochs to training_dae_scATAC.')

    # ______________ Eval clustering Setting _________
    parser.add_argument('--training_dae_scATAC', type=bool, default=True, help='Training dae.')

    parser.add_argument('--pretrain_path2', type=str,
                        default='/home/lrren/Single_cell_expermients/DSC-Net-master-pytorch/dae_zinb/SNARE/pre_train/dae_scATAC.pkl')

    # ______________ Build cell graph ______________


    params = parser.parse_args()
    params.device = device

    # ======================================================================= read data from the data_root folder
    print('===== Load scRNA-seq and scATAC data together =====')

    # data_root = '/home/lrren/Single_cell_expermients/scMulti_omics/scMulti_omics_data/CellLineMixtrue/'
    data_root = '/home/lrren/Single_cell_expermients/DSC-Net-master-pytorch/Data/DCCA/SNARE'

    X1 = os.path.join(data_root, 'scRNA_seq_SNARE.tsv')
    X2 = os.path.join(data_root, 'scATAC_seq_SNARE.txt')
    x3 = os.path.join(data_root, 'cell_metadata.txt')  # cell type information

    # # adata: scRNA-seq with samples x genes
    # # adata: scATAC    with samples x peaks
    x1, x2, train_index, test_index, label_ground_truth, _ = read_dataset(File1=X1, File2=X2,
                                                                          File3=x3, File4=None,
                                                                          transpose=True, test_size_prop=0.0,
                                                                          state=1, format_rna="table",
                                                                          formar_epi="table")

    x2 = normalize(x2, filter_min_counts=True,
                   size_factors=False, normalize_input=False,
                   logtrans_input=False)

    print('===== Normalize =====')

    # For scATAC
    x_scATAC = x2.X
    # pca = PCA(n_components=3000)
    # x_scATAC = pca.fit_transform(x_scATAC)

    x_scATACraw = x2.raw.X
    x_scATAC_size_factor = x2.obs['size_factors'].values

    x_scATAC = torch.from_numpy(x_scATAC).to(device)
    x_scATACraw = torch.from_numpy(x_scATACraw).to(device)
    x_scATAC_size_factor = torch.from_numpy(x_scATAC_size_factor).to(device)

    N2, M2 = np.shape(x_scATAC)

    if params.training_dae_scATAC:
        print("===== Pretrain a DAE_scATAC.")
        dae_scATAC = DAE_Ber(M2, hidden1=3000, hidden2=2500, hidden3=1000, hidden4=128, z_emb_size=16, dropout_rate=0.1)
        dae_scATAC.to(device)
        print(dae_scATAC)
        optimizer = Adam(dae_scATAC.parameters(), lr=0.00001)
        train_loss_list2 = []

        for epoch in range(params.epoch2):
            # kl_weight = min(1, epoch / 10)
            total_loss = 0
            optimizer.zero_grad()
            emb_scATAC, recon_x = dae_scATAC(x_scATAC)
            loss_ber = torch.mean(binary_cross_entropy( recon_x, x_scATAC ))
            loss_dae_scATAC = loss_ber
            loss_dae_scATAC.backward()
            optimizer.step()
            train_loss_list2.append(loss_dae_scATAC.item())
            print("epoch {} loss={:.4f} ".format(epoch, loss_dae_scATAC))

            ## check the convergence
            # if len(train_loss_list2) >= 2:
            #     if abs(train_loss_list2[-1] - train_loss_list2[-2]) / train_loss_list2[-2] < 1e-5:
            #         print("converged!!!")
            #         print(epoch)
            #         break

        # save as .mat(txt) and visualization
        emb_scATAC = emb_scATAC.data.cpu().numpy()
        np.savetxt('emb_scATAC.txt', emb_scATAC)
        sio.savemat('emb_scATAC.mat', {'emb_scATAC': emb_scATAC})

        reducer = umap.UMAP(n_neighbors=15, n_components=2, metric="euclidean")
        emb_scATAC_umap = reducer.fit_transform(emb_scATAC)

        np.savetxt('emb_scATAC_umap.txt', emb_scATAC_umap)
        sio.savemat('emb_scATAC_umap.mat', {'emb_scATAC_umap': emb_scATAC_umap})

        kmeans = KMeans(n_clusters=4)
        label_pred_emb_scATAC_umap = kmeans.fit_predict(emb_scATAC_umap)

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

        torch.save(dae_scATAC.state_dict(), params.pretrain_path2)
    print("model saved to {}.".format(params.pretrain_path2))
    print('===== Finished of training_dae_scATAC =====')

