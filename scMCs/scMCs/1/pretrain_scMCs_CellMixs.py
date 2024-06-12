_author_ = 'lrren'
# coding: utf-8

import math
import sys
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
from DAE_ZINB import DAE_ZINB
from DAE_Ber import DAE_Ber
from torch.optim import Adam

# import umap.plot
from matplotlib import pyplot as plt
from scipy.io import loadmat
from process_data import read_dataset, normalize, normalize2
from process_data import dopca
from sklearn.cluster import KMeans
from post_clustering import spectral_clustering, acc, nmi, DI_calcu, JC_calcu
from sklearn import metrics
import scipy.io as sio
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

# contrastive loss
def compute_joint(view1, view2):
    """Compute the joint probability matrix P"""

    bn, k = view1.size()
    assert (view2.size(0) == bn and view2.size(1) == k)

    p_i_j = view1.unsqueeze(2) * view2.unsqueeze(1)
    p_i_j = p_i_j.sum(dim=0)
    p_i_j = (p_i_j + p_i_j.t()) / 2.  # symmetrise
    p_i_j = p_i_j / p_i_j.sum()  # normalise

    return p_i_j


def crossview_contrastive_Loss(view1, view2, lamb=9.0, EPS=sys.float_info.epsilon):
    """Contrastive loss for maximizng the consistency"""
    _, k = view1.size()
    p_i_j = compute_joint(view1, view2)
    assert (p_i_j.size() == (k, k))

    p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k)
    p_j = p_i_j.sum(dim=0).view(1, k).expand(k, k)

    #     Works with pytorch <= 1.2
    #     p_i_j[(p_i_j < EPS).data] = EPS
    #     p_j[(p_j < EPS).data] = EPS
    #     p_i[(p_i < EPS).data] = EPS

    # Works with pytorch > 1.2
    p_i_j = torch.where(p_i_j < EPS, torch.tensor([EPS], device=p_i_j.device), p_i_j)
    p_j = torch.where(p_j < EPS, torch.tensor([EPS], device=p_j.device), p_j)
    p_i = torch.where(p_i < EPS, torch.tensor([EPS], device=p_i.device), p_i)

    loss = - p_i_j * (torch.log(p_i_j) \
                      - (lamb + 1) * torch.log(p_j) \
                      - (lamb + 1) * torch.log(p_i))

    loss = loss.sum()

    return loss


class SelfAttention(nn.Module):
    """
    attention_1
    """
    def __init__(self, dropout):
        super(SelfAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):

        queries = q
        keys = k
        values = v
        n, d = queries.shape
        scores = torch.mm(queries, keys.t()) / math.sqrt(d)
        att_weights = F.softmax(scores, dim=1)
        att_emb = torch.mm(self.dropout(att_weights), values)
        return att_weights, att_emb


class MLP(nn.Module):

    def __init__(self, z_emb_size1, dropout_rate):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(z_emb_size1, z_emb_size1),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),

            # nn.Linear(z_emb_size1, z_emb_size1),
            # nn.ReLU(),
            # nn.Dropout(p=dropout_rate),
        )
    def forward(self, z_x, z_y):
        q_x = self.mlp(z_x)
        q_y = self.mlp(z_y)
        return q_x, q_y


class Omics_label_Predictor(nn.Module):
    def __init__(self, z_emb_size1):
        super(Omics_label_Predictor, self).__init__()

        # input to first hidden layer
        self.hidden1 = nn.Linear(z_emb_size1, 5)

        # second hidden layer and output
        self.hidden2 = nn.Linear(5, 2)

    def forward(self, X):

        X = F.sigmoid(self.hidden1(X))
        y_pre = F.softmax(self.hidden2(X), dim=1)
        # y_pre = F.sigmoid(self.hidden2(X))
        return y_pre


class scMODF(nn.Module):
    def __init__(self, N, in_dim1, in_dim2, hidden1_1, hidden1_2, hidden1_3, z_emb_size1, dropout_rate1,
                 hidden2_1, hidden2_2, hidden2_3, hidden2_4, z_emb_size2, dropout_rate2, params
                 ):
        super(scMODF, self).__init__()
        self.N = N
        self.params = params
        self.DAE_ZINB = DAE_ZINB(in_dim1, hidden1_1, hidden1_2, hidden1_3, z_emb_size1, dropout_rate1)
        self.DAE_Ber = DAE_Ber(in_dim2, hidden2_1, hidden2_2, hidden2_3, hidden2_4, z_emb_size2, dropout_rate2)
        self.attlayer1 = SelfAttention(dropout=0.1)
        self.attlayer2 = SelfAttention(dropout=0.1)
        self.attlayer3 = SelfAttention(dropout=0.1)
        self.olp = Omics_label_Predictor(z_emb_size1)
        self.fc = nn.Linear(z_emb_size1+z_emb_size2, z_emb_size1)
        self.mlp = MLP(z_emb_size1, dropout_rate=0.1)

    def forward(self, x1, x2, scale_factor):
        ## encoder
        z_x = self.DAE_ZINB.fc_encoder(x1).to(device)
        z_y  = self.DAE_Ber.fc_encoder(x2).to(device)

        ## attention for omics specific information of scRNA-seq
        zx_weights, z_gx = self.attlayer1(z_x, z_x, z_x)

        ## attention for omics specific information of scATAC
        zy_weights, z_gy = self.attlayer2(z_y, z_y, z_y)

        # # omics-label predictor
        z_conxy = torch.cat([z_gx, z_gy], dim=0)
        y_pre = self.olp(z_conxy)

        # omics-label predictor
        # z_conxy = torch.cat([z_x, z_y], dim=0)
        # y_pre_zx = self.olp(z_x)
        # y_pre_zy = self.olp(z_y)

        ## cell similarity cross scRNA and scATAC
        # project z_x and z_y into the same space
        q_x, q_y = self.mlp(z_x, z_y)

        # contrastive loss to maximize consistency
        cl_loss = crossview_contrastive_Loss(q_x, q_y)

        # capture the consistency information
        emb_con = torch.cat([q_x, q_y], dim=1)
        z_xy = self.fc(emb_con)
        # z_xy = (q_x + q_y)/2

        # z_I = z_gx + self.params.lam * z_gy + self.params.beta * z_xy
        z_I = self.params.beta * z_gx + self.params.lam * z_gy + z_xy
        # z_I = 10*z_gx + 0.001 * z_gy + 1 * z_xy

        # decoder for DAE_ZINB
        latent_zinb = self.DAE_ZINB.fc_decoder(z_I)

        normalized_x_zinb = F.softmax(self.DAE_ZINB.decoder_scale(latent_zinb), dim=1)

        batch_size = normalized_x_zinb.size(0)
        scale_factor.resize_(batch_size, 1)
        scale_factor.repeat(1, normalized_x_zinb.size(1))

        scale_x_zinb = torch.exp(scale_factor) * normalized_x_zinb  # recon_x
        # scale_x = normalized_x  # recon_x

        disper_x_zinb = torch.exp(self.DAE_ZINB.decoder_r(latent_zinb))  # theta
        dropout_rate_zinb = self.DAE_ZINB.dropout(latent_zinb) # pi

        # decoder for DAE_Ber
        latent_ber = self.DAE_Ber.fc_decoder(z_I)
        recon_x_ber = self.DAE_Ber.decoder_scale(latent_ber)
        Final_x_ber = torch.sigmoid(recon_x_ber)

        return z_x, z_y, z_gx, z_gy, q_x, q_y, z_xy, z_I, y_pre, cl_loss, \
               normalized_x_zinb, dropout_rate_zinb, disper_x_zinb, scale_x_zinb, Final_x_ber


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    torch.cuda.cudnn_enabled = False
    # np.random.seed(0)
    torch.manual_seed(1000)
    # torch.cuda.manual_seed(0)
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    print('===== Using device: ' + device)

    # ################ Parameter setting
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
    params.device = device

    # ======================================================================= read data from the data_root folder
    print('===== Load scRNA-seq and scATAC data together =====')

    data_root = '../DSC-Net-master-pytorch/Data/DCCA/SNARE'

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
    x1 = normalize(x1, filter_min_counts=True,
                   size_factors=True, normalize_input=False,
                   logtrans_input=True)

    x2 = normalize(x2, filter_min_counts=True,
                   size_factors=False, normalize_input=False,
                   logtrans_input=False)

    print('===== Normalize =====')

    # For scRNA
    x_scRNA = x1.X
    np.savetxt('CellMix.txt', x_scRNA)

    x_scRNAraw = x1.raw.X
    x_scRNA_size_factor = x1.obs['size_factors'].values

    x_scRNA = torch.from_numpy(x_scRNA).to(device)
    x_scRNAraw = torch.from_numpy(x_scRNAraw).to(device)
    x_scRNA_size_factor = torch.from_numpy(x_scRNA_size_factor).to(device)

    # For scATAC
    x_scATAC = x2.X

    x_scATACraw = x2.raw.X
    x_scATAC_size_factor = x2.obs['size_factors'].values

    x_scATAC = torch.from_numpy(x_scATAC).to(device)
    x_scATACraw = torch.from_numpy(x_scATACraw).to(device)

    classes, label_ground_truth = np.unique(label_ground_truth, return_inverse=True)
    classes = classes.tolist()

    N1, M1 = np.shape(x_scRNA)
    N2, M2 = np.shape(x_scATAC)

    # tensor(N1+N2, 1) \in {0, 1}
    ol_x1 = torch.ones(N1, 1)
    ol_x2 = torch.zeros(N1, 1)
    ol_y1 = torch.zeros(N2, 1)
    ol_y2 = torch.ones(N2, 1)
    ol_x = torch.cat([ol_x1, ol_x2], dim=1).to(device)
    ol_y = torch.cat([ol_y1, ol_y2], dim=1).to(device)
    ol = torch.cat([ol_x, ol_y], dim=0).to(device)
    ol1 = ol.cpu().numpy()
    # print(ol)

    ce_loss = nn.CrossEntropyLoss()
    if params.training_dae_scRNA:
        print("===== Pretrain a scMODF.")
        scMCs = scMODF(N1, M1, M2,
                       hidden1_1=500, hidden1_2=300, hidden1_3=128, z_emb_size1=16, dropout_rate1=0.1,
                       hidden2_1=3000, hidden2_2=2500, hidden2_3=1000, hidden2_4=128, z_emb_size2=16, dropout_rate2=0.1,
                       params=params
                       ).to(device)
        print(scMCs)

        # # load the pretrained weights of ae
        DAE_ZINB_state_dict = torch.load(params.pretrain_path1)
        scMCs.DAE_ZINB.load_state_dict(DAE_ZINB_state_dict)

        DAE_Ber_state_dict = torch.load(params.pretrain_path2)
        scMCs.DAE_Ber.load_state_dict(DAE_Ber_state_dict)
        print("===== Pretrained weights are loaded successfully.")

        optimizer = Adam(scMCs.parameters(), lr=0.0001)
        train_loss_list1 = []
        ans1 = []
        ans2 = []
        for epoch in range(params.epoch1):

            total_loss = 0
            optimizer.zero_grad()

            z_x, z_y, z_gx, z_gy, q_x, q_y, z_xy, z_I, y_pre, cl_loss, \
            normalized_x_zinb, dropout_rate_zinb, disper_x_zinb, scale_x_zinb, Final_x_ber = scMCs(x_scRNA, x_scATAC, x_scRNA_size_factor)

            # ZINB loss
            loss_zinb = torch.mean(log_zinb_positive(x_scRNA, scale_x_zinb, disper_x_zinb, dropout_rate_zinb, eps=1e-8))

            # Ber loss
            loss_Ber = torch.mean(binary_cross_entropy(Final_x_ber, x_scATAC))

            # CE loss
            loss_ce = ce_loss(y_pre, ol)
            y_pre = y_pre.detach().cpu().numpy()

            # contrastive loss
            loss_cl = cl_loss

            loss = loss_zinb + params.alpha1 * loss_Ber + params.alpha2 * loss_ce + params.alpha3 * loss_cl
            loss.backward()
            optimizer.step()

            train_loss_list1.append(loss.item())
            print("epoch {} => loss_zinb={:.4f} loss_Ber={:.4f} loss_ce={:.4f} loss_cl={:.4f} loss={:.4f}".format(epoch, loss_zinb, loss_Ber, loss_ce, loss_cl, loss))
        
       # ************************************************************************************************
        print("===== save as .mat(txt) and visualization on scRNA-seq")
        z_x = z_gx.data.cpu().numpy()
        #np.savetxt('z_x.txt', z_x)
        #sio.savemat('z_x.mat', {'z_x': z_x})

        z_y = z_gy.data.cpu().numpy()
        #np.savetxt('z_y.txt', z_y)
        #sio.savemat('z_y.mat', {'z_y': z_y})

        z_xxy = z_I.data.cpu().numpy()
        #np.savetxt('z_xxy.txt', z_xxy)
        #sio.savemat('z_xxy.mat', {'z_xxy': z_xxy})

        # emb_scRNA_wg = emb_zinb.data.cpu().numpy()
        # np.savetxt('emb_scRNA_wg.txt', emb_scRNA_wg)
        # sio.savemat('emb_scRNA_wg.mat', {'emb_scRNA_wg': emb_scRNA_wg})

        reducer = umap.UMAP(n_neighbors=15, n_components=2, metric="euclidean")
        z_I_umap = reducer.fit_transform(z_xxy)
        #np.savetxt('z_xxy_umap.txt', z_I_umap)
        #sio.savemat('z_xxy_umap.mat', {'z_xxy_umap': z_I_umap})

        kmeans = KMeans(n_clusters=4, random_state=100)
        label_pred_z_I_umap = kmeans.fit_predict(z_I_umap)

        scatter=plt.scatter(z_I_umap[:, 0], z_I_umap[:, 1], c=label_ground_truth, s=10)
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


    print("model saved to {}.".format(params.pretrain_path3 + "scMCs" + "_alpha1_" + str(params.alpha1)
                   + "_alpha2_" + str(params.alpha2) + "_alpha3_" + str(params.alpha3) + "_lambda_" + str(params.lam)
                   + "_beta_" + str(params.beta) + "_epoch_" + str(params.epoch1) + ".pkl"))
    print('===== Finished of training_scmifc =====')
