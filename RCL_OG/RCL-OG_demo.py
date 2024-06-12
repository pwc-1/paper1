import argparse
import logging
import os
import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from gendata.simulation import DAG, IIDSimulation
from RCL_OG import DAG_RL
from utils.analyze_utils import graph_prunned_by_coef
from utils.true_probability import probability
from utils.plot_dag import GraphDAG

parser = argparse.ArgumentParser(description='Configuration')
config = parser.parse_args(args=[])
config.num_variables = 10
config.num_samples = 200
config.embed_dim = 128
config.hidden_dim = 128
config.heads = 4
config.dropout_rate = 0
config.device_type = 'cpu'
config.device_ids = 0
config.learning_rate = 1e-5
config.epoch = 3000
config.score_type = 'BIC_different_var'
config.reg_type = 'LR'
config.capacity = 20000
config.greedy = 0.5
config.batch_size = 32

if torch.cuda.is_available():
    logging.info('GPU is available.')
else:
    logging.info('GPU is unavailable.')
    if config.device_type == 'gpu':
        raise ValueError("GPU is unavailable, "
                         "please set device_type = 'cpu'.")
if config.device_type == 'gpu':
    if config.device_ids:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(config.device_ids)
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
config.device = device
for i in range(1):
    for num_variables in [10]:
        config.num_variables = num_variables
        for num_edges in [2]:
            ##read real data
            # true_dag = np.loadtxt('real_data_graph.csv', dtype=np.float, delimiter=',', unpack=False)
            # X = np.loadtxt('real_data.csv', dtype=np.float, delimiter=',', unpack=False)
            # config.num_variables = X.shape[1]
            # config.num_samples = X.shape[0]
            #generate simulated data
            weighted_random_dag = DAG.erdos_renyi(n_nodes=num_variables, n_edges=int(num_edges * num_variables), weight_range=(0.5, 2.0), seed=i)
            dataset = IIDSimulation(W=weighted_random_dag, n=config.num_samples, method='linear', sem_type='gauss')  # 生成模型
            true_dag, X = dataset.B, dataset.X
            adj = np.ones([config.num_variables, config.num_variables]) - np.eye(config.num_variables)
            ##pre pruning
            #adj = np.array(graph_prunned_by_coef(adj, X, if_normal=True, th=0.09)).T
            ##run algorithm
            rl = DAG_RL(config, X, adj)
            rl.learn()
            ##sampling
            # sample_num, max_tpr, max_tpr_samples = rl.sample_num(true_dag)
            # print(sample_num, max_tpr, max_tpr_samples)
            #get sample results
            positions, logp = rl.sample(1000)
            mets1 = rl.get_e_matrix(positions, logp, true_dag)##get mean result
            mets2 = rl.get_max_matrix(positions, true_dag)##get result with best score
            mets3 = rl.max_sample_matrix(true_dag)##get result with biggest probability
            ##record results
            file_handle = open('LG'+str(num_variables)+'result_ER'+str(num_edges)+'.txt', 'a')
            file_handle.write('type:{},fdr:{},tpr:{},fpr:{},shd:{},precision:{},F1:{}\n'.format('RCL-OG-E', mets1['fdr'], mets1['tpr'], mets1['fpr'], mets1['shd'], mets1['precision'], mets1['F1']))
            file_handle.write('type:{},fdr:{},tpr:{},fpr:{},shd:{},precision:{},F1:{}\n'.format('RCL-OG-E', mets2['fdr'], mets2['tpr'], mets2['fpr'], mets2['shd'], mets2['precision'], mets2['F1']))
            file_handle.write('type:{},fdr:{},tpr:{},fpr:{},shd:{},precision:{},F1:{}\n'.format('RCL-OG-E', mets3['fdr'], mets3['tpr'], mets3['fpr'], mets3['shd'], mets3['precision'], mets3['F1']))
            file_handle.close()
            ##test performance of approximate transition probability
            # samples = rl.sample_state_action(50)
            # samples['states'] = np.squeeze(samples['states']).tolist()
            # samples['actions'] = np.squeeze(samples['actions']).tolist()
            # real_pro = probability(X, config, adj)
            # pros = []
            # target_pros = []
            # for j in tqdm(range(50)):
            #     pro = rl.get_probability(samples['states'][j], samples['actions'][j])
            #     target_pro = real_pro.get_probability(samples['states'][j], samples['actions'][j])
            #     pros.append(pro)
            #     target_pros.append(target_pro)
            # plt.scatter(pros, target_pros, c='b', edgecolors='r')
            # plt.show()
            # file_handle = open('tran_pro.txt', 'a')
            # file_handle.write(str(pros)+'\n'+str(target_pros)+'\n')
            # file_handle.close()
