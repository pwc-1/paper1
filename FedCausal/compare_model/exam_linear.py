import argparse
from time import time
import torch
from compare_model.linear_server import LinearServer
from utils.evaluation import MetricsDAG
from dataset.Generate_Data import *
from utils.random_sample import random_sample

parser = argparse.ArgumentParser(description='Configuration')
config = parser.parse_args(args=[])
config.device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

config.graph_type = 'er'
config.d = 11
config.bias = True
config.e = 20
config.seed = 0
config.K = 10
config.gen_method = 'multiiid'
config.ns = random_sample(config.K, 2000, 'average')
config.sem_type = 'gauss'
config.method = 'linear'

config.lambda1 = 0.01
config.lambda2 = 0.01
config.max_iter = 20
config.rho_max = 1e+16
config.h_tol = 1e-8
config.C = 1
config.loss_type = 'l2'
config.threshold = 0.3

for d in [10, 20, 40]:
    config.d = d
    config.e = 2 * d
    for seed in range(10):
        config.seed = seed
        dataset_property = property_generation(config.K)
        B_ture, dataset = generate_data(config.graph_type, config.d, config.e, config.seed, config.K, config.gen_method, config.ns, config.sem_type, dataset_property, config.method)
        begin_time = time()
        Server = LinearServer(config, dataset)
        Server.run(B_ture)
        B_est = Server.get_adj()
        end_time = time()
        met = MetricsDAG(B_est, B_ture)
        file_handle = open('results/linear_result.txt', 'a')
        file_handle.write('type:{},time:{},fdr:{},tpr:{},shd:{}\n'.format('fednd-'+str(config.d) + '-' + str(config.e) + '-' + str(config.seed), end_time - begin_time, met.metrics['fdr'], met.metrics['tpr'], met.metrics['shd']))
        file_handle.close()

# true_dag = np.loadtxt('../dataset/Gv2.csv', dtype=np.float, delimiter=',', unpack=False)
# X = np.loadtxt('../dataset/real_data_7466.csv', dtype=np.float, delimiter=',', unpack=False)
# indexs = X.shape[0]
# index = np.random.choice(indexs, 7460, replace=False)
# dataset = []
# for i in range(10):
#     dataset.append(X[index[i*746:(i+1)*746]])
# dataset = np.array(dataset)
# Server = LinearServer(config, dataset)
# Server.run(true_dag)
# B_est = Server.get_adj()
# met = MetricsDAG(B_est, true_dag)
# print(met.metrics)