import argparse
import torch
from models.linear_server import LinearServer
from utils.evaluation import MetricsDAG
from dataset.Generate_Data import *

#experiment for real data

parser = argparse.ArgumentParser(description='Configuration')
config = parser.parse_args(args=[])
config.device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

config.graph_type = 'er'
config.d = 11
config.dims = [11, 10, 1]
config.bias = True
config.e = 20
config.seed = 0
config.K = 9
config.gen_method = 'multiiid'
config.ns = [200, 200, 200, 200, 200, 200, 200, 200, 200, 200]
config.sem_type = 'gauss'
config.method = 'linear'

config.lambda1 = 0.01
config.lambda2 = 0.01
config.lambda3 = 0.001
config.max_iter = 20
config.rho_max = 1e+16
config.h_tol = 1e-8
config.C = 1
config.loss_type = 'l2'
config.threshold = 0.3

# read ground true graph and data
true_dag = np.loadtxt('dataset/Gv2.csv', dtype=np.float, delimiter=',', unpack=False)
X = np.loadtxt('dataset/real_data_7466.csv', dtype=np.float, delimiter=',', unpack=False)
# divide the data to the clients
dataset = []
config.ns = []
indexs = X.shape[0]
index = np.random.choice(indexs, 7460, replace=False)
for i in range(10):
    dataset.append(X[index[i*746:(i+1)*746]])
    config.ns.append(746)
# run algorithm
dataset = np.array(dataset)
Server = LinearServer(config, dataset)
Server.run(true_dag)
B_est = Server.get_adj()
met = MetricsDAG(B_est, true_dag)
print(B_est)
print(met.metrics)