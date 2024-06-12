import argparse
import math

import numpy as np
import torch

from gendata.evaluation import MetricsDAG
from gendata.simulation import DAG, IIDSimulation
from models.q_net import q_net
from models.rewards import get_reward
from scores.bge_score import BGeScore
from utils.analyze_utils import from_order_to_graph, graph_prunned_by_coef
from utils.replaybuffer import replaybuffer
from utils.env import causalenv
from utils.true_probability import probability

parser = argparse.ArgumentParser(description='Configuration')
config = parser.parse_args(args=[])
config.num_variables = 20
config.num_samples = 200
config.embed_dim = 64
config.hidden_dim = 64
config.heads = 4
config.dropout_rate = 0.0
config.device_type = 'cpu'
config.device_ids = 0
config.learning_rate = 0.001
config.score_type = 'BIC_different_var'
config.reg_type = 'LR'
config.capacity = 100_000
config.greedy = 1
config.batch_size = 64

seeds = []
num = 0
for i in range(100):
    weighted_random_dag = DAG.erdos_renyi(n_nodes=config.num_variables, n_edges=int(2 * config.num_variables), weight_range=(0.5, 2.0), seed=i)
    dataset = IIDSimulation(W=weighted_random_dag, n=200, method='linear', sem_type='gauss')  # 生成模型
    true_dag, X = dataset.B, dataset.X
    # graph = np.array(graph_prunned_by_coef(from_order_to_graph([3, 4, 1, 2, 0]), X, if_normal=False)).T
    # print(graph)
    adj = np.ones([config.num_variables, config.num_variables]) - np.eye(config.num_variables)
    adj = np.array(graph_prunned_by_coef(adj, X, if_normal=True, th=0.09)).T
    met = MetricsDAG(adj, true_dag)
    if met.metrics['tpr'] > 0.8:
        num += 1
        seeds.append(i)
        print(true_dag)
        print(adj)
        print(i, met.metrics)
        if num == 25:
            print(seeds, len(seeds))
            break
