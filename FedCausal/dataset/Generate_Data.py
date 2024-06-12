from utils.simulator import DAG, IIDSimulation
import numpy as np

def generate_data(graph_type, node, edge, seed, num_client, gen_method, ns, sem_type, dataset_property=None, method='nonlinear'):
    if graph_type == 'er':
        B_true, W_true = DAG.erdos_renyi_2(n_nodes=node, n_edges=edge, weight_range=(0.5, 2.0), seed=seed, num_clients=num_client)
    elif graph_type == 'sf':
        B_true, W_true = DAG.scale_free_2(n_nodes=node, n_edges=edge, weight_range=(0.5, 2.0), seed=seed, num_clients=num_client)
    else:
        assert False, "invalid graph type {}".format(graph_type)
    if gen_method == 'noniid':
        dataset = NonIID_Simulation(W_true, dataset_property, ns, seed)
    elif gen_method == 'multiiid':
        dataset = Multi_IID_Simulation(W_true, sem_type, ns, method, seed)
    else:
        assert False, "invalid gen_method {}".format(gen_method)
    return B_true, dataset

def NonIID_Simulation(W_true, dp, ns, seed):
    num_client, d = W_true.shape[0], W_true.shape[1]
    dataset = []
    for i in range(num_client):
        #choice of noise scale
        noise_scale = 0.8 if dp[i,2] == 0 else 1
        #choice of linear/non-linear
        if dp[i,0] == 0:
            method, sem_type = 'linear', 'gauss'
        else:
            method = 'nonlinear'
            if dp[i,1] == 0:
                sem_type = 'mlp'
            # elif dp[i,1] == 1:
            #     sem_type = 'gp'
            elif dp[i,1] == 1:
                sem_type = 'mim'
            else:
                sem_type = 'gp-add'
        print(method, sem_type)
        data_part = IIDSimulation(W=W_true[i], n=ns[i], method=method, sem_type=sem_type, noise_scale=noise_scale)
        dataset.append(data_part.X)
    return dataset

def Multi_IID_Simulation(W_true, sem_type, ns, method, seed):
    num_client, _ = W_true.shape[0], W_true.shape[1]
    dataset = []
    data = IIDSimulation(W_true[0], n=sum(ns), method=method, sem_type=sem_type, noise_scale=1.0)
    for i in range(num_client):
        data_part = data.X[sum(ns[0:i]):sum(ns[0:i+1]), :]
        dataset.append(data_part)
    return dataset

def property_generation(num_client):
    linearity = np.random.randint(0, 2, (num_client, 1))
    nonlinear_type = np.random.randint(0, 3, (num_client, 1))
    noise_scale = np.random.randint(0, 2, (num_client, 1))
    return np.hstack([linearity, nonlinear_type, noise_scale])