import numpy as np
import warnings
warnings.simplefilter(action='ignore',category=FutureWarning)
from find_k_methods import elbow_method, davies_bouldin
from tqdm import tqdm
from sklearn.cluster import KMeans

import time
def findkbydb(npemb):
    dataset = 'reverb45k'

    n_jobs=30
    # n_jobs=1

    if dataset == 'OPIEC' or dataset == 'reverb45k':
        E_init = npemb
        input_embed = []
        print(np.shape(E_init)[0])
        for id in range(np.shape(E_init)[0]):

            input_embed.append(E_init[id])
        input_embed = np.array(input_embed)

    if dataset == 'OPIEC':
        level_one_min, level_one_max, level_one_gap = 500, 1000, 100
    elif dataset == 'reverb45k':
        level_one_min, level_one_max, level_one_gap = 5000, 10000, 1000
    else:
        level_one_min, level_one_max, level_one_gap = 1, 13, 1
    cluster_list = range(level_one_min, level_one_max, level_one_gap)
    #print('level_one_min, level_one_max, level_one_gap:', level_one_min, level_one_max, level_one_gap)
    k_list = list(cluster_list)

    method2first_cluster_num_dict = dict()


    k_min = level_one_min
    k_max = level_one_max
    #print('k_min:', k_min, 'k_max:', k_max)
    #print('k_list:', type(k_list), len(k_list), k_list)
    index_davies_bouldin = np.zeros((len(k_list)))





    for i in tqdm(range(len(k_list))):
        k = k_list[i]
        km1 = KMeans(n_clusters=k, n_init=30, max_iter=300, tol=1e-9, n_jobs=n_jobs).fit(input_embed)

        index_davies_bouldin[i] = davies_bouldin(input_embed, km1.cluster_centers_, km1.labels_)


    est_k_davies_bouldin = k_list[elbow_method(index_davies_bouldin)]
    #print('For davies_bouldin : Selected k =', est_k_davies_bouldin)
    method2first_cluster_num_dict['davies_bouldin'] = est_k_davies_bouldin

    #print('Golden cluster number : ', cluster_num)
    #print()

    if dataset == 'OPIEC' or dataset == 'reverb45k':
        #print('Level two:')
        data = input_embed

        for method in method2first_cluster_num_dict:
            level_one_k = method2first_cluster_num_dict[method]
            t0 = time.time()
            real_time = time.strftime("%Y_%m_%d") + ' ' + time.strftime("%H:%M:%S")
            #print('time:', real_time)
            #print('Method:', method, 'level_one_k:', level_one_k)
            level_two_min, level_two_max, level_two_gap = level_one_k - level_one_gap, level_one_k + level_one_gap, int(
                level_one_gap / 10)
            minK, maxK = level_two_min, level_two_max
            cluster_list = range(level_two_min, level_two_max, level_two_gap)
            k_list = list(cluster_list)
            est_k = 0
            #print('level_two_min, level_two_max, level_two_gap:', level_two_min, level_two_max, level_two_gap)

            if method == 'davies_bouldin':
                index = np.zeros((len(k_list)))
                for i in tqdm(range(len(k_list))):
                    k = k_list[i]
                    if k != 1:
                        km1 = KMeans(n_clusters=k, n_init=30, max_iter=300, tol=1e-9, n_jobs=n_jobs).fit(data)
                        index[i] = davies_bouldin(data, km1.cluster_centers_, km1.labels_)
                est_k = k_list[elbow_method(index)]

            #print(method, ' k: ', est_k)
            #print()

        #print('Golden cluster number : ', cluster_num)
        return est_k
