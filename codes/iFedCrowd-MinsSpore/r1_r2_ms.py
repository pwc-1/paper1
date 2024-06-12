# -*- coding: utf-8 -*-
import numpy as np
import sympy
import random
import matplotlib.pyplot as plt

import mindspore
from mindspore.common.initializer import One, Normal
from mindspore.common import initializer as init

random.seed(1)
alpha = 80
beta = 50


def cal_r1_ms(gamma_list, T_list):
    gamma_list = mindspore.Tensor(np.array(gamma_list), dtype=mindspore.float32)
    T_list = mindspore.Tensor(np.array(T_list), dtype=mindspore.float32)
    r1_list = gamma_list * T_list
    return float(mindspore.ops.max(r1_list, axis=0)[1])


def cal_r1_max_ms(gamma_list, T_list):
    log_rate = mindspore.Tensor(np.array([2] * len(gamma_list)), dtype=mindspore.float32)
    log_rate = mindspore.ops.log(log_rate)
    log_rate += mindspore.Tensor(np.array([1] * len(gamma_list)), dtype=mindspore.float32)
    gamma_list = mindspore.Tensor(np.array(gamma_list), dtype=mindspore.float32)
    T_list = mindspore.Tensor(np.array(T_list), dtype=mindspore.float32)
    return float(mindspore.ops.max(log_rate*gamma_list*T_list, axis=0)[1])


def cal_r1_random_ms(gamma_list, T_list):
    # log_rate = mindspore.Tensor(np.array([1+sympy.log(2)]*len(gamma_list)), dtype=mindspore.float32)
    log_rate = mindspore.Tensor(np.array([2] * len(gamma_list)), dtype=mindspore.float32)
    log_rate = mindspore.ops.log(log_rate)
    log_rate += mindspore.Tensor(np.array([1] * len(gamma_list)), dtype=mindspore.float32)

    gamma_list = mindspore.Tensor(np.array(gamma_list), dtype=mindspore.float32)
    T_list = mindspore.Tensor(np.array(T_list), dtype=mindspore.float32)
    r1_list_max = log_rate * gamma_list * T_list
    r1_list_min = gamma_list * T_list
    return float(random.choice(r1_list_max))



def cal_r2_ms(delta_list):
    delta_list = mindspore.Tensor(np.array(delta_list), dtype=mindspore.float32)
    return float(mindspore.ops.max(delta_list, axis=0)[1])


def cal_r2_max_ms(delta_list):
    rate = mindspore.Tensor(np.array([1.7]*len(delta_list)), dtype=mindspore.float32)
    delta_list = mindspore.Tensor(np.array(delta_list), dtype=mindspore.float32)
    return float(mindspore.ops.max(rate * delta_list, axis=0)[1])


def cal_r2_random_ms(delta_list):
    rate = mindspore.Tensor(np.array([1.2]*len(delta_list)), dtype=mindspore.float32)
    delta_list = mindspore.Tensor(np.array(delta_list), dtype=mindspore.float32)
    return float(mindspore.ops.max(rate*delta_list, axis=0)[1])



def cal_A_ms(gamma, T, r1):
    h = r1 / (gamma * T) - 1
    h_exp = float(mindspore.ops.exp(mindspore.Tensor(np.array([h]), dtype=mindspore.float32))[0]-1)
    return float(mindspore.ops.min(mindspore.Tensor(np.array([h_exp, 1])), axis=0)[0])

# def cal_A_max(gamma, T, r1):
#     return random.uniform(0, 0.2)



def utility_ms(gamma_list, delta_list, T_list, num):
    A_list = []
    r1 = cal_r1_ms(gamma_list, T_list)
    r2 = cal_r2_ms(delta_list)

    for i in range(num):
        A_list.append(cal_A_ms(gamma_list[i], 2, r1))

    delta_list_tensor = mindspore.Tensor(np.array(delta_list), dtype=mindspore.float32)
    delta_list_tensor = mindspore.Tensor(np.array([1]*len(delta_list)), dtype=mindspore.float32)/delta_list_tensor

    delta_list_tensor_r2 = mindspore.Tensor(np.array(delta_list), dtype=mindspore.float32)
    delta_list_tensor_r2 = mindspore.Tensor(np.array([r2] * len(delta_list)), dtype=mindspore.float32) / delta_list_tensor_r2
    delta_list_tensor_r2 = mindspore.ops.log(delta_list_tensor_r2)

    F_list_tensor = delta_list_tensor * delta_list_tensor_r2

    A_list_tensor = mindspore.Tensor(np.array(A_list), dtype=mindspore.float32)
    T_list_tensor = mindspore.Tensor(np.array(T_list), dtype=mindspore.float32)

    sum_tensor = mindspore.Tensor(np.array([r1]*len(A_list)), dtype=mindspore.float32) * A_list_tensor / T_list_tensor
    sum_tensor += mindspore.Tensor(np.array([r2]*len(A_list)), dtype=mindspore.float32) * F_list_tensor
    sum_cost = mindspore.Tensor.sum(sum_tensor)

    # print('A', alpha * sum(A_list))
    # print('F', beta * sum(F_list))
    # print('Cost', sum_cost)

    # print('A', A_list)


    client_utility_list_tensor = mindspore.Tensor(np.array([r1]*len(A_list)), dtype=mindspore.float32) * A_list_tensor / T_list_tensor
    client_utility_list_tensor += mindspore.Tensor(np.array([r2]*len(A_list)), dtype=mindspore.float32) * F_list_tensor
    client_utility_list_tensor -= mindspore.Tensor(np.array(gamma_list), dtype=mindspore.float32) * (mindspore.Tensor(np.array([1]*len(A_list)), dtype=mindspore.float32) * A_list_tensor) * mindspore.ops.log(mindspore.Tensor(np.array([1]*len(A_list)), dtype=mindspore.float32)+A_list_tensor)
    delta_list_tensor2 = mindspore.Tensor(np.array(delta_list), dtype=mindspore.float32)
    client_utility_list_tensor -= mindspore.ops.exp(delta_list_tensor2*F_list_tensor)
    client_utility_list_tensor += mindspore.Tensor(np.array([5]*len(delta_list)), dtype=mindspore.float32)

    res_sum = mindspore.Tensor(np.array([alpha]), dtype=mindspore.float32) * mindspore.Tensor.sum(A_list_tensor)
    res_sum += mindspore.Tensor(np.array([beta]), dtype=mindspore.float32) * mindspore.Tensor.sum(F_list_tensor)
    res_sum *= 1/num
    res_sum -= mindspore.ops.max(T_list_tensor)[1]
    res_sum -= sum_cost
    print("**********************************************************")
    print("utility_ms")
    print(res_sum)
    print(client_utility_list_tensor)
    return res_sum[0], list(client_utility_list_tensor.asnumpy())




def utility_max_ms(gamma_list, delta_list, T_list, num):
    A_list = []
    r1 = cal_r1_max_ms(gamma_list, T_list)
    r2 = cal_r2_max_ms(delta_list)
    for i in range(num):
        A_list.append(cal_A_ms(gamma_list[i], 2, r1))

    A_list_tensor = mindspore.Tensor(np.array(A_list), dtype=mindspore.float32)

    delta_list_tensor = mindspore.Tensor(np.array(delta_list), dtype=mindspore.float32)
    tensor_1 = mindspore.Tensor(np.array([1]*len(delta_list)), dtype=mindspore.float32)
    tensor_r2= mindspore.Tensor(np.array([r2]*len(delta_list)), dtype=mindspore.float32)
    tensor_1 = tensor_1/delta_list_tensor
    tensor_r2_log = mindspore.ops.log(tensor_r2/delta_list_tensor)
    F_list_tensor = tensor_1 * tensor_r2_log


    T_list_tensor = mindspore.Tensor(np.array(T_list), dtype=mindspore.float32)
    tensor_r1 = mindspore.Tensor(np.array([r1]*len(A_list)), dtype=mindspore.float32)
    sum_tensor = tensor_r1*A_list_tensor/T_list_tensor
    sum_tensor += tensor_r2 * F_list_tensor

    sum_cost = mindspore.Tensor.sum(sum_tensor)

    # print('A', alpha * sum(A_list))
    # print('F', beta * sum(F_list))
    # print('Cost', sum_cost)

    # print('A', A_list)

    client_utility_list_tensor = mindspore.Tensor(np.array([r1] * len(A_list)),
                                                  dtype=mindspore.float32) * A_list_tensor / T_list_tensor
    client_utility_list_tensor += mindspore.Tensor(np.array([r2] * len(A_list)),
                                                   dtype=mindspore.float32) * F_list_tensor
    client_utility_list_tensor -= mindspore.Tensor(np.array(gamma_list), dtype=mindspore.float32) * (
                mindspore.Tensor(np.array([1] * len(A_list)),
                                 dtype=mindspore.float32) * A_list_tensor) * mindspore.ops.log(
        mindspore.Tensor(np.array([1] * len(A_list)), dtype=mindspore.float32) + A_list_tensor)
    delta_list_tensor2 = mindspore.Tensor(np.array(delta_list), dtype=mindspore.float32)
    client_utility_list_tensor -= mindspore.ops.exp(delta_list_tensor2 * F_list_tensor)
    client_utility_list_tensor += mindspore.Tensor(np.array([5] * len(delta_list)), dtype=mindspore.float32)


    # client_utility_list = []
    # for i in range(num):
    #     client_utility_list.append(
    #         r1 * (A_list[i] / T_list[i]) + r2 * F_list[i] - gamma_list[i] * (1 + A_list[i]) * sympy.log(
    #             1 + A_list[i]) - sympy.exp(delta_list[i] * F_list[i]) + 5)

    res_sum = mindspore.Tensor(np.array([alpha]), dtype=mindspore.float32) * mindspore.Tensor.sum(A_list_tensor)
    res_sum += mindspore.Tensor(np.array([beta]), dtype=mindspore.float32) * mindspore.Tensor.sum(F_list_tensor)
    res_sum *= 1 / num
    res_sum -= mindspore.ops.max(T_list_tensor)[1]
    res_sum -= sum_cost
    print("**********************************************************")
    print("utility_max_ms")
    print(res_sum)
    print(client_utility_list_tensor)
    return res_sum[0], list(client_utility_list_tensor.asnumpy())



def utility_random_ms(gamma_list, delta_list, T_list, num):
    A_list = []

    r1 = cal_r1_random_ms(gamma_list, T_list)
    r2 = cal_r2_random_ms(delta_list)
    for i in range(num):
        A_list.append(cal_A_ms(gamma_list[i], 2, r1))

    A_list_tensor = mindspore.Tensor(np.array(A_list), dtype=mindspore.float32)

    delta_list_tensor = mindspore.Tensor(np.array(delta_list), dtype=mindspore.float32)
    tensor_1 = mindspore.Tensor(np.array([1] * len(delta_list)), dtype=mindspore.float32)
    tensor_r2 = mindspore.Tensor(np.array([r2] * len(delta_list)), dtype=mindspore.float32)
    tensor_1 = tensor_1 / delta_list_tensor
    tensor_r2_log = mindspore.ops.log(tensor_r2 / delta_list_tensor)
    F_list_tensor = tensor_1 * tensor_r2_log

    T_list_tensor = mindspore.Tensor(np.array(T_list), dtype=mindspore.float32)
    tensor_r1 = mindspore.Tensor(np.array([r1] * len(A_list)), dtype=mindspore.float32)
    sum_tensor = tensor_r1 * A_list_tensor / T_list_tensor
    sum_tensor += tensor_r2 * F_list_tensor

    sum_cost = mindspore.Tensor.sum(sum_tensor)

    # print('A', alpha * sum(A_list))
    # print('F', beta * sum(F_list))
    # print('Cost', sum_cost)
    # print('r1', r1)
    # print('A', A_list)

    client_utility_list_tensor = mindspore.Tensor(np.array([r1] * len(A_list)), dtype=mindspore.float32) * A_list_tensor / T_list_tensor
    client_utility_list_tensor += mindspore.Tensor(np.array([r2] * len(A_list)),  dtype=mindspore.float32) * F_list_tensor
    client_utility_list_tensor -= mindspore.Tensor(np.array(gamma_list), dtype=mindspore.float32) * (mindspore.Tensor(np.array([1] * len(A_list)),  dtype=mindspore.float32) * A_list_tensor) * mindspore.ops.log( mindspore.Tensor(np.array([1] * len(A_list)), dtype=mindspore.float32) + A_list_tensor)
    delta_list_tensor2 = mindspore.Tensor(np.array(delta_list), dtype=mindspore.float32)
    client_utility_list_tensor -= mindspore.ops.exp(delta_list_tensor2 * F_list_tensor)
    client_utility_list_tensor += mindspore.Tensor(np.array([5] * len(delta_list)), dtype=mindspore.float32)

    res_sum = mindspore.Tensor(np.array([alpha]), dtype=mindspore.float32) * mindspore.Tensor.sum(A_list_tensor)
    res_sum += mindspore.Tensor(np.array([beta]), dtype=mindspore.float32) * mindspore.Tensor.sum(F_list_tensor)
    res_sum *= 1 / num
    res_sum -= mindspore.ops.max(T_list_tensor)[1]
    res_sum -= sum_cost
    print("**********************************************************")
    print("utility_random_ms")
    print(res_sum)
    print(client_utility_list_tensor)
    return res_sum[0], list(client_utility_list_tensor.asnumpy())

# impact of gamma

num = 5
delta_list = [1]*num
T_list = [2]*num
gamma_list_list = []
for j in range(6):
    gamma_list_list.append([random.uniform(j+1, j+5) for i in range(num)])


##### iFedCrowd
Utility_list_gamma = []
r1_list_gamma = []
client_utility_list_list = []
for j in range(6):
    res = utility_ms(gamma_list_list[j], delta_list, T_list, num)

    Utility_list_gamma.append(res[0])
    client_utility_list_list.append(res[1])

    r1_list_gamma.append(cal_r1_ms(gamma_list_list[j], T_list))


client_utility_avg_gamma = []
for utility_list in client_utility_list_list:
    utility_list_tensor = mindspore.Tensor(np.array(utility_list), dtype=mindspore.float32)
    utility_list_tensor = mindspore.Tensor.sum(utility_list_tensor) / len(utility_list)
    client_utility_avg_gamma.append(utility_list_tensor)


##### max
Utility_list_gamma_max = []
r1_list_gamma_max = []
client_utility_list_list_max = []
for j in range(6):
    res = utility_max_ms(gamma_list_list[j], delta_list, T_list, num)

    Utility_list_gamma_max.append(res[0])
    client_utility_list_list_max.append(res[1])

    r1_list_gamma_max.append(cal_r1_max_ms(gamma_list_list[j], T_list))


client_utility_avg_gamma_max = []
for utility_list in client_utility_list_list_max:
    utility_list_tensor = mindspore.Tensor(np.array(utility_list), dtype=mindspore.float32)
    utility_list_tensor = mindspore.Tensor.sum(utility_list_tensor) / len(utility_list)
    client_utility_avg_gamma_max.append(utility_list_tensor)


##### random
Utility_list_gamma_random = []
r1_list_gamma_random = []
client_utility_list_list_random = []
for j in range(6):
    res = utility_random_ms(gamma_list_list[j], delta_list, T_list, num)

    Utility_list_gamma_random.append(res[0])
    client_utility_list_list_random.append(res[1])

    r1_list_gamma_random.append(cal_r1_random_ms(gamma_list_list[j], T_list))


client_utility_avg_gamma_random = []
for utility_list in client_utility_list_list_random:
    utility_list_tensor = mindspore.Tensor(np.array(utility_list), dtype=mindspore.float32)
    utility_list_tensor = mindspore.Tensor.sum(utility_list_tensor) / len(utility_list)
    client_utility_avg_gamma_random.append(utility_list_tensor)



########################################################################################################################
#impact of delta

num = 5
gamma_list = [random.uniform(1, 5) for i in range(num)]
T_list = [2]*num
delta_list_list = []
for j in range(6):
    delta_list_list.append([random.uniform(j+1, j+2) for i in range(num)])


########ours
Utility_list_delta = []
r2_list_delta = []
client_utility_list_list = []
for j in range(6):
    res = utility_ms(gamma_list, delta_list_list[j], T_list, num)

    Utility_list_delta.append(res[0])
    client_utility_list_list.append(res[1])

    r2_list_delta.append(cal_r2_ms(delta_list_list[j]))

client_utility_avg_delta = []
for utility_list in client_utility_list_list:
    utility_list_tensor = mindspore.Tensor(np.array(utility_list), dtype=mindspore.float32)
    utility_list_tensor = mindspore.Tensor.sum(utility_list_tensor) / len(utility_list)
    client_utility_avg_delta.append(utility_list_tensor)


########## max
Utility_list_delta_max = []
r2_list_delta_max = []
client_utility_list_list_max = []
for j in range(6):
    res = utility_max_ms(gamma_list, delta_list_list[j], T_list, num)

    Utility_list_delta_max.append(res[0])
    client_utility_list_list_max.append(res[1])

    r2_list_delta_max.append(cal_r2_max_ms(delta_list_list[j]))

client_utility_avg_delta_max = []
for utility_list in client_utility_list_list_max:
    utility_list_tensor = mindspore.Tensor(np.array(utility_list), dtype=mindspore.float32)
    utility_list_tensor = mindspore.Tensor.sum(utility_list_tensor) / len(utility_list)
    client_utility_avg_delta_max.append(utility_list_tensor)



########## random
Utility_list_delta_random = []
r2_list_delta_random = []
client_utility_list_list_random = []
for j in range(6):
    res = utility_random_ms(gamma_list, delta_list_list[j], T_list, num)

    Utility_list_delta_random.append(res[0])
    client_utility_list_list_random.append(res[1])

    r2_list_delta_random.append(cal_r2_random_ms(delta_list_list[j]))

client_utility_avg_delta_random = []
for utility_list in client_utility_list_list_random:
    utility_list_tensor = mindspore.Tensor(np.array(utility_list), dtype=mindspore.float32)
    utility_list_tensor = mindspore.Tensor.sum(utility_list_tensor) / len(utility_list)
    client_utility_avg_delta_random.append(utility_list_tensor)






plt.rc('font', family='Times New Roman')
fig = plt.figure(figsize=(8, 3.5))
ax = fig.add_subplot(1, 2, 1)
# plt.subplot(121)
# labels = ['MNIST', 'CIFAR-10', 'ActPred', 'ActTrack']
gamma = [1, 2, 3, 4, 5, 6]
gamma_x = np.array(gamma)
# FedCrowd = [0.935, 0.510, 0.511, 0.480]
# FedCrowd_alpha = [0.914, 0.495, 0.505, 0.465]
# FedCrowd_beta = [0.899, 0.466, 0.504, 0.441]
# FedCrowd_gamma = [0.914, 0.483, 0.502, 0.471]

# x = np.arange(len(labels))  # the label locations
width = 0.2  # the width of the bars

# fig, ax = plt.subplots()
# print(Utility_list_gamma)


r1_list_gamma_random = [13.731745719909668, 9.01563917358592, 15.10265823517395, 25.755151748657227, 13.00537385725195, 20.11395754865296]
r1_list_gamma_max = [14.864921569824219, 17.45597267150879, 21.47945213317871, 26.349014282226562, 29.652387619018555, 27.03360939025879]
r1_list_gamma = [8.77946949005127, 10.309786796569824, 12.686120986938477, 15.562165260314941, 17.513193130493164, 15.966498374938965]
# plt.bar(gamma, Utility_list_gamma, width, color=[214/255,39/255,40/255])
rects1 = ax.bar(gamma_x-width, r1_list_gamma,  width, label='iFedCrowd', color=[214/255,39/255,40/255])
rects2 = ax.bar(gamma_x,  r1_list_gamma_random, width, label='Random', color=[255/255,127/255,14/255])
rects3 = ax.bar(gamma_x + width, r1_list_gamma_max, width, label='MAX', color=[3/255,140/255,101/255])
# rects2 = ax.bar(x - width/2, FedCrowd_alpha, width, label='FedTA(w/o '+r'$\alpha$)', color=[255/255,127/255,14/255])
# rects3 = ax.bar(x + width/2, FedCrowd_beta, width, label='FedTA(w/o '+ r'$\beta$)', color=[3/255,140/255,101/255])
# rects4 = ax.bar(x + 1.5*width, FedCrowd_gamma, width, label='FedTA(w/o '+ r'$\gamma$)', color=[31/255,119/255,180/255])

# Add some text for labels, title and custom x-axis tick labels, etc.
plt.ylabel(r'Reward rate ($r_1$)', fontsize=20)
plt.xlabel(r'$\Gamma$', fontsize=20)
# ax.set_title('Ablation experiment', fontsize=20)
# ax.set_xticks(gamma_k, fontsize=20)
# ax.set_xticklabels(labels, fontsize=20)
# ax.legend(fontsize=18, labelspacing=0.4, columnspacing=0.4)
plt.xticks(gamma_x, fontsize=20)
plt.yticks(fontsize=20)


# plt.subplot(122)
ax = fig.add_subplot(1, 2, 2)
# labels = ['MNIST', 'CIFAR-10', 'ActPred', 'ActTrack']
delta = [0, 1, 2, 3, 4, 5]
delta_x = np.array(delta)
# FedCrowd = [0.935, 0.510, 0.511, 0.480]
# FedCrowd_alpha = [0.914, 0.495, 0.505, 0.465]
# FedCrowd_beta = [0.899, 0.466, 0.504, 0.441]
# FedCrowd_gamma = [0.914, 0.483, 0.502, 0.471]

# x = np.arange(len(labels))  # the label locations
width = 0.2  # the width of the bars
#
# fig, ax = plt.subplots()
# print(Utility_list_delta)
# plt.bar(delta, Utility_list_delta, width, color=[255/255,127/255,14/255])


r2_list_delta = [1.83003568649292, 2.882478952407837, 3.7974042892456055, 4.703040599822998, 5.7784423828125, 6.983187675476074]
r2_list_delta_random = [2.196043014526367, 3.458974838256836, 2.24748253960153, 4.19364869251328, 6.934131145477295, 6.04856128492748]
r2_list_delta_max = [3.111060857772827, 4.900214195251465, 6.455587387084961, 7.995169162750244, 9.823351860046387, 11.871418952941895]

rects1 = ax.bar(delta_x - width, r2_list_delta, width, label='iFedCrowd', color=[214/255,39/255,40/255])
rects2 = ax.bar(delta_x, r2_list_delta_random, width, label='Random', color=[255/255,127/255,14/255])
rects3 = ax.bar(delta_x + width, r2_list_delta_max, width, label='MAX', color=[3/255,140/255,101/255])
# rects2 = ax.bar(x - width/2, FedCrowd_alpha, width, label='FedTA(w/o '+r'$\alpha$)', color=[255/255,127/255,14/255])
# rects3 = ax.bar(x + width/2, FedCrowd_beta, width, label='FedTA(w/o '+ r'$\beta$)', color=[3/255,140/255,101/255])
# rects4 = ax.bar(x + 1.5*width, FedCrowd_gamma, width, label='FedTA(w/o '+ r'$\gamma$)', color=[31/255,119/255,180/255])

# Add some text for labels, title and custom x-axis tick labels, etc.
plt.ylabel(r'Reward rate ($r_2$)', fontsize=20)
plt.xlabel(r'$\Delta$', fontsize=20)
# ax.set_title('Ablation experiment', fontsize=20)
# ax.set_xticks(gamma_k, fontsize=20)
# ax.set_xticklabels(labels, fontsize=20)
ax.legend(fontsize=20, labelspacing=0.2, columnspacing=0.2)
plt.xticks(delta, fontsize=20)
plt.yticks(fontsize=20)

plt.subplots_adjust(left=0.09, bottom=0.174, right=0.99, top=0.98, wspace=0.25, hspace=0.2)
plt.show()






