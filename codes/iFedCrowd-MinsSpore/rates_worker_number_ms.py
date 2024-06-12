# -*- coding: utf-8 -*-
import numpy as np
import sympy
import random
import matplotlib.pyplot as plt
import mindspore
from mindspore.common.initializer import One, Normal



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

    print(res_sum, r1, r2)
    print(client_utility_list_tensor)
    return res_sum[0], list(client_utility_list_tensor.asnumpy()), r1, r2



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

    print(res_sum, r1, r2)
    print(client_utility_list_tensor)

    return res_sum[0], list(client_utility_list_tensor.asnumpy()), r1, r2





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

    print(res_sum, r1, r2)
    print(client_utility_list_tensor)

    return res_sum[0], list(client_utility_list_tensor.asnumpy()), r1, r2

####################################################################################
# impact of worker number

num_list = [5, 10, 15, 20, 25, 30]
gamma_range = [3, 5]
delta_range = [2, 4]
T_min = 2


##### iFedCrowd
Utility_list_iFed = []
r1_list_iFed = []
r2_list_iFed = []
client_utility_list_list_iFed = []
for j in range(6):
    worker_number = num_list[j]
    gamma_list = []
    delta_list = []
    T_list = []

    for i in range(worker_number):
        gamma_list.append(random.uniform(gamma_range[0], gamma_range[1]))
        delta_list.append(random.uniform(delta_range[0], delta_range[1]))
        T_list.append(T_min)
    res = utility_ms(gamma_list, delta_list, T_list, worker_number)
    Utility_list_iFed.append(res[0])
    client_utility_list_list_iFed.append(res[1])
    r1_list_iFed.append(res[2])
    r2_list_iFed.append(res[3])

client_utility_avg_iFed = []
for utility_list in client_utility_list_list_iFed:
    utility_list_tensor = mindspore.Tensor(np.array(utility_list), dtype=mindspore.float32)
    utility_list_tensor = mindspore.Tensor.sum(utility_list_tensor) / len(utility_list)
    client_utility_avg_iFed.append(utility_list_tensor)


##### MAX
Utility_list_MAX = []
r1_list_MAX = []
r2_list_MAX = []
client_utility_list_list_MAX = []
for j in range(6):
    worker_number = num_list[j]
    gamma_list = []
    delta_list = []
    T_list = []
    for i in range(worker_number):
        gamma_list.append(random.uniform(gamma_range[0], gamma_range[1]))
        delta_list.append(random.uniform(delta_range[0], delta_range[1]))
        T_list.append(T_min)
    res = utility_max_ms(gamma_list, delta_list, T_list, worker_number)
    Utility_list_MAX.append(res[0])
    client_utility_list_list_MAX.append(res[1])
    r1_list_MAX.append(res[2])
    r2_list_MAX.append(res[3])

client_utility_avg_MAX = []
for utility_list in client_utility_list_list_MAX:
    utility_list_tensor = mindspore.Tensor(np.array(utility_list), dtype=mindspore.float32)
    utility_list_tensor = mindspore.Tensor.sum(utility_list_tensor) / len(utility_list)
    client_utility_avg_iFed.append(utility_list_tensor)



##### Random
Utility_list_random = []
r1_list_random = []
r2_list_random = []
client_utility_list_list_random = []
for j in range(6):
    worker_number = num_list[j]
    gamma_list = []
    delta_list = []
    T_list = []
    for i in range(worker_number):
        gamma_list.append(random.uniform(gamma_range[0], gamma_range[1]))
        delta_list.append(random.uniform(delta_range[0], delta_range[1]))
        T_list.append(T_min)
    res = utility_random_ms(gamma_list, delta_list, T_list, worker_number)
    Utility_list_random.append(res[0])
    client_utility_list_list_random.append(res[1])
    r1_list_random.append(res[2])
    r2_list_random.append(res[3])

client_utility_avg_random = []
for utility_list in client_utility_list_list_random:
    utility_list_tensor = mindspore.Tensor(np.array(utility_list), dtype=mindspore.float32)
    utility_list_tensor = mindspore.Tensor.sum(utility_list_tensor) / len(utility_list)
    client_utility_avg_random.append(utility_list_tensor)


r1_list_iFed = [12.117993596505551, 10.609973987645558, 9.405389588565718, 9.073554699462472, 8.80848495236758, 8.506074623238788]
r1_list_random = [11.632715225219727, 14.336898803710938, 12.039434432983398, 10.1851224899292, 14.450881004333496, 12.873027801513672]
r1_list_MAX = [16.472192764282227, 16.62200164794922, 16.851259231567383, 16.81887435913086, 16.587600708007812, 16.68348503112793]
plt.rc('font', family='Times New Roman')
plt.figure(figsize=(8, 3.5))
plt.subplot(121)
plt.plot(num_list, r1_list_iFed, linestyle='-', label="iFedCrowd", color=[214/255,39/255,40/255], marker='s', linewidth=3, markersize=10)
plt.plot(num_list, r1_list_random, label="Random", color=[255/255,127/255,14/255], marker='v', linewidth=3, markersize=10)
plt.plot(num_list, r1_list_MAX, label="MAX", color=[3/255,140/255,101/255], marker='o', linewidth=3, markersize=10)


plt.xticks(num_list, fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Number of workers', fontsize=20)
plt.ylabel(r'Reward rate ($r_1$)', fontsize=20)
# plt.legend(fontsize=20)
plt.legend(fontsize=18, labelspacing=0.1, columnspacing=0.2, loc='upper center', bbox_to_anchor=(0.55, 0.95))


r2_list_iFed = [5.694867473874465, 4.890541391107845, 4.31989305759058, 3.9641532750770683, 3.7996370007120404, 3.5511890356357667]
r2_list_random = [4.121167832082025, 4.786346537602698, 4.533241965602714, 5.736447482992129, 5.044471388281418, 4.791279115945162]
r2_list_MAX = [6.750728130340576, 6.416701316833496, 6.409204483032227, 6.698897838592529, 6.748645782470703, 6.52020263671875]

plt.subplot(122)
plt.plot(num_list, r2_list_iFed, linestyle='-', label="iFedCrowd", color=[214/255,39/255,40/255], marker='s', linewidth=3, markersize=10)
plt.plot(num_list, r2_list_random, label="Random", color=[255/255,127/255,14/255], marker='v', linewidth=3, markersize=10)
plt.plot(num_list, r2_list_MAX, label="MAX", color=[3/255,140/255,101/255], marker='o', linewidth=3, markersize=10)

plt.xticks(num_list, fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Number of workers ', fontsize=20)
plt.ylabel(r'Reward rate ($r_2$)', fontsize=20)
# plt.legend(fontsize=20)
# plt.legend(fontsize=20, labelspacing=0.1, columnspacing=0.2)

plt.subplots_adjust(left=0.095, bottom=0.18, right=0.99, top=0.988, wspace=0.267, hspace=0.2)
# plt.title("Task Assignment", fontsize=20)
plt.show()







