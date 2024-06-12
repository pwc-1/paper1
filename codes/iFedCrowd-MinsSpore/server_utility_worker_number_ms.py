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
    return float(1.2*mindspore.ops.max(r1_list_min, axis=0)[1])


def cal_r2_ms(delta_list):
    delta_list = mindspore.Tensor(np.array(delta_list), dtype=mindspore.float32)
    return float(mindspore.ops.max(delta_list, axis=0)[1])


def cal_r2_max_ms(delta_list):
    rate = mindspore.Tensor(np.array([1.8]*len(delta_list)), dtype=mindspore.float32)
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
    client_utility_list_tensor -= mindspore.Tensor(np.array(gamma_list), dtype=mindspore.float32) * (mindspore.Tensor(np.array([1]*len(A_list)), dtype=mindspore.float32) + A_list_tensor) * mindspore.ops.log(mindspore.Tensor(np.array([1]*len(A_list)), dtype=mindspore.float32)+A_list_tensor)
    delta_list_tensor2 = mindspore.Tensor(np.array(delta_list), dtype=mindspore.float32)
    client_utility_list_tensor -= mindspore.ops.exp(delta_list_tensor2*F_list_tensor)
    # client_utility_list_tensor += mindspore.Tensor(np.array([5]*len(delta_list)), dtype=mindspore.float32)

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
                                 dtype=mindspore.float32) + A_list_tensor) * mindspore.ops.log(
        mindspore.Tensor(np.array([1] * len(A_list)), dtype=mindspore.float32) + A_list_tensor)
    delta_list_tensor2 = mindspore.Tensor(np.array(delta_list), dtype=mindspore.float32)
    client_utility_list_tensor -= mindspore.ops.exp(delta_list_tensor2 * F_list_tensor)
    # client_utility_list_tensor += mindspore.Tensor(np.array([5] * len(delta_list)), dtype=mindspore.float32)


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
    client_utility_list_tensor -= mindspore.Tensor(np.array(gamma_list), dtype=mindspore.float32) * (mindspore.Tensor(np.array([1] * len(A_list)),  dtype=mindspore.float32) + A_list_tensor) * mindspore.ops.log( mindspore.Tensor(np.array([1] * len(A_list)), dtype=mindspore.float32) + A_list_tensor)
    delta_list_tensor2 = mindspore.Tensor(np.array(delta_list), dtype=mindspore.float32)
    client_utility_list_tensor -= mindspore.ops.exp(delta_list_tensor2 * F_list_tensor)
    # client_utility_list_tensor += mindspore.Tensor(np.array([5] * len(delta_list)), dtype=mindspore.float32)

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
    client_utility_avg_iFed.append(float(utility_list_tensor))


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
    client_utility_avg_MAX.append(float(utility_list_tensor))



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
    client_utility_avg_random.append(float(utility_list_tensor))


Utility_list_iFed = [15.1745116035077, 35.0935567130499, 52.1168163778054, 63.3330490764691, 71.9197041197491, 76.9152374604507]
Utility_list_random = [32.7189512383843, 10.8653842979947, -11.3949793897476, 13.9789639406879, -50.8830660486202, -30.6401937824266]
Utility_list_MAX = [5.483829644297, -9.536302873239201, -61.4798331922453, -114.75717140420011, -169.2718380329042, -215.9588330355839]

plt.rc('font', family='Times New Roman')
plt.figure(figsize=(8, 3.5))
plt.subplot(121)
plt.plot(num_list, Utility_list_iFed, linestyle='-', label="iFedCrowd", color=[214/255,39/255,40/255], marker='s', linewidth=3, markersize=10)
plt.plot(num_list, Utility_list_random, label="Random", color=[255/255,127/255,14/255], marker='v', linewidth=3, markersize=10)
plt.plot(num_list, Utility_list_MAX, label="MAX", color=[3/255,140/255,101/255], marker='o', linewidth=3, markersize=10)


client_utility_avg_iFed = [5.257878624230901, 4.38134392920638, 3.1818480446938713, 2.474695915139209, 2.368367547755704, 2.101698723349419]
client_utility_avg_random = [4.518997436522445, 5.011469576231625, 4.890575954647184, 5.413669537168218, 3.8292677562791684, 4.150656391209376]
client_utility_avg_MAX = [5.5247156620025635, 5.515056133270264, 5.498727083206177, 5.463884592056274, 5.452575302124023, 5.305729402749104875]

plt.xticks(num_list, fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Number of workers', fontsize=20)
plt.ylabel('Utility of task publisher', fontsize=20)
plt.legend(fontsize=18, labelspacing=0.1, columnspacing=0.2, loc='upper center', bbox_to_anchor=(0.35, 0.38))


plt.subplot(122)
plt.plot(num_list, client_utility_avg_iFed, linestyle='-', label="iFedCrowd", color=[214/255,39/255,40/255], marker='s', linewidth=3, markersize=10)
plt.plot(num_list, client_utility_avg_random, label="Random", color=[255/255,127/255,14/255], marker='v', linewidth=3, markersize=10)
plt.plot(num_list, client_utility_avg_MAX, label="MAX", color=[3/255,140/255,101/255], marker='o', linewidth=3, markersize=10)

plt.xticks(num_list, fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Number of workers ', fontsize=20)
plt.ylabel('Average utility of workers', fontsize=20)
# plt.legend(fontsize=20)

plt.subplots_adjust(left=0.124, bottom=0.18, right=0.99, top=0.98, wspace=0.295, hspace=0.2)
plt.show()