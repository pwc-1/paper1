import numpy as np
import sympy
import random
import matplotlib.pyplot as plt

random.seed(1)
alpha = 80
beta = 50

def A1_cal(gamma_list, T_list):
    A1_list = []
    for i in range(len(gamma_list)):
        A1_list.append(gamma_list[i]*T_list[i])
    return max(A1_list)

def A1_cal_max(gamma_list, T_list):
    A1_list = []
    for i in range(len(gamma_list)):
        A1_list.append((1+sympy.log(2))*gamma_list[i] * T_list[i])
    return max(A1_list)

def A1_cal_random(gamma_list, T_list):
    A1_list_min = []
    A1_list_max = []
    for i in range(len(gamma_list)):
        A1_list_max.append((1+sympy.log(2))*gamma_list[i] * T_list[i])
        A1_list_min.append(gamma_list[i] * T_list[i])
    return 1.2*max(A1_list_min)

def B2_cal(delta_list):
    return max(delta_list)

def B2_cal_max(delta_list):
    return 1.8*max(delta_list)

def B2_cal_random(delta_list):
    return 1.2*max(delta_list)

def C1_cal(gamma, T, A1):
    h = A1/(gamma * T) - 1
    a = min([sympy.exp(h) - 1, 1])
    return a



def utility(gamma_list, delta_list, T_list, num):
    A_list = []

    F_list = []
    A1 = A1_cal(gamma_list, T_list)
    B2 = B2_cal(delta_list)
    for i in range(num):
        A_list.append(C1_cal(gamma_list[i], 2, A1))
        F_list.append((1/delta_list[i]) * np.log(B2/delta_list[i]))

    sum_cost = 0
    for i in range(num):
        sum_cost += A1* A_list[i] / T_list[i] + B2 * F_list[i]


    client_utility_list = []
    for i in range(num):
        client_utility_list.append(A1*(A_list[i]/T_list[i]) + B2*F_list[i] - gamma_list[i] * (1+A_list[i])*sympy.log(1 + A_list[i]) -sympy.exp(delta_list[i]*F_list[i]))

    return (1/num) * (alpha * sum(A_list) + beta * sum(F_list)) - max(T_list) -sum_cost, client_utility_list, A1, B2



def utility_max(gamma_list, delta_list, T_list, num):
    A_list = []
    F_list = []
    A1 = A1_cal_max(gamma_list, T_list)
    B2 = B2_cal_max(delta_list)
    for i in range(num):
        A_list.append(C1_cal(gamma_list[i], 2, A1))
        F_list.append((1/delta_list[i]) * np.log(B2/delta_list[i]))

    sum_cost = 0
    for i in range(num):
        sum_cost += A1* A_list[i] / T_list[i] + B2 * F_list[i]


    client_utility_list = []
    for i in range(num):
        client_utility_list.append(A1*(A_list[i]/T_list[i]) + B2*F_list[i] - gamma_list[i] * (1+A_list[i])*sympy.log(1 + A_list[i]) -sympy.exp(delta_list[i]*F_list[i]))

    return (1/num) * (alpha * sum(A_list) + beta * sum(F_list)) - max(T_list) -sum_cost, client_utility_list, A1, B2


def utility_random(gamma_list, delta_list, T_list, num):
    A_list = []
    F_list = []
    A1 = A1_cal_random(gamma_list, T_list)
    B2 = B2_cal_random(delta_list)
    for i in range(num):
        A_list.append(C1_cal(gamma_list[i], 2, A1))
        F_list.append((1/delta_list[i]) * np.log(B2/delta_list[i]))

    sum_cost = 0
    for i in range(num):
        sum_cost += A1* A_list[i] / T_list[i] + B2 * F_list[i]


    client_utility_list = []
    for i in range(num):
        client_utility_list.append(A1*(A_list[i]/T_list[i]) + B2*F_list[i] - gamma_list[i] * (1+A_list[i])*sympy.log(1 + A_list[i]) -sympy.exp(delta_list[i]*F_list[i]))

    return (1/num) * (alpha * sum(A_list) + beta * sum(F_list)) - max(T_list) -sum_cost, client_utility_list, A1, B2



####################################################################################
# impact of worker number

num_list = [5, 10, 15, 20, 25, 30]
gamma_range = [3, 5]
delta_range = [2, 4]
T_min = 2


##### iFedCrowd
Utility_list_iFed = []
A1_list_iFed = []
B2_list_iFed = []
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
    res = utility(gamma_list, delta_list, T_list, worker_number)
    Utility_list_iFed.append(res[0])
    client_utility_list_list_iFed.append(res[1])
    A1_list_iFed.append(res[2])
    B2_list_iFed.append(res[3])

client_utility_avg_iFed = []
for utility_list in client_utility_list_list_iFed:
    client_utility_avg_iFed.append(sum(utility_list)/len(utility_list))


##### MAX
Utility_list_MAX = []
A1_list_MAX = []
B2_list_MAX = []
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
    res = utility_max(gamma_list, delta_list, T_list, worker_number)
    Utility_list_MAX.append(res[0])
    client_utility_list_list_MAX.append(res[1])
    A1_list_MAX.append(res[2])
    B2_list_MAX.append(res[3])

client_utility_avg_MAX = []
for utility_list in client_utility_list_list_MAX:
    client_utility_avg_MAX.append(sum(utility_list)/len(utility_list))



##### Random
Utility_list_random = []
A1_list_random = []
B2_list_random = []
client_utility_list_list_random = []
forf j in range(6):
    worker_number = num_list[j]
    gamma_list = []
    delta_list = []
    T_list = []
    for i in range(worker_number):
        gamma_list.append(random.uniform(gamma_range[0], gamma_range[1]))
        delta_list.append(random.uniform(delta_range[0], delta_range[1]))
        T_list.append(T_min)
    res = utility_random(gamma_list, delta_list, T_list, worker_number)
    Utility_list_random.append(res[0])
    client_utility_list_list_random.append(res[1])
    A1_list_random.append(res[2])
    B2_list_random.append(res[3])

client_utility_avg_random = []
for utility_list in client_utility_list_list_random:
    client_utility_avg_random.append(sum(utility_list)/len(utility_list))
