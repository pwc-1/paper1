# -*- coding: utf-8 -*-
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

    return (1/num) * (alpha * sum(A_list) + beta * sum(F_list)) - max(T_list) -sum_cost, client_utility_list



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

    return (1/num) * (alpha * sum(A_list) + beta * sum(F_list)) - max(T_list) -sum_cost, client_utility_list


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

    return (1/num) * (alpha * sum(A_list) + beta * sum(F_list)) - max(T_list) -sum_cost, client_utility_list



# impact of gamma

num = 5
delta_list = [1]*num
T_list = [2]*num
gamma_list_list = []
for j in range(6):
    gamma_list_list.append([random.uniform(j+1, j+5) for i in range(num)])


##### iFedCrowd
Utility_list_gamma = []
client_utility_list_list = []
for j in range(6):
    res = utility(gamma_list_list[j], delta_list, T_list, num)

    Utility_list_gamma.append(res[0])
    client_utility_list_list.append(res[1])

client_utility_avg_gamma = []
for utility_list in client_utility_list_list:
    client_utility_avg_gamma.append(sum(utility_list)/len(utility_list))


##### max
Utility_list_gamma_max = []
client_utility_list_list_max = []
for j in range(6):
    res = utility_max(gamma_list_list[j], delta_list, T_list, num)

    Utility_list_gamma_max.append(res[0])
    client_utility_list_list_max.append(res[1])



client_utility_avg_gamma_max = []
for utility_list in client_utility_list_list_max:
    client_utility_avg_gamma_max.append(sum(utility_list)/len(utility_list))


##### random
Utility_list_gamma_random = []
client_utility_list_list_random = []
for j in range(6):
    res = utility_random(gamma_list_list[j], delta_list, T_list, num)

    Utility_list_gamma_random.append(res[0])
    client_utility_list_list_random.append(res[1])


client_utility_avg_gamma_random = []
for utility_list in client_utility_list_list_random:
    client_utility_avg_gamma_random.append(sum(utility_list)/len(utility_list))



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
client_utility_list_list = []
for j in range(6):
    res = utility(gamma_list, delta_list_list[j], T_list, num)

    Utility_list_delta.append(res[0])
    client_utility_list_list.append(res[1])


client_utility_avg_delta = []
for utility_list in client_utility_list_list:
    client_utility_avg_delta.append(sum(utility_list) / len(utility_list))


########## max
Utility_list_delta_max = []
client_utility_list_list_max = []
for j in range(6):
    res = utility_max(gamma_list, delta_list_list[j], T_list, num)

    Utility_list_delta_max.append(res[0])
    client_utility_list_list_max.append(res[1])


client_utility_avg_delta_max = []
for utility_list in client_utility_list_list_max:
    client_utility_avg_delta_max.append(sum(utility_list) / len(utility_list))



########## random
Utility_list_delta_random = []
client_utility_list_list_random = []
for j in range(6):
    res = utility_random(gamma_list, delta_list_list[j], T_list, num)

    Utility_list_delta_random.append(res[0])
    client_utility_list_list_random.append(res[1])

client_utility_avg_delta_random = []
for utility_list in client_utility_list_list_random:
    client_utility_avg_delta_random.append(sum(utility_list) / len(utility_list))









