import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import UnivariateSpline

log_path = 'exp'
file_name = 'USDL_full_split3_7e-06_Mon Aug 14 23 00 04 2023.log'
state = 'test'
best_ep = 41
f_path = os.path.join(log_path, file_name)
loss_list = []

# show plt of rho
with open(f_path, 'r') as f:
    for line in f:
        if line.startswith('epoch') and line.split(',')[1].split(' ')[1] == state:
            loss_list.append(float(line.split(',')[3].split(':')[-1]) - 1)
    f.close()

# x = np.arange(len(loss_list))
# spl = UnivariateSpline(x, loss_list, s=0.2)
# x_smooth = np.linspace(0, len(loss_list)-1, 100)
# y_smooth = spl(x_smooth)

# s = pd.Series(y_smooth, name='rho')
s = pd.Series(loss_list, name='RL2*100')
sns.lineplot(data=s)
# title = state + ',' + file_name.split('_')[3]
title = state + ',' + '1e-05'
plt.title(title)
plt.show()
plt.savefig('curve.png')

# calculate mean and std
if state == 'test':
    adjacent_five = loss_list[best_ep - 2:best_ep + 3]
    print(adjacent_five)
    print('mean:', np.mean(adjacent_five))
    print('std:', np.std(adjacent_five))