import os
from multiprocessing import Pool

runs = 16

config_list = [[3e-5, 1024, 0],
               [1e-5, 1024, 0],
               [3e-6, 1024, 0],
               [1e-6, 1024, 0],
               [3e-7, 1024, 0],
               [1e-7, 1024, 0],
               [3e-8, 1024, 0],
               [1e-8, 1024, 0],
               [1e-8, 1024, 1],
               [1e-8, 1024, 1],
               [1e-8, 1024, 1],
               [1e-8, 1024, 1],
               [1e-8, 1024, 1],
               [1e-8, 1024, 1],
               [1e-8, 1024, 1],
               [1e-8, 1024, 1]]


def func(i):
    with open('train.sh', 'a') as f:
        f.write(f'python main.py --lr={config_list[i][0]} --warmup={config_list[i][2]} --max_epoch=150 --use_i3d_bb=0 --use_swin_bb=1' + '\n')
        f.write(f'python main.py --lr={config_list[i][0]} --warmup={config_list[i][2]} --max_epoch=150 --use_i3d_bb=1 --use_swin_bb=0' + '\n')


for i in range(runs):
    func(i)
