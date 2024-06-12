import os
# from multiprocessing import Pool

runs = 4

# use_goat, use_formation, use_self, lr, warmup
use_goat_list = [1]
use_formation_list = [0]
use_self_list = [0]
lr_list = [1e-5, 3e-5, 7e-6]
train_batch = [4, 2]
weight_decay = [1e-4]
dropout = [0, 0.2]


config_list = []
for train_bs in train_batch:
    for lr in lr_list:
        for dp in dropout:
            for wd in weight_decay:
                config_list.append([train_bs, lr, wd, dp])


def func(i):
    with open('train.sh', 'a') as f:
        f.write(
            f'python -u main.py --num_workers={4 if config_list[i][0] != 8 else 2} --attn_drop={config_list[i][3]} '
            f'--lr={config_list[i][1]} --weight_decay={config_list[i][2]} --train_batch_size={config_list[i][0]} '
            f'--use_i3d_bb=1 --use_swin_bb=0' + '\n')
        # f.write(
        #     f'python -u main.py  --num_workers={4 if config_list[i][0] != 8 else 2} '
        #     f'--lr={config_list[i][1]} --weight_decay={config_list[i][2]} --train_batch_size={config_list[i][0]} --attn_drop={config_list[i][3]} '
        #     f'--use_i3d_bb=0 --use_swin_bb=1' + '\n')


# pool = Pool(runs)
# pool.map(func, range(runs))
# pool.close()
# pool.join()
for i in range(len(config_list)):
    func(i)
