import os
import warnings
import mindspore.context as context

class DefaultConfig(object):
    load_img_path = None  # load model path
    load_txt_path = None

    # data parameters
    data_path = '/home/zjgao/NUSWIDE21_deep.mat'
    pretrain_model_path = './data/imagenet-vgg-f.mat'
    training_size = 13375
    query_size = 2000
    database_size = 193834
    batch_size = 64
    device = 'GPU' if context.get_context("device_target") == "GPU" else 'CPU'
    # hyper-parameters
    max_epoch = 150
    gamma = 1
    eta = 1
    bit = 64  # final binary code length
    lr = 10 ** (-2)  # initial learning rate
    hash_lr = 10 ** (-2.5)
    use_gpu = True

    valid = True

    print_freq = 2  # print info every N epoch

    result_dir = 'result'

    def parse(self, kwargs):
        """
        update configuration by kwargs.
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Waning: opt has no attribute %s" % k)
            setattr(self, k, v)

        print('User config:')
        for k, v in self.__dict__.items():
            if not k.startswith('__'):
                print(k, getattr(self, k))


opt = DefaultConfig()