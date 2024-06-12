import mindspore.nn as nn


class Linear_For_Backbone(nn.Cell):
    def __init__(self, args):
        super(Linear_For_Backbone, self).__init__()
        if args.use_swin_bb:
            input_dim = 1536
        else:
            input_dim = 768

        self.linear = nn.Dense(input_dim, 1024)
        self.relu = nn.ReLU()

    def construct(self, x):
        return self.relu(self.linear(x))
