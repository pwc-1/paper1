import mindspore.nn as nn
import mindspore.ops as P
import mindspore.numpy as mnp


class DAE_ZINB(nn.Cell):

    def __init__(self, in_dim, hidden1, hidden2, hidden3, z_emb_size, dropout_rate):
        super(DAE_ZINB, self).__init__()

        self.in_dim = in_dim

        self.fc_encoder = nn.SequentialCell(
            nn.Dense(self.in_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),

            nn.Dense(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),

            nn.Dense(hidden2, hidden3),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),

            nn.Dense(hidden3, z_emb_size),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate)
        )

        self.fc_decoder = nn.SequentialCell(
            nn.Dense(z_emb_size, hidden3),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),

            nn.Dense(hidden3, hidden2),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),

            nn.Dense(hidden2, hidden1),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate)
        )

        self.decoder_scale = nn.Dense(hidden1, self.in_dim)
        self.decoder_r = nn.Dense(hidden1, self.in_dim)
        self.dropout = nn.Dense(hidden1, self.in_dim)

        self.softmax = nn.Softmax(axis=1)
        self.exp = P.Exp()

    def construct(self, x, scale_factor=1.0):
        emb = self.fc_encoder(x)

        latent = self.fc_decoder(emb)

        normalized_x = self.softmax(self.decoder_scale(latent))

        batch_size = normalized_x.shape[0]
        # scale_factor = scale_factor.expand([batch_size, 1])
        scale_factor = scale_factor * mnp.ones((batch_size, 1))
        scale_factor = scale_factor.tile((1, normalized_x.shape[1]))

        scale_x = self.exp(scale_factor) * normalized_x

        disper_x = self.exp(self.decoder_r(latent))
        dropout_rate = self.dropout(latent)

        return emb, normalized_x, dropout_rate, disper_x, scale_x

if __name__ == '__main__':
    import numpy as np
    import mindspore
    import mindspore.dataset as ds
    from mindspore import Tensor
    from mindspore import context
    from mindspore import nn
    from mindspore.train.model import Model

    # 创建测试数据
    X = np.random.rand(10, 5).astype(np.float32)

    # 转换为MindSpore Tensor
    input_data = Tensor(X)

    # 创建模型实例
    model = DAE_ZINB(5, 10, 8, 6, 4, 0.5)

    # 执行前向传播
    result = model(input_data)

    print(result)

