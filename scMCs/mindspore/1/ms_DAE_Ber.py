
import mindspore.nn as nn


class DAE_Ber(nn.Cell):

    def __init__(self, in_dim, hidden1, hidden2, hidden3, hidden4, z_emb_size, dropout_rate):
        super(DAE_Ber, self).__init__()

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

            nn.Dense(hidden3, hidden4),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),

            nn.Dense(hidden4, z_emb_size),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate)
        )

        self.fc_decoder = nn.SequentialCell(
            nn.Dense(z_emb_size, hidden4),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),

            nn.Dense(hidden4, hidden3),
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

        self.sigmoid = nn.Sigmoid()

    def construct(self, x, scale_factor=1.0):
        emb = self.fc_encoder(x)

        latent = self.fc_decoder(emb)
        recon_x = self.decoder_scale(latent)
        Final_x = self.sigmoid(recon_x)

        return emb, Final_x


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
    input_data = Tensor(X)

    # 创建模型实例
    model = DAE_Ber(5, 10, 8, 6, 4, 2, 0.5)


    # 创建测试数据集
    dataset = ds.NumpySlicesDataset([input_data], shuffle=False)
    dataset = dataset.batch(1)

    # 创建模型
    model = Model(model)

    # 运行前向传播
    result = model.predict(input_data)

    # 打印输出
    print(result)
