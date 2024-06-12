import mindspore
from mindspore import nn, ops, Parameter, Tensor
from mindspore_model.transformer import transformer_Endcoder, transformer_Decoder

class q_net(nn.Cell):
    def __init__(self, num_variables, num_samples, embed_dim=128, hidden_dim=128, heads=4, dropout_rate=0.0) -> None:
        super(q_net, self).__init__()
        self.num_variables = num_variables
        self.num_samples = num_samples
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.dropout_rate = dropout_rate
        self.trans_encoder = transformer_Endcoder(num_variables=num_variables, num_samples=num_samples, embed_dim=embed_dim, hidden_dim=hidden_dim, heads=heads, dropout_rate=dropout_rate)
        self.trans_decoder = transformer_Decoder(num_variables=num_variables, embed_dim=embed_dim, hidden_dim=hidden_dim, heads=heads, dropout_rate=dropout_rate)
        self.MLP = nn.SequentialCell(
            nn.Dense(hidden_dim, hidden_dim * 2),
            nn.Dense(hidden_dim * 2, hidden_dim),
            nn.Dense(hidden_dim, 1)
        )

    def __int__(self, new_q_net):
        super(q_net, self).__int__()

    def construct(self, data, position):
        embed = self.trans_encoder(data, position)
        #with decoder
        # output = self.trans_decoder(embed, position)
        #without decoder
        output = embed
        q_value = self.MLP(output)
        position = position.permute(0, 2, 1)
        zero = -9e15 * ops.ones_like(position)
        q_value = ops.where(position == 1, zero, q_value)
        return ops.squeeze(q_value, -1)

