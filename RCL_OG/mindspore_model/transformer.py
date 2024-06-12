import mindspore
from mindspore import nn, ops, Parameter, Tensor

class transformer_Endcoder(nn.Cell):
    def __init__(self, num_variables, num_samples, embed_dim=128, hidden_dim=128, heads=4, dropout_rate=0.0) -> None:
        super(transformer_Endcoder, self).__init__()
        self.dropout_rate = dropout_rate
        self.embed_dim = embed_dim
        self.heads = heads
        self.hidden_dim = hidden_dim

        self.data_embedding = nn.Dense(num_samples, embed_dim)
        self.position_embedding1 = nn.Dense(1, embed_dim)
        self.position_embedding2 = nn.Dense(1, embed_dim)
        # position+data
        self.attention = MultiHeadAttention(num_variables=num_variables, input_dim=embed_dim * 2, output_dim=hidden_dim,
                                            heads=heads,
                                            dropout_rate=dropout_rate)
        # only position
        # self.attention = MultiHeadAttention(num_variables=num_variables, input_dim=embed_dim, output_dim=hidden_dim, heads=heads,
        #                                     dropout_rate=dropout_rate, device=device)
        # position+data
        self.ln1 = nn.LayerNorm(normalized_shape=[embed_dim*2])
        # only position
        # self.ln1 = nn.LayerNorm(normalized_shape=embed_dim, device=device)
        # position+data
        self.ln2 = nn.LayerNorm(normalized_shape=[embed_dim*2])
        # only position
        # self.ln2 = nn.LayerNorm(normalized_shape=embed_dim, device=device)
        # position+data
        self.mlp = nn.SequentialCell(nn.Dense(embed_dim*2, embed_dim*2),
                                         nn.GELU(),
                                         nn.Dense(embed_dim*2, hidden_dim))
        # only position
        # self.mlp = nn.Sequential(nn.Linear(in_features=embed_dim, out_features=embed_dim*2, device=device),
        #                          nn.GELU().to(device=device),
        #                          nn.Linear(in_features=embed_dim*2, out_features=hidden_dim, device=device))

    def construct(self, data, position):
        data_embed = self.data_embedding(data)
        position_embed1 = self.position_embedding1(position.permute(0, 2, 1))
        #position+data
        embed = self.ln1(ops.cat([data_embed, position_embed1], -1))
        #only position
        # embed = self.ln1(position_embed1)
        embed = self.attention(embed)
        # position+data
        data_embed += embed
        position_embed2 = self.position_embedding2(position.permute(0, 2, 1))
        # position+data
        embed = self.ln2(ops.cat([data_embed, position_embed2], -1))
        # only position
        # embed = self.ln2(position_embed2)
        embed = self.mlp(embed)
        # position+data
        output = data_embed + embed
        # only position
        # output = embed
        return output

class transformer_Decoder(nn.Cell):
    def __init__(self, num_variables, embed_dim=128, hidden_dim=128, heads=4, dropout_rate=0.0) -> None:
        super(transformer_Decoder, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.dropout_rate = dropout_rate
        self.position_embedding1 = nn.Dense(1, embed_dim)
        self.position_embedding2 = nn.Dense(1, embed_dim)
        # position+data
        self.attention = MultiHeadAttention(num_variables=num_variables, input_dim=embed_dim * 2, output_dim=hidden_dim,
                                                    heads=heads,
                                                    dropout_rate=dropout_rate)
        # only position
        # self.attention = MultiHeadAttention(num_variables=num_variables, input_dim=embed_dim, output_dim=hidden_dim,
        #                                     heads=heads,
        #                                     dropout_rate=dropout_rate, device=device)
        # position+data
        self.ln1 = nn.LayerNorm(normalized_shape=[embed_dim*2])
        # only position
        # self.ln1 = nn.LayerNorm(normalized_shape=embed_dim, device=device)
        # position+data
        self.ln2 = nn.LayerNorm(normalized_shape=[embed_dim*2])
        # only position
        # self.ln2 = nn.LayerNorm(normalized_shape=embed_dim, device=device)
        # position+data
        self.mlp = nn.SequentialCell(nn.Dense(embed_dim * 2, hidden_dim),
                                         nn.GELU())
        # only position
        # self.mlp = nn.Sequential(nn.Linear(in_features=embed_dim, out_features=hidden_dim, device=device),
        #                          nn.GELU().to(device=device))
        self.ln3 = nn.LayerNorm(normalized_shape=[embed_dim])

    def construct(self, data_embed, position):
        position_embed = self.position_embedding1(position.permute(0, 2, 1))
        # position+data
        embed = self.ln1(ops.cat([data_embed, position_embed], -1))
        # only position
        # embed = self.ln1(position_embed)
        embed = self.attention(embed)
        # position+data
        data_embed += embed
        position_embed2 = self.position_embedding2(position.permute(0, 2, 1))
        # position+data
        embed = self.ln2(ops.cat([data_embed, position_embed2], -1))
        # only position
        # embed = self.ln2(position_embed2)
        embed = self.mlp(embed)
        # position+data
        output = data_embed + embed
        # only position
        # output = embed
        return output


class MultiHeadAttention(nn.Cell):
    def __init__(self, num_variables, input_dim, output_dim, heads=4, dropout_rate=0.1):
        super(MultiHeadAttention, self).__init__()
        self.num_variables = num_variables
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.heads = heads
        self.dropout_rate = dropout_rate
        self.w_q = nn.SequentialCell(
            nn.Dense(input_dim,
                      output_dim),
            nn.ReLU()
        )
        self.w_k = nn.SequentialCell(
            nn.Dense(input_dim,
                      output_dim),
            nn.ReLU()
        )
        self.w_v = nn.SequentialCell(
            nn.Dense(input_dim,
                      output_dim),
            nn.ReLU()
        )
        #self.ln = nn.LayerNorm([num_variables, output_dim])

    def construct(self, embed):
        Q = self.w_q(embed)
        K = self.w_k(embed)
        V = self.w_v(embed)
        Q_ = ops.cat(ops.split(Q,
                                     Q.shape[2] // self.heads,
                                     2),
                         0)
        K_ = ops.cat(ops.split(K,
                                     K.shape[2] // self.heads,
                                     2),
                         0)
        V_ = ops.cat(ops.split(V,
                                     V.shape[2] // self.heads,
                                     2),
                         0)
        output = ops.matmul(Q_, K_.permute(0, 2, 1))
        output = output / (K_.shape[-1] ** 0.5)
        output = ops.softmax(output, 1)
        output = ops.matmul(output, V_)
        output = ops.cat(ops.split(output,
                                       output.shape[0] // self.heads,
                                       0),
                           2)
        return output
