import torch
import torch.nn as nn
import torch.nn.functional as F

class transformer_Endcoder(nn.Module):
    def __init__(self, num_variables, num_samples, embed_dim=128, hidden_dim=128, heads=4, dropout_rate=0.0, device=None) -> None:
        super(transformer_Endcoder, self).__init__()
        self.device = device
        self.dropout_rate = dropout_rate
        self.embed_dim = embed_dim
        self.heads = heads
        self.hidden_dim = hidden_dim
        self.data_embedding = nn.Linear(in_features=num_samples, out_features=embed_dim, device=device)
        self.position_embedding1 = nn.Linear(in_features=1, out_features=embed_dim, device=device)
        self.position_embedding2 = nn.Linear(in_features=1, out_features=embed_dim, device=device)
        # position+data
        self.attention = MultiHeadAttention(num_variables=num_variables, input_dim=embed_dim * 2, output_dim=hidden_dim,
                                            heads=heads,
                                            dropout_rate=dropout_rate, device=device)
        # only position
        # self.attention = MultiHeadAttention(num_variables=num_variables, input_dim=embed_dim, output_dim=hidden_dim, heads=heads,
        #                                     dropout_rate=dropout_rate, device=device)
        # position+data
        self.ln1 = nn.LayerNorm(normalized_shape=embed_dim*2, device=device)
        # only position
        # self.ln1 = nn.LayerNorm(normalized_shape=embed_dim, device=device)
        # position+data
        self.ln2 = nn.LayerNorm(normalized_shape=embed_dim*2, device=device)
        # only position
        # self.ln2 = nn.LayerNorm(normalized_shape=embed_dim, device=device)
        # position+data
        self.mlp = nn.Sequential(nn.Linear(in_features=embed_dim*2, out_features=embed_dim*2, device=device),
                                         nn.GELU().to(device=device),
                                         nn.Linear(in_features=embed_dim*2, out_features=hidden_dim, device=device))
        # only position
        # self.mlp = nn.Sequential(nn.Linear(in_features=embed_dim, out_features=embed_dim*2, device=device),
        #                          nn.GELU().to(device=device),
        #                          nn.Linear(in_features=embed_dim*2, out_features=hidden_dim, device=device))

    def forward(self, data, position) -> torch.Tensor:
        data_embed = self.data_embedding(data)
        position_embed1 = self.position_embedding1(position.permute(0, 2, 1))
        #position+data
        embed = self.ln1(torch.cat([data_embed, position_embed1], dim=-1))
        #only position
        # embed = self.ln1(position_embed1)
        embed = self.attention(embed)
        # position+data
        data_embed += embed
        position_embed2 = self.position_embedding2(position.permute(0, 2, 1))
        # position+data
        embed = self.ln2(torch.cat([data_embed, position_embed2], dim=-1))
        # only position
        # embed = self.ln2(position_embed2)
        embed = self.mlp(embed)
        # position+data
        output = data_embed + embed
        # only position
        # output = embed
        return output

class transformer_Decoder(nn.Module):
    def __init__(self, num_variables, embed_dim=128, hidden_dim=128, heads=4, dropout_rate=0.0, device=None) -> None:
        super(transformer_Decoder, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.dropout_rate = dropout_rate
        self.device = device
        self.position_embedding1 = nn.Linear(in_features=1, out_features=embed_dim, device=device)
        self.position_embedding2 = nn.Linear(in_features=1, out_features=embed_dim, device=device)
        # position+data
        self.attention = MultiHeadAttention(num_variables=num_variables, input_dim=embed_dim * 2, output_dim=hidden_dim,
                                                    heads=heads,
                                                    dropout_rate=dropout_rate, device=device)
        # only position
        # self.attention = MultiHeadAttention(num_variables=num_variables, input_dim=embed_dim, output_dim=hidden_dim,
        #                                     heads=heads,
        #                                     dropout_rate=dropout_rate, device=device)
        # position+data
        self.ln1 = nn.LayerNorm(normalized_shape=embed_dim*2, device=device)
        # only position
        # self.ln1 = nn.LayerNorm(normalized_shape=embed_dim, device=device)
        # position+data
        self.ln2 = nn.LayerNorm(normalized_shape=embed_dim*2, device=device)
        # only position
        # self.ln2 = nn.LayerNorm(normalized_shape=embed_dim, device=device)
        # position+data
        self.mlp = nn.Sequential(nn.Linear(in_features=embed_dim * 2, out_features=hidden_dim, device=device),
                                         nn.GELU().to(device=device))
        # only position
        # self.mlp = nn.Sequential(nn.Linear(in_features=embed_dim, out_features=hidden_dim, device=device),
        #                          nn.GELU().to(device=device))
        self.ln3 = nn.LayerNorm(normalized_shape=embed_dim, device=device)

    def forward(self, data_embed, position) -> torch.Tensor:
        position_embed = self.position_embedding1(position.permute(0, 2, 1))
        # position+data
        embed = self.ln1(torch.cat([data_embed, position_embed], dim=-1))
        # only position
        # embed = self.ln1(position_embed)
        embed = self.attention(embed)
        # position+data
        data_embed += embed
        position_embed2 = self.position_embedding2(position.permute(0, 2, 1))
        # position+data
        embed = self.ln2(torch.cat([data_embed, position_embed2], dim=-1))
        # only position
        # embed = self.ln2(position_embed2)
        embed = self.mlp(embed)
        # position+data
        output = data_embed + embed
        # only position
        # output = embed
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, num_variables, input_dim, output_dim, heads=4, dropout_rate=0.1, device=None):
        super(MultiHeadAttention, self).__init__()
        self.num_variables = num_variables
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.heads = heads
        self.dropout_rate = dropout_rate
        self.device = device
        self.w_q = nn.Sequential(
            nn.Linear(in_features=input_dim,
                      out_features=output_dim,
                      device=device),
            nn.ReLU().to(device=device)
        )
        self.w_k = nn.Sequential(
            nn.Linear(in_features=input_dim,
                      out_features=output_dim,
                      device=device),
            nn.ReLU().to(device=device)
        )
        self.w_v = nn.Sequential(
            nn.Linear(in_features=input_dim,
                      out_features=output_dim,
                      device=device),
            nn.ReLU().to(device=device)
        )
        #self.ln = nn.LayerNorm([num_variables, output_dim])

    def forward(self, embed) -> torch.Tensor:
        Q = self.w_q(embed)
        K = self.w_k(embed)
        V = self.w_v(embed)
        Q_ = torch.cat(torch.split(Q,
                                     split_size_or_sections=Q.shape[2] // self.heads,
                                     dim=2),
                         dim=0)
        K_ = torch.cat(torch.split(K,
                                     split_size_or_sections=K.shape[2] // self.heads,
                                     dim=2),
                         dim=0)
        V_ = torch.cat(torch.split(V,
                                     split_size_or_sections=V.shape[2] // self.heads,
                                     dim=2),
                         dim=0)
        output = torch.matmul(Q_, K_.permute(0, 2, 1))
        output = output / (K_.shape[-1] ** 0.5)
        output = F.softmax(output, dim=1)
        output = F.dropout(output, p=self.dropout_rate)
        output = torch.matmul(output, V_)
        output = torch.cat(torch.split(output,
                                       split_size_or_sections=output.shape[0] // self.heads,
                                       dim=0),
                           dim=2)
        return output
