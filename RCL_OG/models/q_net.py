import torch
import torch.nn as nn
import torch.nn.functional as F
from models.transformer import transformer_Endcoder, transformer_Decoder

class q_net(nn.Module):
    def __init__(self, num_variables, num_samples, embed_dim=128, hidden_dim=128, heads=4, dropout_rate=0.0, device=None) -> None:
        super(q_net, self).__init__()
        self.num_variables = num_variables
        self.num_samples = num_samples
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.dropout_rate = dropout_rate
        self.device = device
        self.trans_encoder = transformer_Endcoder(num_variables=num_variables, num_samples=num_samples, embed_dim=embed_dim, hidden_dim=hidden_dim, heads=heads, dropout_rate=dropout_rate, device=device)
        self.trans_decoder = transformer_Decoder(num_variables=num_variables, embed_dim=embed_dim, hidden_dim=hidden_dim, heads=heads, dropout_rate=dropout_rate, device=device)
        self.MLP = nn.Sequential(
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim * 2, device=device),
            nn.Linear(in_features=hidden_dim * 2, out_features=hidden_dim, device=device),
            nn.Linear(in_features=hidden_dim, out_features=1, device=device)
        )

    def __int__(self, new_q_net):
        super(q_net, self).__int__()

    def forward(self, data, position) -> torch.Tensor:
        embed = self.trans_encoder(data, position)
        #with decoder
        # output = self.trans_decoder(embed, position)
        #without decoder
        output = embed
        q_value = self.MLP(output)
        position = position.permute(0, 2, 1)
        zero = -9e15 * torch.ones_like(position)
        q_value = torch.where(position == 1, zero, q_value)
        return torch.squeeze(q_value, dim=-1)

