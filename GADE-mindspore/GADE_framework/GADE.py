import mindspore
import mindspore.nn as nn
from mindnlp.transformers.models import *
from mindspore.common.initializer import initializer, XavierUniform
import numpy as np

class GADE(nn.Cell):
    def __init__(self, args):
        super(GADE, self).__init__()
        self.args = args
        self.lrm = LRM(args.max_seq_length)
        self.gim = GIM([args.gcn_dim]*(args.gcn_layer + 1), args.dropout)

    def construct(self, batch):
        features, inter_strength_mat, label, mask = self.lrm(batch)
        pred = self.gim(features, inter_strength_mat)
        return pred, label, mask

class LRM(nn.Cell):
    def __init__(self, max_seq_length=128):
        super(LRM, self).__init__()
        self.max_seq_length = max_seq_length

        # BERT tokenizer
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        config = BertConfig.from_pretrained('bert-base-cased')

        self.encoder = BertModel.from_pretrained('bert-base-cased', config=config)

        self.dim = 768
        self.sim1 = nn.Dense(self.dim*2, self.dim)
        self.relu = nn.ReLU()
        self.sim2 = nn.Dense(self.dim, 1)

    def encode_feature(self, cand_docs):
        input_ids = []
        segment_ids = []
        input_masks = []

        for s in cand_docs["input_tokens"]:
            tokens = ["[CLS]"] + s + ["[SEP]"] + cand_docs["description_token"] + ["[SEP]"]
            tokens = self.bert_tokenizer.convert_tokens_to_ids(tokens)
            seg_pos = len(s) + 2
            seg_ids = [0] * seg_pos + [1] * (len(tokens) - seg_pos)
            mask = [1] * len(tokens)
            padding = [0] * (self.max_seq_length - len(tokens))
            tokens += padding
            input_ids.append(tokens)
            seg_ids += padding
            segment_ids.append(seg_ids)
            mask += padding
            input_masks.append(mask)

        input_ids = mindspore.Tensor(input_ids, dtype=mindspore.int32)
        segment_ids = mindspore.Tensor(segment_ids, dtype=mindspore.int32)
        input_masks = mindspore.Tensor(input_masks, dtype=mindspore.int32)
        outputs = self.encoder(input_ids, token_type_ids=segment_ids, attention_mask=input_masks)
        return outputs[1]

    def document_interaction_graph_construction(self, cand_docs, max_n):
        features = self.encode_feature(cand_docs)
        num_nodes, fdim = features.shape
        N = num_nodes
        
        A_feat = mindspore.ops.concat((mindspore.ops.tile(features, (1,N)).view(N*N,-1), mindspore.ops.tile(features, (N, 1))), -1).view(N, -1, 2 * self.dim)
        A_feat = self.sim2(self.relu(self.sim1(A_feat)))
        A_feat = mindspore.ops.squeeze(A_feat)
        A_feat = mindspore.ops.softmax(A_feat, -1)
        A_ = mindspore.ops.zeros((max_n, max_n), dtype=mindspore.float32)
        A_[:num_nodes, :num_nodes] = A_feat

        labels = cand_docs["labels"].copy()
        mask = [1] * len(cand_docs["labels"])

        if max_n - num_nodes > 0:
            features = mindspore.ops.concat((features, mindspore.ops.zeros((max_n - num_nodes, fdim), dtype=mindspore.dtype.float32)), 0)
            labels += [-10] * (max_n - num_nodes)
            mask += [0] * (max_n - num_nodes)

        return features, A_, labels, mask

    def construct(self,batch_data):
        features = []
        inter_strength_mat = []
        label = []
        mask = []

        max_n = 0

        for bd in batch_data:
            if len(bd["labels"]) > max_n:
                max_n = len(bd["labels"])

        for bd in batch_data:
            feat, _A, l, m = self.document_interaction_graph_construction(bd, max_n)
            features.append(feat)
            inter_strength_mat.append(_A)
            label.append(l)
            mask.append(m)

        features = mindspore.ops.stack(tuple(features), 0)
        inter_strength_mat = mindspore.ops.stack(tuple(inter_strength_mat), 0)
        label = mindspore.Tensor(label, dtype=mindspore.int32)
        mask = mindspore.Tensor(mask, dtype=mindspore.int32)

        return features, inter_strength_mat, label, mask


class Aggregator(nn.Cell):
    def __init__(self):
        super(Aggregator, self).__init__()

    def construct(self, features, A):
        x = mindspore.ops.bmm(A, features)
        return x


# GCN Layer
class GraphConvLayer(nn.Cell):
    def __init__(self, in_dim, out_dim, aggregator):
        super(GraphConvLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.aggregator = aggregator()

        init_range = 1.0 / np.sqrt(in_dim)
        initial = np.random.uniform(-init_range, init_range, (in_dim*2, out_dim)).astype(np.float32)
        self.weight = mindspore.Parameter(mindspore.Tensor(initial, mindspore.float32), name='w')
        initial_bias = np.random.uniform(-init_range, init_range, (out_dim, )).astype(np.float32)
        self.bias = mindspore.Parameter(mindspore.Tensor(initial_bias, mindspore.float32), name='b')

    def construct(self, features, A):
        batch, node_num, d = features.shape
        assert d == self.in_dim
        agg_features = self.aggregator(features, A)
        agg_features = mindspore.ops.concat((features, agg_features), -1)
        agg_features = agg_features.view(batch*node_num, -1)
        out_features = mindspore.ops.matmul(agg_features, self.weight)
        out_features += self.bias
        out = mindspore.ops.relu(out_features)
        out = out.view(batch, node_num, -1)
        return out


class GIM(nn.Cell):
    def __init__(self, dims, dropout=0.0):
        super(GIM, self).__init__()

        self.convs = []
        self.layers = len(dims) - 1
        self.dropout = nn.Dropout(1-dropout)

        for i in range(len(dims) - 1):
            self.convs.append(GraphConvLayer(dims[i], dims[i + 1], Aggregator))
        self.convs = nn.CellList(self.convs)
        self.layernorm = mindspore.nn.LayerNorm(normalized_shape=(dims[-1],))
        fc1 = nn.Dense(dims[-1], dims[-1])
        prelu = nn.PReLU(dims[-1])
        fc2 = nn.Dense(dims[-1], 2)
        self.classifier = nn.SequentialCell([fc1, prelu, fc2])
        self.fc = nn.Dense(dims[-1]*2, dims[-1])

    def construct(self, x, A):
        x = self.dropout(x)
        x_loc = x.view(-1, x.shape[-1])
        for conv in self.convs:
            x = conv(x, A)

        out = x.shape[-1]
        x = x.view(-1, out)
        x = x + x_loc
        x = self.layernorm(x)
        pred = self.classifier(x)

        return pred


