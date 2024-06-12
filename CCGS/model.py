from mindformers import BertForPreTraining, BertConfig
from mindspore.common.initializer import initializer
from mindspore import dtype as mstype
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops

class Conv1D(nn.Cell):
    def __init__(self, in_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=True):
        super(Conv1D, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=in_dim, out_channels=out_dim, kernel_size=kernel_size, padding=padding,
                                stride=stride, has_bias=bias)

    def construct(self, x):
        # suppose all the input with shape (batch_size, seq_len, dim)
        x = ops.transpose(x,(0,2,1))  # (batch_size, dim, seq_len)
        x = self.conv1d(x)
        return ops.transpose(x,(0,2,1))  # (batch_size, seq_len, dim)



class VisualProjection(nn.Cell):
    def __init__(self, visual_dim, dim, drop_rate=0.0):
        super(VisualProjection, self).__init__()
        self.drop = nn.Dropout(p=drop_rate)
        self.linear = Conv1D(in_dim=visual_dim, out_dim=dim, kernel_size=1, stride=1, bias=True, padding=0)

    def construct(self, visual_features):
        # the input visual feature with shape (batch_size, seq_len, visual_dim)
        visual_features = self.drop(visual_features)
        output = self.linear(visual_features)  # (batch_size, seq_len, dim)
        return output

class CQAttention(nn.Cell):
    def __init__(self, dim, drop_rate=0.0):
        super(CQAttention, self).__init__()

        self.w4C = mindspore.Parameter(default_input=initializer('XavierUniform', [dim,1], mstype.float32),requires_grad=True)
        self.w4Q = mindspore.Parameter(default_input=initializer('XavierUniform', [dim,1], mstype.float32),requires_grad=True)
        self.w4mlu = mindspore.Parameter(default_input=initializer('XavierUniform', [1,1,dim], mstype.float32),requires_grad=True)
        self.dropout = nn.Dropout(p=drop_rate)
        self.cqa_linear = Conv1D(in_dim=4 * dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True)

    def construct(self, context, query, c_mask, q_mask):
        score = self.trilinear_attention(context, query)  # (batch_size, c_seq_len, q_seq_len)
        score_ = nn.Softmax(dim=2)(mask_logits(score, q_mask.unsqueeze(1)))  # (batch_size, c_seq_len, q_seq_len)
        score_t = nn.Softmax(dim=1)(mask_logits(score, c_mask.unsqueeze(2)))  # (batch_size, c_seq_len, q_seq_len)
        score_t = score_t.transpose(1, 2)  # (batch_size, q_seq_len, c_seq_len)
        c2q = ops.matmul(score_, query)  # (batch_size, c_seq_len, dim)
        q2c = ops.matmul(torch.matmul(score_, score_t), context)  # (batch_size, c_seq_len, dim)
        output = ops.cat([context, c2q, ops.mul(context, c2q), ops.mul(context, q2c)], dim=2)
        output = self.cqa_linear(output)  # (batch_size, c_seq_len, dim)
        return output

    def trilinear_attention(self, context, query):
        batch_size, c_seq_len, dim = context.shape
        batch_size, q_seq_len, dim = query.shape
        context = self.dropout(context)
        query = self.dropout(query)
        subres0 = ops.matmul(context, self.w4C).expand([-1, -1, q_seq_len])  # (batch_size, c_seq_len, q_seq_len)
        subres1 = ops.transpose(ops.matmul(query, self.w4Q),(0,2,1)).expand([-1, c_seq_len, -1])
        subres2 = ops.matmul(context * self.w4mlu, ops.transpose(query,(0,2,1)))
        res = subres0 + subres1 + subres2  # (batch_size, c_seq_len, q_seq_len)
        return res

class WeightedPool(nn.Cell):
    def __init__(self, dim):
        super(WeightedPool, self).__init__()

        self.weight = mindspore.Parameter(default_input=initializer('XavierUniform', [dim,1], mstype.float32),requires_grad=True)

    def construct(self, x, mask):
        alpha = mindspore.numpy.tensordot(x, self.weight, axes=1)  # shape = (batch_size, seq_length, 1)
        alpha = mask_logits(alpha, mask=mask.unsqueeze(2))
        alphas = nn.Softmax(axis=1)(alpha)
        pooled_x = ops.matmul(ops.transpose(x,(0,2,1)), alphas)  # (batch_size, dim, 1)
        pooled_x = pooled_x.squeeze(2)
        return pooled_x

class CQConcatenate(nn.Cell):
    def __init__(self, dim):
        super(CQConcatenate, self).__init__()
        self.weighted_pool = WeightedPool(dim=dim)
        self.conv1d = Conv1D(in_dim=2 * dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True)

    def construct(self, context, query, q_mask):
        pooled_query = self.weighted_pool(query, q_mask)
        _, c_seq_len, _ = context.shape
        pooled_query = ops.stack([pooled_query] * c_seq_len,axis=2)  # (batch_size, c_seq_len, dim)
        output = ops.cat([context, pooled_query], axis=2)  # (batch_size, c_seq_len, 2*dim)
        output = self.conv1d(output)
        return output


def mask_logits(inputs, mask, mask_value=-1e30):
    mask = mask.type(mindspore.float32)
    return inputs + (1.0 - mask) * mask_value

class PositionalEmbedding(nn.Cell):
    """Construct the embeddings from word, position and token_type embeddings."""
    def __init__(self, num_embeddings, embedding_dim):
        super(PositionalEmbedding, self).__init__()
        self.position_embeddings = nn.Embedding(num_embeddings, embedding_dim)

    def construct(self, inputs):
        bsz, seq_length = inputs.shape[:2]
        position_ids = ops.arange(seq_length, dtype=mindspore.int32)
        position_ids = position_ids.unsqueeze(0).repeat(bsz, 1)  # (N, L)
        position_embeddings = self.position_embeddings(position_ids)
        return position_embeddings


class HighLightLayer(nn.Cell):
    def __init__(self, dim):
        super(HighLightLayer, self).__init__()
        self.conv1d = Conv1D(in_dim=dim, out_dim=1, kernel_size=1, stride=1, padding=0, bias=True)

    def construct(self, x, mask):
        # compute logits
        logits = self.conv1d(x)
        logits = logits.squeeze(2)
        logits = mask_logits(logits, mask)
        # compute score
        scores = nn.Sigmoid()(logits)
        return scores

    @staticmethod
    def compute_loss(scores, labels, mask, epsilon=1e-12):
        labels = labels.type(torch.float32)
        weights = torch.where(labels == 0.0, labels + 1.0, 2.0 * labels)
        loss_per_location = nn.BCELoss(reduction='none')(scores, labels)
        loss_per_location = loss_per_location * weights
        mask = mask.type(torch.float32)
        loss = torch.sum(loss_per_location * mask) / (torch.sum(mask) + epsilon)
        return loss


class GlobalSpanModel(nn.Cell):
    def __init__(self,pretrained_model='bert',inner_dim=1024):
        super(GlobalSpanModel, self).__init__()
        self.video_affine = VisualProjection(visual_dim=1024,dim=768,drop_rate=0.1)
        #self.cq_attention = CQAttention(dim=768, drop_rate=0.1)
        self.cq_concat = CQConcatenate(dim=768)
        self.highlight_layer = HighLightLayer(dim=768)


        if pretrained_model == 'bert':
            config = BertConfig.from_pretrained("bert_base_uncased")
            config.seq_length = 512
            config.compute_dtype =  mindspore.float32
            self.P_Encoder = BertForPreTraining(config)
        self.inner_dim = inner_dim
        self.dense = nn.Dense(self.P_Encoder.config.hidden_size,self.inner_dim * 2)
        self.celoss = nn.CrossEntropyLoss()

    def sinusoidal_position_embedding(self, batch_size, seq_len, output_dim):
        position_ids = ops.arange(0, seq_len, dtype=mindspore.float32).unsqueeze(-1)

        indices = ops.arange(0, output_dim // 2, dtype=mindspore.float32)
        indices = ops.pow(10000, -2 * indices / output_dim)
        embeddings = position_ids * indices
        embeddings = ops.stack([ops.sin(embeddings), ops.cos(embeddings)], axis=-1)
        embeddings = ops.stack([embeddings]*batch_size,axis=0)
        embeddings = ops.reshape(embeddings, (batch_size, seq_len, output_dim))
        return embeddings

    def calc_loss(self,y_true, y_pred):
        # 1. 取出真实的标签
        y_true = y_true.view(-1)
        y_pred = y_pred.view(-1)

        # 3. 奇偶向量相乘
        y_pred = y_pred * 20

        # 4. 取出负例-正例的差值
        y_pred = y_pred[:, None] - y_pred[None, :]  # 这里是算出所有位置 两两之间余弦的差值
        # 矩阵中的第i行j列  表示的是第i个余弦值-第j个余弦值
        y_true = y_true[:, None] < y_true[None, :]  # 取出负例-正例的差值
        y_true = y_true.to(mindspore.float32)
        y_pred = y_pred - (1 - y_true) * 1e12
        y_pred = y_pred.view(-1)
        y_pred = ops.cat((mindspore.tensor([0.]), y_pred), axis=0)  # 这里加0是因为e^0 = 1相当于在log中加了1
        return ops.logsumexp(y_pred, axis=0)

    def get_hidden(self,p_hidden_sep):
        p_hidden_sep = ops.stack(p_hidden_sep)
        p_hidden_sep = self.dense(p_hidden_sep)
        p_hidden_sep = ops.split(p_hidden_sep, self.inner_dim*2, axis=-1)
        p_hidden_sep = ops.stack(p_hidden_sep, axis=-2)

        qw, kw = p_hidden_sep[..., :self.inner_dim], p_hidden_sep[..., self.inner_dim:]

        pos_emb = self.sinusoidal_position_embedding(p_hidden_sep.shape[0], p_hidden_sep.shape[1], self.inner_dim)
        # cos_pos,sin_pos: (batch_size, seq_len, 1, inner_dim)
        cos_pos = pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)
        sin_pos = pos_emb[..., None, ::2].repeat_interleave(2, dim=-1)
        qw2 = ops.stack([-qw[..., 1::2], qw[..., ::2]], axis=-1)
        qw2 = qw2.reshape(qw.shape)
        qw = qw * cos_pos + qw2 * sin_pos
        kw2 = ops.stack([-kw[..., 1::2], kw[..., ::2]], axis=-1)
        kw2 = kw2.reshape(kw.shape)
        kw = kw * cos_pos + kw2 * sin_pos

        logit= ops.einsum('bmhd,bnhd->bhmn', qw, kw)
        return logit
    def construct(
            self,
            input_ids, attention_mask, token_types,ious=None,vfeats=None,vfeats_mask=None
    ):


        sequence_output, pooled_output, embedding_table = self.P_Encoder.bert(input_ids=input_ids,input_mask=attention_mask,token_type_ids=ops.zeros_like(input_ids))

        p_hidden = sequence_output

        video_features = self.video_affine(vfeats)

        #features = self.cq_attention(video_features, p_hidden, vfeats_mask, attention_mask)
        features = self.cq_concat(video_features,p_hidden,attention_mask)

        features = self.highlight_layer(features,vfeats_mask)


        p_hidden_sep1, p_hidden_sep2 = [], []

        n_i1, n_i2 = 0,0
        for i in token_types[0]:
            if i == 1:
                p_hidden_sep1.append(p_hidden[0, n_i1] + features[0, :])
            n_i1 += 1

        for i in token_types[1]:
            if i == 1:
                p_hidden_sep2.append(p_hidden[1, n_i2]+features[1,:])
            n_i2 += 1


        p_hidden_sep1 = self.get_hidden(p_hidden_sep1)
        p_hidden_sep2 = self.get_hidden(p_hidden_sep2)

        p_hidden_sep1 = p_hidden_sep1[0, 0, :-1, 1:].contiguous()

        p_hidden_sep2 = p_hidden_sep2[0, 0, :-1, 1:].contiguous()

        loss = None
        IOUloss, CEloss = 0, 0
        if ious != None:
            IOUloss = self.calc_loss(ious, p_hidden_sep1)
            CEloss = self.celoss(
                ops.cat([p_hidden_sep1.view(1, -1).squeeze(0), p_hidden_sep2.view(1, -1).squeeze(0)]).unsqueeze(0),
                ious.view(1, -1).argmax(1).to(mindspore.int32))

        argmax = p_hidden_sep1.view(-1).argmax().item()

        return {'logits': p_hidden_sep1, 'IOUloss': IOUloss, 'CEloss': CEloss,
                'start': int(argmax / p_hidden_sep1.shape[0]), 'end': int(argmax % p_hidden_sep1.shape[0])}

    def forward_test(
            self,
            input_ids, attention_mask, token_types,ious=None,vfeats=None,vfeats_mask=None
    ):
        sequence_output, pooled_output, embedding_table = self.P_Encoder.bert(input_ids=input_ids,input_mask=attention_mask,token_type_ids=ops.zeros_like(input_ids))
        p_hidden = sequence_output

        video_features = self.video_affine(vfeats)
        #features = self.cq_attention(video_features, p_hidden, vfeats_mask, attention_mask)
        features = self.cq_concat(video_features,p_hidden,attention_mask)

        features = self.highlight_layer(features,vfeats_mask)
        p_hidden_sep = []
        for i in token_types.nonzero()[:,1]:
           p_hidden_sep.append(p_hidden[0,i]+features[0,:])
        p_hidden_sep = ops.stack(p_hidden_sep)
        p_hidden_sep = self.dense(p_hidden_sep)
        p_hidden_sep = ops.split(p_hidden_sep, self.inner_dim*2, axis=-1)
        p_hidden_sep = ops.stack(p_hidden_sep, axis=-2)

        qw, kw = p_hidden_sep[..., :self.inner_dim], p_hidden_sep[..., self.inner_dim:]

        pos_emb = self.sinusoidal_position_embedding(input_ids.shape[0], p_hidden_sep.shape[1], self.inner_dim)
        # cos_pos,sin_pos: (batch_size, seq_len, 1, inner_dim)
        cos_pos = pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)
        sin_pos = pos_emb[..., None, ::2].repeat_interleave(2, dim=-1)
        qw2 = ops.stack([-qw[..., 1::2], qw[..., ::2]], axis=-1)
        qw2 = qw2.reshape(qw.shape)
        qw = qw * cos_pos + qw2 * sin_pos
        kw2 = ops.stack([-kw[..., 1::2], kw[..., ::2]], axis=-1)
        kw2 = kw2.reshape(kw.shape)
        kw = kw * cos_pos + kw2 * sin_pos
        logits = ops.einsum('bmhd,bnhd->bhmn', qw, kw)

        logits = logits[0,0,:-1,1:]
        IOUloss,CEloss = 0,0
        if ious != None:
            IOUloss = self.calc_loss(ious,logits)
            CEloss = self.celoss(logits.view(1,-1),ious.view(1,-1).argmax(1))

        argmax = logits.view(-1).argmax()

        return {'logits':logits,'IOUloss':IOUloss,'CEloss':CEloss,'start':int(argmax/logits.shape[0]),'end':int(argmax%logits.shape[0])}
