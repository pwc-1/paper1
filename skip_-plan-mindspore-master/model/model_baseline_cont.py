from model.transformer import *
from utils import *
import mindspore as ms


class Model(nn.Cell):
    def __init__(self, args, memory_frames=None, memory_lowlevel_labels=None):
        super(Model, self).__init__()
        self.args = args
        self.frameduration = args.frameduration
        self.adapool = nn.AdaptiveAvgPool1d(1)
        self.max_traj_len = args.max_traj_len
        self.memory_size = args.memory_size
        self.query_length = args.query_length
        self.memory_length = args.memory_length
        self.query_embed = nn.Embedding(self.query_length, args.d_model)
        self.framesembed_init = nn.Embedding(self.memory_length, args.d_model)
        self.d_model = args.d_model
        self.H = args.H
        decoder_layer = TransformerDecoderLayer(
            self.d_model,
            args.H,
            args.dim_feedforward,
            args.decoder_dropout,
            "relu",
            normalize_before=True,
            memory_size=args.memory_size,
            bs=args.batch_size,
        )
        decoder_norm = nn.LayerNorm([self.d_model], epsilon=1e-5)
        self.decoder = TransformerDecoder(
            decoder_layer, args.N, decoder_norm, return_intermediate=False
        )
        # self.dropout_feas = nn.Dropout(args.feat_dropout)
        self.cls_classifier = MLP(self.d_model, 106, [args.mlp_mid])
        self.feat_reshape = MLP(640, self.d_model, [1024])
        self.apply(self._init_weights)
        # self.merge = MLP(3, 1, [6])
        self.merge = MLP(3, 1, [3 * args.smallmid_ratio])

    def construct(self, frames):
        # initial and goal
        frames1 = frames.astype(ms.float32)
        frames_initial = (
            self.merge(self.feat_reshape(frames1[:, 0]).transpose(0, 2, 1))
            .transpose(0, 2, 1)
            .transpose(1, 0, 2)
        )
        frames_goal = (
            self.merge(self.feat_reshape(frames1[:, -1]).transpose(0, 2, 1))
            .transpose(0, 2, 1)
            .transpose(1, 0, 2)
        )

        frames_initial = ops.Reshape()(frames_initial, (32, 1024))
        frames_goal = ops.Reshape()(frames_goal, (32, 1024))

        query_embed = ops.unsqueeze(self.query_embed.embedding_table, 1)
        query_embed = ops.tile(query_embed, (1, frames1.shape[0], 1))

        query_embed[0, :] = frames_initial
        query_embed[-1, :] = frames_goal

        # memory
        framesembed = ops.unsqueeze(self.framesembed_init.embedding_table, 1)
        framesembed = ops.tile(framesembed, (1, self.memory_size, 1))
        # net
        out = self.decoder(query_embed, framesembed)
        out = ops.Reshape()(out, (4, 32, 1024))
      
        cls_output = ops.transpose(self.cls_classifier(out), (1, 0, 2))

        cls_output = cls_output[:, : self.max_traj_len, :]
        return cls_output

    @staticmethod
    def _init_weights(cell):
        r"""Initialize weights like BERT - N(0.0, 0.02), bias = 0."""
        if isinstance(cell, nn.Embedding):
            cell.embedding_table.set_data(
                ms.common.initializer.initializer(
                    ms.common.initializer.TruncatedNormal(
                        sigma=0.02, mean=0.0, a=-2.0, b=2.0
                    ),
                    cell.embedding_table.shape,
                    cell.embedding_table.dtype,
                )
            )
        if isinstance(cell, nn.MultiheadAttention):
            cell.in_proj_weight.set_data(
                ms.common.initializer.initializer(
                    ms.common.initializer.TruncatedNormal(
                        sigma=0.02, mean=0.0, a=-2.0, b=2.0
                    ),
                    cell.in_proj_weight.shape,
                    cell.in_proj_weight.dtype,
                )
            )
            cell.out_proj.weight.set_data(
                ms.common.initializer.initializer(
                    ms.common.initializer.TruncatedNormal(
                        sigma=0.02, mean=0.0, a=-2.0, b=2.0
                    ),
                    cell.out_proj.weight.shape,
                    cell.out_proj.weight.dtype,
                )
            )
        if isinstance(cell, nn.Dense):
            cell.weight.set_data(
                ms.common.initializer.initializer(
                    ms.common.initializer.TruncatedNormal(
                        sigma=0.02, mean=0.0, a=-2.0, b=2.0
                    ),
                    cell.weight.shape,
                    cell.weight.dtype,
                )
            )
            cell.bias.set_data(
                ms.common.initializer.initializer(
                    "zero", cell.bias.shape, cell.bias.dtype
                )
            )
