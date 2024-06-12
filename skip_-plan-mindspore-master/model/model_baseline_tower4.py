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
        self.query_embed2 = nn.Embedding(args.query_length, args.d_model)
        self.framesembed_init2 = nn.Embedding(args.memory_length, args.d_model)
        self.query_embed3 = nn.Embedding(args.query_length, args.d_model)
        self.framesembed_init3 = nn.Embedding(args.memory_length, args.d_model)

        self.d_model = args.d_model
        self.H = args.H
        decoder_layer2 = TransformerDecoderLayer(
            self.d_model,
            args.H,
            args.dim_feedforward,
            args.decoder_dropout,
            "relu",
            normalize_before=True,
            memory_size=args.memory_size,
            bs=args.batch_size,
        )
        decoder_layer3 = TransformerDecoderLayer(
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
        self.decoder2 = TransformerDecoder(
            decoder_layer2, args.N, decoder_norm, return_intermediate=False
        )
        self.decoder3 = TransformerDecoder(
            decoder_layer3, args.N, decoder_norm, return_intermediate=False
        )
        self.cls_classifier2 = MLP(self.d_model, 106, [args.mlp_mid])
        self.cls_classifier3 = MLP(self.d_model, 106, [args.mlp_mid])
        self.cls_classifier = MLP(self.d_model, 106, [args.mlp_mid])

        self.cls_classifier_merge = MLP(8, 4, [24])

        self.feat_reshape = MLP(640, self.d_model, [1024])
        if args.init_weight:
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
        # print(frames_initial.shape, frames_goal.shape)

        frames_initial = ops.Reshape()(frames_initial, (32, 1024))
        frames_goal = ops.Reshape()(frames_goal, (32, 1024))

        query_embed2 = ops.unsqueeze(self.query_embed2.embedding_table, 1)
        query_embed2 = ops.tile(query_embed2, (1, frames1.shape[0], 1))
        query_embed2[0, :] = frames_initial
        query_embed2[-1, :] = frames_goal
        # print(query_embed2.shape)

        query_embed3 = ops.unsqueeze(self.query_embed3.embedding_table, 1)
        query_embed3 = ops.tile(query_embed3, (1, frames1.shape[0], 1))
        query_embed3[0, :] = frames_initial
        query_embed3[-1, :] = frames_goal

        # memory
        framesembed2 = ops.unsqueeze(self.framesembed_init2.embedding_table, 1)
        framesembed2 = ops.tile(framesembed2, (1, self.memory_size, 1))
        framesembed3 = ops.unsqueeze(self.framesembed_init3.embedding_table, 1)
        framesembed3 = ops.tile(framesembed3, (1, self.memory_size, 1))
        # print(framesembed2.shape)
        # net
        out2 = self.decoder2(query_embed2, framesembed2)
        out3 = self.decoder3(query_embed3, framesembed3)
        out2 = ops.Reshape()(out2, (5, 32, 1024))
        out3 = ops.Reshape()(out3, (5, 32, 1024))
        # print(out2.shape)

        output2 = out2.transpose(1, 0, 2)[:, :3, :]
        output3 = out3.transpose(1, 0, 2)[:, :3, :]

        input2 = ops.zeros(
            (frames.shape[0], self.max_traj_len, self.d_model), ms.float32
        )
        input3 = ops.zeros(
            (frames.shape[0], self.max_traj_len, self.d_model), ms.float32
        )
        input2[:, 0:2, :] = output2[:, 0:2, :]
        input2[:, 3:4, :] = output2[:, 2:3, :]
        input3[:, 0:1, :] = output3[:, 0:1, :]
        input3[:, 2:4, :] = output3[:, 1:3, :]

        input_merge = ops.Concat(axis=1)([input2, input3])

        cls_output_merge = self.cls_classifier_merge(
            input_merge.transpose(0, 2, 1)
        ).transpose(0, 2, 1)
        # print(cls_output_merge.shape)
        
        cls_output2 = self.cls_classifier2(output2)
        cls_output3 = self.cls_classifier3(output3)
        cls_output = self.cls_classifier(cls_output_merge)
        #     print(cls_output.shape, cls_output2.shape, cls_output3.shape)
        #    (32, 4, 106) (32, 3, 106) (32, 3, 106)
        return cls_output2, cls_output3, cls_output

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
