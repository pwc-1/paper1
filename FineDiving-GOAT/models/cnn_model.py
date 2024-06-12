import numpy as np

from models.Backbone import *
from utils.goat_utils import *
from roi_align.roi_align import RoIAlign  # RoIAlign module

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops


class GCN_Module(nn.Cell):
    def __init__(self, args):
        super(GCN_Module, self).__init__()
        self.args = args

        # dimension and layers number
        NFR = args.num_features_relation  # dimension of features of phi(x) and theta(x) in 'Embedded Dot-Product'
        NG = args.num_graph
        NFG = args.num_features_gcn  # dimension of features of node in graphs
        NFG_ONE = NFG

        # set modules
        self.fc_rn_theta_list = nn.CellList([nn.Dense(NFG, NFR) for i in range(NG)])
        self.fc_rn_phi_list = nn.CellList([nn.Dense(NFG, NFR) for i in range(NG)])
        self.fc_gcn_list = nn.CellList([nn.Dense(NFG, NFG_ONE, has_bias=False) for i in range(NG)])
        self.nl_gcn_list = nn.CellList([nn.LayerNorm([NFG_ONE]) for i in range(NG)])

    def construct(self, graph_boxes_features, boxes_in_flat):
        """
        graph_boxes_features  [B*T,N,NFG]
        """

        # GCN graph modeling
        # Prepare boxes similarity relation
        B, N, NFG = graph_boxes_features.shape  # Note! B is actually B*T
        NFR = self.args.num_features_relation
        NG = self.args.num_graph
        NFG_ONE = NFG
        OH, OW = self.args.out_size
        pos_threshold = self.args.pos_threshold

        # Prepare position mask
        graph_boxes_positions = boxes_in_flat  # B*T*N, 4
        graph_boxes_positions[:, 0] = (graph_boxes_positions[:, 0] + graph_boxes_positions[:, 2]) / 2
        graph_boxes_positions[:, 1] = (graph_boxes_positions[:, 1] + graph_boxes_positions[:, 3]) / 2
        graph_boxes_positions = graph_boxes_positions[:, :2].reshape(B, N, 2)  # B*T, N, 2
        graph_boxes_distances = calc_pairwise_distance_3d(graph_boxes_positions, graph_boxes_positions)  # B(*T)?, N, N
        position_mask = (graph_boxes_distances > (pos_threshold * OW))

        # GCN list
        relation_graph = None
        graph_boxes_features_list = []
        for i in range(NG):
            # calculate similarity
            graph_boxes_features_theta = self.fc_rn_theta_list[i](graph_boxes_features)  # B*T,N,NFR
            graph_boxes_features_phi = self.fc_rn_phi_list[i](graph_boxes_features)  # B*T,N,NFR
            similarity_relation_graph = ops.matmul(graph_boxes_features_theta, graph_boxes_features_phi.swapaxes(1, 2))  # B*T,N,N
            similarity_relation_graph = similarity_relation_graph / np.sqrt(NFR)
            similarity_relation_graph = similarity_relation_graph.reshape(-1, 1)  # B*T*N*N, 1

            # Build relation graph
            relation_graph = similarity_relation_graph
            relation_graph = relation_graph.reshape(B, N, N)  # B*T,N,N
            relation_graph[position_mask] = -float('inf')
            relation_graph = ops.softmax(relation_graph, axis=2)

            # Graph convolution
            one_graph_boxes_features = self.fc_gcn_list[i](
                ops.matmul(relation_graph, graph_boxes_features))  # B, N, NFG_ONE  # G*X*W
            one_graph_boxes_features = self.nl_gcn_list[i](one_graph_boxes_features)
            one_graph_boxes_features = ops.relu(one_graph_boxes_features)  # ReLu(G*X*W)
            graph_boxes_features_list.append(one_graph_boxes_features)

        # fuse multi graphs
        graph_boxes_features = ops.sum(ops.stack(graph_boxes_features_list), dim=0)  # B*T, N, NFG

        return graph_boxes_features, relation_graph


class GCNnet_artisticswimming(nn.Cell):
    """
    main module of GCN for the volleyball dataset
    """

    def __init__(self, args):
        super(GCNnet_artisticswimming, self).__init__()
        self.args = args

        # set parameters
        N = self.args.num_boxes
        D = self.args.emb_features  # output feature map channel of backbone
        K = self.args.crop_size[0]  # crop size of roi align
        NFB = self.args.num_features_boxes
        NFR, NFG = self.args.num_features_relation, self.args.num_features_gcn

        # set backbone
        self.backbone = MyInception_v3(transform_input=False, pretrained=True)
        if not args.train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # set modules
        self.roi_align = RoIAlign(*self.args.crop_size)
        self.fc_emb_1 = nn.Dense(K * K * D, NFB)  # change dimension of backbone features
        self.nl_emb_1 = nn.LayerNorm([NFB])
        self.gcn_list = nn.CellList([GCN_Module(args) for i in range(self.args.gcn_layers)])
        self.dropout_global = nn.Dropout(p=self.args.train_dropout_prob)

        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Dense):
                cell.weight.set_data(ms.common.initializer.initializer(
                    ms.common.initializer.HeNormal(), cell.weight.shape, cell.weight.dtype))
                if cell.has_bias:
                    cell.bias.set_data(ms.common.initializer.initializer("zeros", cell.bias.shape, cell.bias.dtype))

    def loadmodel(self, filepath):
        state = ms.load(filepath)
        self.backbone.load_state_dict(state['backbone_state_dict'])
        self.fc_emb_1.load_state_dict(state['fc_emb_state_dict'])
        if is_main_process():
            print('Load model states from: ', filepath)

    def savemodel(self, filepath):
        ms.save_checkpoint(self.state_dict(), filepath)
        print('model saved to:', filepath)

    def construct(self, images_in, boxes_in):

        # read config parameters
        B = images_in.shape[0]
        T = images_in.shape[1]
        t = self.args.num_selected_frames
        H, W = self.args.img_size
        OH, OW = self.args.out_size  # output feature map size of backbone
        N = self.args.num_boxes
        NFB = self.args.num_features_boxes
        NFR, NFG = self.args.num_features_relation, self.args.num_features_gcn
        NG = self.args.num_graph
        D = self.args.emb_features
        K = self.args.crop_size[0]

        # Reshape the input data
        images_in_flat = ops.reshape(images_in, (B * T, 3, H, W))  # B*T, 3, H, W
        boxes_in_flat = ops.reshape(boxes_in, (B * T * N, 4))  # B*T*N, 4

        # build box matrix for RoIAlign
        boxes_idx = [i * ops.ones(N, dtype=ms.int_) for i in range(B * T)]
        boxes_idx = ops.stack(boxes_idx).to(device=boxes_in.device)  # B*T, N
        boxes_idx_flat = ops.reshape(boxes_idx, (B * T * N,))  # B*T*N,

        # Use backbone to extract features of images_in
        outputs = self.backbone(images_in_flat)

        # Build  features
        assert outputs[0].shape[2:4] == (OH, OW)
        features_multiscale = []
        for features in outputs:  # B*T, D // 2, OH, OW
            if features.shape[2:4] != (OH, OW):
                features = ops.interpolate(features, size=(OH, OW), mode='bilinear', align_corners=True)
            features_multiscale.append(features)
        features_multiscale = ops.cat(features_multiscale, axis=1)  # B*T, D, OH, OW

        # RoI Align
        boxes_in_flat.requires_grad = False
        boxes_idx_flat.requires_grad = False
        boxes_features = self.roi_align(features_multiscale,
                                        boxes_in_flat,
                                        boxes_idx_flat)  # B*T*N, D, K, K,
        boxes_features = boxes_features.reshape(B, T, N, -1)  # B,T,N, D*K*K

        # Embedding
        boxes_features = self.fc_emb_1(boxes_features)  # B,T,N, NFB
        boxes_features = self.nl_emb_1(boxes_features)
        boxes_features = ops.relu(boxes_features)

        # GCN
        if self.args.gcn_temporal_fuse:
            graph_boxes_features = boxes_features.reshape(B, T * N, NFG)
        else:
            graph_boxes_features = boxes_features.reshape(B * T, N, NFG)
        for i in range(len(self.gcn_list)):
            graph_boxes_features, relation_graph = self.gcn_list[i](graph_boxes_features, boxes_in_flat)

        # fuse graph_boxes_features with boxes_features
        graph_boxes_features = graph_boxes_features.reshape(B, T, N, NFG)
        boxes_features = boxes_features.reshape(B, T, N, NFB)
        boxes_states = graph_boxes_features + boxes_features

        # reshape as query
        boxes_states = boxes_states.reshape(B, -1, t, N, NFG)
        boxes_states = boxes_states.mean(2).mean(2)  # B,60,1024

        return boxes_states
