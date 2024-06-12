import mindspore as ms
import mindspore.ops as ops

import numpy as np


class Trainer:
    def __init__(self,base_model, regressor, group, mse, nll, optimizer, args, gcn, attn_encoder, linear_bp):
        self.base_model = base_model
        self.regressor = regressor
        self.group = group
        self.mse = mse
        self.nll = nll
        self.optimizer = optimizer
        self.args = args
        self.gcn = gcn
        self.attn_encoder = attn_encoder
        self.linear_bp = linear_bp
        self.grads = None
        self.grad_fn = ms.value_and_grad(self.forward_fn, None, optimizer.parameters, has_aux=True)

    def set_train(self):
        # base_model.train()
        self.regressor.set_train()
        self.linear_bp.set_train()
        if self.args.use_goat:
            self.gcn.set_train()
            self.attn_encoder.set_train()

    def set_test(self):
        # base_model.train()
        self.regressor.set_train(False)
        self.linear_bp.set_train(False)
        if self.args.use_goat:
            self.gcn.set_train(False)
            self.attn_encoder.set_train(False)

    def forward_fn(self, feature_1, label_1, feature_2, label_2, data, target):
            loss = 0.0
            label = [label_1, label_2]
            theta = self.args.score_range
            if not self.args.use_i3d_bb:
                feature_1 = self.linear_bp(feature_1)  # B,540,1024
                feature_2 = self.linear_bp(feature_2)  # B,540,1024

            ######### GOAT START ##########
            if self.args.use_goat:
                if self.args.use_formation:
                    q1 = data['formation_features']  # B,540,1024
                    k1 = q1
                    feature_1 = self.attn_encoder(q1, k1, feature_1)
                    feature_1 = feature_1.mean(1)  # B,1024

                    q2 = target['formation_features']  # B,540,1024
                    k2 = q2
                    feature_2 = self.attn_encoder(q2, k2, feature_2)
                    feature_2 = feature_2.mean(1)  # B,1024
                elif self.args.use_bp:
                    q1 = data['bp_features']  # B,540,768
                    k1 = q1
                    feature_1 = self.attn_encoder(q1, k1, feature_1)
                    feature_1 = feature_1.mean(1)  # B,1024

                    q2 = target['bp_features']  # B,540,1024
                    k2 = q2
                    feature_2 = self.attn_encoder(q2, k2, feature_2)
                    feature_2 = feature_2.mean(1)  # B,1024
                elif self.args.use_self:
                    q1 = feature_1
                    k1 = q1
                    feature_1 = self.attn_encoder(q1, k1, feature_1)
                    feature_1 = feature_1.mean(1)  # B,1024

                    q2 = feature_2  # B,540,1024
                    k2 = q2
                    feature_2 = self.attn_encoder(q2, k2, feature_2)
                    feature_2 = feature_2.mean(1)  # B,1024
                else:
                    if self.args.use_cnn_features:
                        boxes_features_1 = data['cnn_features']
                        boxes_in_1 = data['boxes']  # B,T,N,4
                        q1 = self.gcn(boxes_features_1, boxes_in_1)  # B,540,1024
                        k1 = q1
                        feature_1 = self.attn_encoder(q1, k1, feature_1)  # B,540,1024
                        feature_1 = feature_1.mean(1)  # B,1024

                        boxes_features_2 = target['cnn_features']
                        boxes_in_2 = target['boxes']  # B,T,N,4
                        q2 = self.gcn(boxes_features_2, boxes_in_2)  # B,540,2024
                        k2 = q2
                        feature_2 = self.attn_encoder(q2, k2, feature_2)  # B,540,2024
                        feature_2 = feature_2.mean(1)  # B,2024
                    else:
                        images_in_1 = data['video']  # B,T,C,H,W
                        boxes_in_1 = data['boxes']  # B,T,N,4
                        q1 = self.gcn(images_in_1, boxes_in_1)  # B,540,1024
                        k1 = q1
                        feature_1 = self.attn_encoder(q1, k1, feature_1)  # B,540,1024
                        feature_1 = feature_1.mean(1)  # B,1024

                        images_in_2 = target['video']  # B,T,C,H,W
                        boxes_in_2 = target['boxes']  # B,T,N,4
                        q2 = self.gcn(images_in_2, boxes_in_2)  # B,540,2024
                        k2 = q2
                        feature_2 = self.attn_encoder(q2, k2, feature_2)  # B,540,2024
                        feature_2 = feature_2.mean(1)  # B,2024

            #########  GOAT END  ##########
            else:
                total_feature = ops.cat((feature_1, feature_2), 0).mean(1)  # 2B,1024
                feature_1 = total_feature[:total_feature.shape[0] // 2]  # B,1024
                feature_2 = total_feature[total_feature.shape[0] // 2:]  # B,1024

            combined_feature_1 = ops.cat((feature_1, feature_2, label[0] / theta), 1)  # 1 is exemplar N * 2049
            combined_feature_2 = ops.cat((feature_2, feature_1, label[1] / theta), 1)  # 2 is exemplar N * 2049

            combined_feature = ops.cat((combined_feature_1, combined_feature_2), 0)  # 2N * 2049
            out_prob, delta = self.regressor(combined_feature)
            # tree-level label
            glabel_1, rlabel_1 = self.group.produce_label(label_2 - label_1)
            glabel_2, rlabel_2 = self.group.produce_label(label_1 - label_2)
            # predictions
            leaf_probs = out_prob[-1].reshape(combined_feature.shape[0], -1)
            leaf_probs_1 = leaf_probs[:leaf_probs.shape[0] // 2]
            leaf_probs_2 = leaf_probs[leaf_probs.shape[0] // 2:]
            delta_1 = delta[:delta.shape[0] // 2]
            delta_2 = delta[delta.shape[0] // 2:]
            # loss
            loss += self.nll(leaf_probs_1, glabel_1.argmax(0))
            loss += self.nll(leaf_probs_2, glabel_2.argmax(0))
            for i in range(self.group.number_leaf()):
                mask = rlabel_1[i] >= 0
                if mask.sum() != 0:
                    loss += self.mse(delta_1[:, i][mask].reshape(-1, 1).float(), rlabel_1[i][mask].reshape(-1, 1).float())
                mask = rlabel_2[i] >= 0
                if mask.sum() != 0:
                    loss += self.mse(delta_2[:, i][mask].reshape(-1, 1).float(), rlabel_2[i][mask].reshape(-1, 1).float())

            return loss, leaf_probs_2, delta_2

    def train_epoch(self, feature_1, label_1, feature_2, label_2, data, target, opti_flag):
        (loss, leaf_probs_2, delta_2), grads = self.grad_fn(feature_1, label_1, feature_2, label_2, data, target)
        if self.grads is None:
            self.grads = grads
        else:
            self.grads = tuple(x + y for x, y in zip(self.grads, grads))
        if opti_flag:
            self.optimizer(self.grads)
        return loss, leaf_probs_2, delta_2

