import mindspore as ms
import mindspore.ops as ops

import numpy as np
from utils.misc import segment_iou, cal_tiou, seg_pool_1d, seg_pool_3d


class Trainer:
    def __init__(self, base_model, psnet_model, decoder, regressor_delta, mse, optimizer, args, bce, gcn, attn_encoder, linear_bp):
        self.base_model = base_model
        self.psnet_model = psnet_model
        self.decoder = decoder
        self.regressor_delta = regressor_delta
        self.mse = mse
        self.optimizer = optimizer
        self.args = args
        self.bce = bce
        self.gcn = gcn
        self.attn_encoder = attn_encoder
        self.linear_bp = linear_bp
        self.grad_fn = ms.value_and_grad(self.forward_fn, None, optimizer.parameters, has_aux=True)

    def set_train(self):
        # base_model.train()
        self.psnet_model.set_train()
        self.decoder.set_train()
        self.regressor_delta.set_train()
        self.linear_bp.set_train()
        if self.args.use_goat:
            self.gcn.set_train()
            self.attn_encoder.set_train()

    def set_test(self):
        # base_model.train()
        self.psnet_model.set_train(False)
        self.decoder.set_train(False)
        self.regressor_delta.set_train(False)
        self.linear_bp.set_train(False)
        if self.args.use_goat:
            self.gcn.set_train(False)
            self.attn_encoder.set_train(False)

    def forward_fn(self, feature_1, label_1_score, feature_2, label_2_score, epoch, label_1_tas, label_2_tas, feamap_1, feamap_2, data, target):
        ############# I3D featrue #############
        N, T, C, T_t, H_t, W_t = (self.args.bs_train, 9, 1024, 2, 4, 4)
        N = feature_1.shape[0]
        if not self.args.use_i3d_bb:
            feature_1 = self.linear_bp(feature_1)  # B,540,1024
            feature_2 = self.linear_bp(feature_2)  # B,540,1024

        # goat
        if self.args.use_goat:
            if self.args.use_formation:
                video_1_fea = []
                video_2_fea = []
                video_1_fea_list = [feature_1[:, i:i + 60] for i in range(0, 540, 60)]  # [B,60,1024]
                video_2_fea_list = [feature_2[:, i:i + 60] for i in range(0, 540, 60)]  # [B,60,1024]
                formation_features_1 = data['formation_features']  # B,540,1024
                formation_features_1_list = [formation_features_1[:, i:i + 60] for i in
                                             range(0, 540, 60)]  # [B,60,1024]
                formation_features_2 = target['formation_features']  # B,540,1024
                formation_features_2_list = [formation_features_2[:, i:i + 60] for i in
                                             range(0, 540, 60)]  # [B,60,1024]

                for i in range(9):
                    q1 = formation_features_1_list[i]
                    k1 = q1
                    feature_1_tmp = self.attn_encoder(q1, k1, video_1_fea_list[i])  # B,60,1024
                    video_1_fea.append(feature_1_tmp.mean(1).unsqueeze(1))  # [B,1,1024]

                    q2 = formation_features_2_list[i]
                    k2 = q2
                    feature_2_tmp = self.attn_encoder(q2, k2, video_2_fea_list[i])  # B,60,1024
                    video_2_fea.append(feature_2_tmp.mean(1).unsqueeze(1))  # [B,1,1024]
                video_1_fea = ops.cat(video_1_fea, axis=1)  # B,9,1024
                video_2_fea = ops.cat(video_2_fea, axis=1)  # B,9,1024
            elif self.args.use_bp:
                video_1_fea = []
                video_2_fea = []
                video_1_fea_list = [feature_1[:, i:i + 60] for i in range(0, 540, 60)]  # [B,60,1024]
                video_2_fea_list = [feature_2[:, i:i + 60] for i in range(0, 540, 60)]  # [B,60,1024]
                bp_features_1 = data['bp_features']  # B,540,768
                bp_features_1_list = [bp_features_1[:, i:i + 60] for i in range(0, 540, 60)]  # [B,60,768]
                bp_features_2 = target['bp_features']  # B,540,768
                bp_features_2_list = [bp_features_2[:, i:i + 60] for i in range(0, 540, 60)]  # [B,60,768]

                for i in range(9):
                    q1 = bp_features_1_list[i]
                    k1 = q1
                    feature_1_tmp = self.attn_encoder(q1, k1, video_1_fea_list[i])  # B,60,1024
                    video_1_fea.append(feature_1_tmp.mean(1).unsqueeze(1))  # [B,1,1024]

                    q2 = bp_features_2_list[i]
                    k2 = q2
                    feature_2_tmp = self.attn_encoder(q2, k2, video_2_fea_list[i])  # B,60,1024
                    video_2_fea.append(feature_2_tmp.mean(1).unsqueeze(1))  # [B,1,1024]
                video_1_fea = ops.cat(video_1_fea, axis=1)  # B,9,1024
                video_2_fea = ops.cat(video_2_fea, axis=1)  # B,9,1024
            elif self.args.use_self:
                video_1_fea = []
                video_2_fea = []
                video_1_fea_list = [feature_1[:, i:i + 60] for i in range(0, 540, 60)]  # [B,60,1024]
                video_2_fea_list = [feature_2[:, i:i + 60] for i in range(0, 540, 60)]  # [B,60,1024]

                for i in range(9):
                    q1 = video_1_fea_list[i]
                    k1 = q1
                    feature_1_tmp = self.attn_encoder(q1, k1, video_1_fea_list[i])  # B,60,1024
                    video_1_fea.append(feature_1_tmp.mean(1).unsqueeze(1))  # [B,1,1024]

                    q2 = video_2_fea_list[i]
                    k2 = q2
                    feature_2_tmp = self.attn_encoder(q2, k2, video_2_fea_list[i])  # B,60,1024
                    video_2_fea.append(feature_2_tmp.mean(1).unsqueeze(1))  # [B,1,1024]
                video_1_fea = ops.cat(video_1_fea, axis=1)  # B,9,1024
                video_2_fea = ops.cat(video_2_fea, axis=1)  # B,9,1024
            else:
                if self.args.use_cnn_features:
                    # video1
                    video_1_fea = []
                    video_1_fea_list = [feature_1[:, i:i + 60] for i in range(0, 540, 60)]  # [B,60,1024]
                    boxes_features_1 = data['cnn_features']
                    boxes_features_1_list = [boxes_features_1[:, i:i + 60] for i in range(0, 540, 60)]  # [B,60,N,4]
                    boxes_in_1 = data['boxes']  # B,T,N,4
                    boxes_in_1_list = [boxes_in_1[:, i:i + 60] for i in range(0, 540, 60)]  # [B,60,N,4]

                    # video2
                    video_2_fea = []
                    video_2_fea_list = [feature_2[:, i:i + 60] for i in range(0, 540, 60)]  # [B,60,1024]
                    boxes_features_2 = target['cnn_features']
                    boxes_features_2_list = [boxes_features_2[:, i:i + 60] for i in range(0, 540, 60)]  # [B,60,N,4]
                    boxes_in_2 = target['boxes']  # B,T,N,4
                    boxes_in_2_list = [boxes_in_2[:, i:i + 60] for i in range(0, 540, 60)]  # [B,60,N,4]

                    for i in range(9):
                        q1 = self.gcn(boxes_features_1_list[i], boxes_in_1_list[i])  # B,60,1024
                        k1 = q1
                        feature_1_tmp = self.attn_encoder(q1, k1, video_1_fea_list[i])  # B,60,1024
                        video_1_fea.append(feature_1_tmp.mean(1).unsqueeze(1))  # [B,1,1024]

                        q2 = self.gcn(boxes_features_2_list[i], boxes_in_2_list[i])  # B,60,1024
                        k2 = q2
                        feature_2_tmp = self.attn_encoder(q2, k2, video_2_fea_list[i])  # B,60,1024
                        video_2_fea.append(feature_2_tmp.mean(1).unsqueeze(1))  # [B,1,1024]
                    video_1_fea = ops.cat(video_1_fea, axis=1)  # B,9,1024
                    video_2_fea = ops.cat(video_2_fea, axis=1)  # B,9,1024
                else:
                    # video1
                    video_1_fea = []
                    video_1_fea_list = [feature_1[:, i:i + 60] for i in range(0, 540, 60)]  # [B,60,1024]
                    images_in_1 = data['video']  # B,T,C,H,W
                    images_in_1_list = [images_in_1[:, i:i + 60] for i in range(0, 540, 60)]  # [B,60,C,H,W]
                    boxes_in_1 = data['boxes']  # B,T,N,4
                    boxes_in_1_list = [boxes_in_1[:, i:i + 60] for i in range(0, 540, 60)]  # [B,60,N,4]
                    # video2
                    video_2_fea = []
                    video_2_fea_list = [feature_2[:, i:i + 60] for i in range(0, 540, 60)]  # [B,60,1024]
                    images_in_2 = target['video']  # B,T,C,H,W
                    images_in_2_list = [images_in_2[:, i:i + 60] for i in range(0, 540, 60)]  # [B,60,C,H,W]
                    boxes_in_2 = target['boxes']  # B,T,N,4
                    boxes_in_2_list = [boxes_in_2[:, i:i + 60] for i in range(0, 540, 60)]  # [B,60,N,4]
                    for i in range(9):
                        q1 = self.gcn(images_in_1_list[i], boxes_in_1_list[i])  # B,60,1024
                        k1 = q1
                        feature_1_tmp = self.attn_encoder(q1, k1, video_1_fea_list[i])  # B,60,1024
                        video_1_fea.append(feature_1_tmp.mean(1).unsqueeze(1))  # [B,1,1024]

                        q2 = self.gcn(images_in_2_list[i], boxes_in_2_list[i])  # B,60,1024
                        k2 = q2
                        feature_2_tmp = self.attn_encoder(q2, k2, video_2_fea_list[i])  # B,60,1024
                        video_2_fea.append(feature_2_tmp.mean(1).unsqueeze(1))  # [B,1,1024]
                    video_1_fea = ops.cat(video_1_fea, axis=1)  # B,9,1024
                    video_2_fea = ops.cat(video_2_fea, axis=1)  # B,9,1024
        else:
            video_1_fea = ops.cat([feature_1[:, i:i + 60].mean(1).unsqueeze(1) for i in range(0, 540, 60)],
                                  1)  # N,9,1024
            video_2_fea = ops.cat([feature_2[:, i:i + 60].mean(1).unsqueeze(1) for i in range(0, 540, 60)],
                                  1)  # N,9,1024
        video_1_feamap_re = ops.cat([feamap_1[:, i:i + 60].mean(1).unsqueeze(1).mean(-3) for i in range(0, 540, 60)],
                                    1).reshape(-1, 9, 1024)
        video_2_feamap_re = ops.cat([feamap_2[:, i:i + 60].mean(1).unsqueeze(1).mean(-3) for i in range(0, 540, 60)],
                                    1).reshape(-1, 9, 1024)

        ############# Procedure Segmentation #############
        com_feature_12_u = ops.cat((video_1_fea, video_2_fea), 0)  # (2N, 9, 1024)
        com_feamap_12_u = ops.cat((video_1_feamap_re, video_2_feamap_re), 0)  # (32N, 9, 1024)

        u_fea_96, transits_pred = self.psnet_model(com_feature_12_u)
        u_feamap_96, transits_pred_map = self.psnet_model(com_feamap_12_u)
        u_feamap_96 = u_feamap_96.reshape(2 * N, u_feamap_96.shape[1], u_feamap_96.shape[2], H_t, W_t)

        label_12_tas = ops.cat((label_1_tas, label_2_tas), 0)
        label_12_pad = ops.zeros(transits_pred.shape)
        # one-hot
        for bs in range(transits_pred.shape[0]):
            label_12_pad[bs, int(label_12_tas[bs, 0]), 0] = 1
            label_12_pad[bs, int(label_12_tas[bs, -1]), -1] = 1

        loss_tas = self.bce(transits_pred, label_12_pad)

        num = round(transits_pred.shape[1] / transits_pred.shape[-1])
        transits_st_ed = ops.zeros(label_12_tas.shape)
        for bs in range(transits_pred.shape[0]):
            for i in range(transits_pred.shape[-1]):
                transits_st_ed[bs, i] = transits_pred[bs, i * num: (i + 1) * num, i].argmax(0).item() + i * num
        label_1_tas_pred = transits_st_ed[:transits_st_ed.shape[0] // 2]
        label_2_tas_pred = transits_st_ed[transits_st_ed.shape[0] // 2:]

        ############# Procedure-aware Cross-attention #############
        u_fea_96_1 = u_fea_96[:u_fea_96.shape[0] // 2].swapaxes(1, 2)
        u_fea_96_2 = u_fea_96[u_fea_96.shape[0] // 2:].swapaxes(1, 2)

        u_feamap_96_1 = u_feamap_96[:u_feamap_96.shape[0] // 2].swapaxes(1, 2)
        u_feamap_96_2 = u_feamap_96[u_feamap_96.shape[0] // 2:].swapaxes(1, 2)

        if epoch / self.args.max_epoch <= self.args.prob_tas_threshold:
            video_1_segs = []
            for bs_1 in range(u_fea_96_1.shape[0]):
                video_1_st = int(label_1_tas[bs_1][0].item())
                video_1_ed = int(label_1_tas[bs_1][1].item())
                video_1_segs.append(seg_pool_1d(u_fea_96_1[bs_1].unsqueeze(0), video_1_st, video_1_ed, self.args.fix_size))
            video_1_segs = ops.cat(video_1_segs, 0).swapaxes(1, 2)

            video_2_segs = []
            for bs_2 in range(u_fea_96_2.shape[0]):
                video_2_st = int(label_2_tas[bs_2][0].item())
                video_2_ed = int(label_2_tas[bs_2][1].item())
                video_2_segs.append(seg_pool_1d(u_fea_96_2[bs_2].unsqueeze(0), video_2_st, video_2_ed, self.args.fix_size))
            video_2_segs = ops.cat(video_2_segs, 0).swapaxes(1, 2)

            video_1_segs_map = []
            for bs_1 in range(u_feamap_96_1.shape[0]):
                video_1_st = int(label_1_tas[bs_1][0].item())
                video_1_ed = int(label_1_tas[bs_1][1].item())
                video_1_segs_map.append(
                    seg_pool_3d(u_feamap_96_1[bs_1].unsqueeze(0), video_1_st, video_1_ed, self.args.fix_size))
            video_1_segs_map = ops.cat(video_1_segs_map, 0)
            video_1_segs_map = video_1_segs_map.reshape(video_1_segs_map.shape[0], video_1_segs_map.shape[1],
                                                        video_1_segs_map.shape[2], -1).swapaxes(2, 3)
            video_1_segs_map = ops.cat([video_1_segs_map[:, :, :, i] for i in range(video_1_segs_map.shape[-1])],
                                       2).swapaxes(1, 2)

            video_2_segs_map = []
            for bs_2 in range(u_fea_96_2.shape[0]):
                video_2_st = int(label_2_tas[bs_2][0].item())
                video_2_ed = int(label_2_tas[bs_2][1].item())
                video_2_segs_map.append(
                    seg_pool_3d(u_feamap_96_2[bs_2].unsqueeze(0), video_2_st, video_2_ed, self.args.fix_size))
            video_2_segs_map = ops.cat(video_2_segs_map, 0)
            video_2_segs_map = video_2_segs_map.reshape(video_2_segs_map.shape[0], video_2_segs_map.shape[1],
                                                        video_2_segs_map.shape[2], -1).swapaxes(2, 3)
            video_2_segs_map = ops.cat([video_2_segs_map[:, :, :, i] for i in range(video_2_segs_map.shape[-1])],
                                       2).swapaxes(1, 2)
        else:
            video_1_segs = []
            for bs_1 in range(u_fea_96_1.shape[0]):
                video_1_st = int(label_1_tas_pred[bs_1][0].item())
                video_1_ed = int(label_1_tas_pred[bs_1][1].item())
                if video_1_st == 0:
                    video_1_st = 1
                if video_1_ed == 0:
                    video_1_ed = 1
                video_1_segs.append(seg_pool_1d(u_fea_96_1[bs_1].unsqueeze(0), video_1_st, video_1_ed, self.args.fix_size))
            video_1_segs = ops.cat(video_1_segs, 0).swapaxes(1, 2)

            video_2_segs = []
            for bs_2 in range(u_fea_96_2.shape[0]):
                video_2_st = int(label_2_tas_pred[bs_2][0].item())
                video_2_ed = int(label_2_tas_pred[bs_2][1].item())
                if video_2_st == 0:
                    video_2_st = 1
                if video_2_ed == 0:
                    video_2_ed = 1
                video_2_segs.append(seg_pool_1d(u_fea_96_2[bs_2].unsqueeze(0), video_2_st, video_2_ed, self.args.fix_size))
            video_2_segs = ops.cat(video_2_segs, 0).swapaxes(1, 2)

            video_1_segs_map = []
            for bs_1 in range(u_feamap_96_1.shape[0]):
                video_1_st = int(label_1_tas_pred[bs_1][0].item())
                video_1_ed = int(label_1_tas_pred[bs_1][1].item())
                if video_1_st == 0:
                    video_1_st = 1
                if video_1_ed == 0:
                    video_1_ed = 1
                video_1_segs_map.append(
                    seg_pool_3d(u_feamap_96_1[bs_1].unsqueeze(0), video_1_st, video_1_ed, self.args.fix_size))
            video_1_segs_map = ops.cat(video_1_segs_map, 0)
            video_1_segs_map = video_1_segs_map.reshape(video_1_segs_map.shape[0], video_1_segs_map.shape[1],
                                                        video_1_segs_map.shape[2], -1).swapaxes(2, 3)
            video_1_segs_map = ops.cat([video_1_segs_map[:, :, :, i] for i in range(video_1_segs_map.shape[-1])],
                                       2).swapaxes(1, 2)

            video_2_segs_map = []
            for bs_2 in range(u_fea_96_2.shape[0]):
                video_2_st = int(label_2_tas_pred[bs_2][0].item())
                video_2_ed = int(label_2_tas_pred[bs_2][1].item())
                if video_2_st == 0:
                    video_2_st = 1
                if video_2_ed == 0:
                    video_2_ed = 1
                video_2_segs_map.append(
                    seg_pool_3d(u_feamap_96_2[bs_2].unsqueeze(0), video_2_st, video_2_ed, self.args.fix_size))
            video_2_segs_map = ops.cat(video_2_segs_map, 0)
            video_2_segs_map = video_2_segs_map.reshape(video_2_segs_map.shape[0], video_2_segs_map.shape[1],
                                                        video_2_segs_map.shape[2], -1).swapaxes(2, 3)
            video_2_segs_map = ops.cat([video_2_segs_map[:, :, :, i] for i in range(video_2_segs_map.shape[-1])],
                                       2).swapaxes(1, 2)

        decoder_video_12_map_list = []
        decoder_video_21_map_list = []
        for i in range(self.args.step_num):
            decoder_video_12_map = self.decoder(video_1_segs[:, i * self.args.fix_size:(i + 1) * self.args.fix_size, :],
                                           video_2_segs_map[:,
                                           i * self.args.fix_size * H_t * W_t:(i + 1) * self.args.fix_size * H_t * W_t,
                                           :])  # N,15,256/64
            decoder_video_21_map = self.decoder(video_2_segs[:, i * self.args.fix_size:(i + 1) * self.args.fix_size, :],
                                           video_1_segs_map[:,
                                           i * self.args.fix_size * H_t * W_t:(i + 1) * self.args.fix_size * H_t * W_t,
                                           :])  # N,15,256/64
            decoder_video_12_map_list.append(decoder_video_12_map)
            decoder_video_21_map_list.append(decoder_video_21_map)

        decoder_video_12_map = ops.cat(decoder_video_12_map_list, 1)
        decoder_video_21_map = ops.cat(decoder_video_21_map_list, 1)

        ############# Fine-grained Contrastive Regression #############
        decoder_12_21 = ops.cat((decoder_video_12_map, decoder_video_21_map), 0)
        delta = self.regressor_delta(decoder_12_21)
        delta = delta.mean(1)
        loss_aqa = self.mse(delta[:delta.shape[0] // 2], (label_1_score - label_2_score)) \
                   + self.mse(delta[delta.shape[0] // 2:], (label_2_score - label_1_score))

        loss = loss_aqa + loss_tas
        score = (delta[:delta.shape[0] // 2] + label_2_score)

        tIoU_results = []
        for bs in range(transits_pred.shape[0] // 2):
            tIoU_results.append(segment_iou(np.array(label_12_tas)[bs],
                                            np.array(transits_st_ed)[bs],
                                            self.args))

        tiou_thresholds = np.array([0.5, 0.75])
        tIoU_correct_per_thr = cal_tiou(tIoU_results, tiou_thresholds)
        Batch_tIoU_5 = tIoU_correct_per_thr[0]
        Batch_tIoU_75 = tIoU_correct_per_thr[1]
        return loss, score, Batch_tIoU_5, Batch_tIoU_75

    def train_epoch(self, feature_1, label_1_score, feature_2, label_2_score, epoch, label_1_tas, label_2_tas, feamap_1, feamap_2, data, target):
        (loss, score, Batch_tIoU_5, Batch_tIoU_75), grads = self.grad_fn(feature_1, label_1_score, feature_2,
                                                                       label_2_score, epoch, label_1_tas, label_2_tas,
                                                                       feamap_1, feamap_2, data, target)
        self.optimizer(grads)
        return loss, score, Batch_tIoU_5, Batch_tIoU_75

