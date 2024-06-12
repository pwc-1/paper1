import os, sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "../"))

import mindspore as ms
import mindspore.ops as ops
import time
import numpy as np


def network_forward_train(base_model, regressor, pred_scores, feature_1, label_1, feature_2, label_2, diff, group, mse,
                          nll, optimizer, opti_flag, epoch, batch_idx, batch_num, args, data, target, gcn, attn_encoder, linear_bp):
    loss = 0.0
    start = time.time()

    # TODO: theta
    label = [label_1, label_2]
    theta = args.score_range
    if not args.use_i3d_bb:
        feature_1 = linear_bp(feature_1)  # B,540,1024
        feature_2 = linear_bp(feature_2)  # B,540,1024

    ######### GOAT START ##########
    if args.use_goat:
        if args.use_formation:
            q1 = data['formation_features']  # B,540,1024
            k1 = q1
            feature_1 = attn_encoder(q1, k1, feature_1)
            feature_1 = feature_1.mean(1)  # B,1024

            q2 = target['formation_features']  # B,540,1024
            k2 = q2
            feature_2 = attn_encoder(q2, k2, feature_2)
            feature_2 = feature_2.mean(1)  # B,1024
        elif args.use_bp:
            q1 = data['bp_features']  # B,540,768
            k1 = q1
            feature_1 = attn_encoder(q1, k1, feature_1)
            feature_1 = feature_1.mean(1)  # B,1024

            q2 = target['bp_features']  # B,540,1024
            k2 = q2
            feature_2 = attn_encoder(q2, k2, feature_2)
            feature_2 = feature_2.mean(1)  # B,1024
        elif args.use_self:
            q1 = feature_1
            k1 = q1
            feature_1 = attn_encoder(q1, k1, feature_1)
            feature_1 = feature_1.mean(1)  # B,1024

            q2 = feature_2  # B,540,1024
            k2 = q2
            feature_2 = attn_encoder(q2, k2, feature_2)
            feature_2 = feature_2.mean(1)  # B,1024
        else:
            if args.use_cnn_features:
                boxes_features_1 = data['cnn_features']
                boxes_in_1 = data['boxes']  # B,T,N,4
                q1 = gcn(boxes_features_1, boxes_in_1)  # B,540,1024
                k1 = q1
                feature_1 = attn_encoder(q1, k1, feature_1)  # B,540,1024
                feature_1 = feature_1.mean(1)  # B,1024

                boxes_features_2 = target['cnn_features']
                boxes_in_2 = target['boxes']  # B,T,N,4
                q2 = gcn(boxes_features_2, boxes_in_2)  # B,540,2024
                k2 = q2
                feature_2 = attn_encoder(q2, k2, feature_2)  # B,540,2024
                feature_2 = feature_2.mean(1)  # B,2024
            else:
                images_in_1 = data['video']  # B,T,C,H,W
                boxes_in_1 = data['boxes']  # B,T,N,4
                q1 = gcn(images_in_1, boxes_in_1)  # B,540,1024
                k1 = q1
                feature_1 = attn_encoder(q1, k1, feature_1)  # B,540,1024
                feature_1 = feature_1.mean(1)  # B,1024

                images_in_2 = target['video']  # B,T,C,H,W
                boxes_in_2 = target['boxes']  # B,T,N,4
                q2 = gcn(images_in_2, boxes_in_2)  # B,540,2024
                k2 = q2
                feature_2 = attn_encoder(q2, k2, feature_2)  # B,540,2024
                feature_2 = feature_2.mean(1)  # B,2024

    #########  GOAT END  ##########
    else:
        total_feature = ops.cat((feature_1, feature_2), 0).mean(1)  # 2B,1024
        feature_1 = total_feature[:total_feature.shape[0] // 2]  # B,1024
        feature_2 = total_feature[total_feature.shape[0] // 2:]  # B,1024


    combined_feature_1 = ops.cat((feature_1, feature_2, label[0] / theta), 1)  # 1 is exemplar N * 2049
    combined_feature_2 = ops.cat((feature_2, feature_1, label[1] / theta), 1)  # 2 is exemplar N * 2049

    combined_feature = ops.cat((combined_feature_1, combined_feature_2), 0)  # 2N * 2049
    out_prob, delta = regressor(combined_feature)
    # tree-level label
    glabel_1, rlabel_1 = group.produce_label(label_2 - label_1)
    glabel_2, rlabel_2 = group.produce_label(label_1 - label_2)
    # predictions
    leaf_probs = out_prob[-1].reshape(combined_feature.shape[0], -1)
    leaf_probs_1 = leaf_probs[:leaf_probs.shape[0] // 2]
    leaf_probs_2 = leaf_probs[leaf_probs.shape[0] // 2:]
    delta_1 = delta[:delta.shape[0] // 2]
    delta_2 = delta[delta.shape[0] // 2:]
    # loss
    loss += nll(leaf_probs_1, glabel_1.argmax(0))
    loss += nll(leaf_probs_2, glabel_2.argmax(0))
    for i in range(group.number_leaf()):
        mask = rlabel_1[i] >= 0
        if mask.sum() != 0:
            loss += mse(delta_1[:, i][mask].reshape(-1, 1).float(), rlabel_1[i][mask].reshape(-1, 1).float())
        mask = rlabel_2[i] >= 0
        if mask.sum() != 0:
            loss += mse(delta_2[:, i][mask].reshape(-1, 1).float(), rlabel_2[i][mask].reshape(-1, 1).float())
    loss.backward()

    if opti_flag:
        optimizer.step()
        optimizer.zero_grad()

    end = time.time()
    batch_time = end - start
    if batch_idx % args.print_freq == 0:
        print('[Training][%d/%d][%d/%d] \t Batch_time %.2f \t Batch_loss: %.4f \t lr1 : %0.5f '
              % (epoch, args.max_epoch, batch_idx, batch_num,
                 batch_time, loss.item(), optimizer.param_groups[0]['lr']))

    # evaluate result of training phase
    relative_scores = group.inference(leaf_probs_2.detach().cpu().numpy(), delta_2.detach().cpu().numpy())
    if args.benchmark == 'MTL':
        if args.usingDD:
            score = (relative_scores + label_2) * diff
        else:
            score = relative_scores + label_2
    elif args.benchmark == 'Seven':
        score = relative_scores + label_2
    else:
        raise NotImplementedError()
    pred_scores.extend(score.numpy())


def network_forward_test(base_model, regressor, pred_scores, feature_1, feature_2_list, label_2_list, diff, group,
                         args, data, target, gcn, attn_encoder, linear_bp):
    score = 0
    if not args.use_i3d_bb:
        feature_1 = linear_bp(feature_1)  # B,540,1024
        for idx, feature_2 in enumerate(feature_2_list):
            feature_2 = linear_bp(feature_2)  # B,540,1024
            feature_2_list[idx] = feature_2

    if args.use_goat:
        if args.use_formation:
            q1 = data['formation_features']  # B,540,1024
            k1 = q1
            feature_1 = attn_encoder(q1, k1, feature_1)
            feature_1 = feature_1.mean(1)  # B,1024
        elif args.use_bp:
            q1 = data['bp_features']  # B,540,768
            k1 = q1
            feature_1 = attn_encoder(q1, k1, feature_1)
            feature_1 = feature_1.mean(1)  # B,1024
        elif args.use_self:
            q1 = feature_1
            k1 = q1
            feature_1 = attn_encoder(q1, k1, feature_1)
            feature_1 = feature_1.mean(1)  # B,1024
        else:
            if args.use_cnn_features:
                boxes_features_1 = data['cnn_features']
                boxes_in_1 = data['boxes']  # B,T,N,4
                q1 = gcn(boxes_features_1, boxes_in_1)  # B,540,1024
                k1 = q1
                feature_1 = attn_encoder(q1, k1, feature_1)  # B,540,1024
                feature_1 = feature_1.mean(1)  # B,1024
            else:
                images_in_1 = data['video']  # B,T,C,H,W
                boxes_in_1 = data['boxes']  # B,T,N,4
                q1 = gcn(images_in_1, boxes_in_1)  # B,540,1024
                k1 = q1
                feature_1 = attn_encoder(q1, k1, feature_1)  # B,540,1024
                feature_1 = feature_1.mean(1)  # B,1024

    feature_1_ori = feature_1
    for tar, feature_2, label_2 in zip(target, feature_2_list, label_2_list):
        feature_1 = feature_1_ori
        # combined_feature = base_model(video_1,video_2, label = [label_2], is_train = False , theta = args.score_range)
        # TODO: theta
        label = [label_2]
        theta = args.score_range

        ######### GOAT START ##########
        if args.use_goat:
            if args.use_formation:
                q2 = tar['formation_features']  # B,540,1024
                k2 = q2
                feature_2 = attn_encoder(q2, k2, feature_2)
                feature_2 = feature_2.mean(1)  # B,1024
            elif args.use_bp:
                q2 = tar['bp_features']  # B,540,768
                k2 = q2
                feature_2 = attn_encoder(q2, k2, feature_2)
                feature_2 = feature_2.mean(1)  # B,1024
            elif args.use_self:
                q2 = feature_2  # B,540,1024
                k2 = q2
                feature_2 = attn_encoder(q2, k2, feature_2)
                feature_2 = feature_2.mean(1)  # B,1024
            else:
                if args.use_cnn_features:
                    boxes_features_2 = tar['cnn_features']
                    boxes_in_2 = tar['boxes']  # B,T,N,4
                    q2 = gcn(boxes_features_2, boxes_in_2)  # B,540,2024
                    k2 = q2
                    feature_2 = attn_encoder(q2, k2, feature_2)  # B,540,2024
                    feature_2 = feature_2.mean(1)  # B,2024
                else:
                    images_in_2 = tar['video']  # B,T,C,H,W
                    boxes_in_2 = tar['boxes']  # B,T,N,4
                    q2 = gcn(images_in_2, boxes_in_2)  # B,540,1024
                    k2 = q2
                    feature_2 = attn_encoder(q2, k2, feature_2)  # B,540,1024
                    feature_2 = feature_2.mean(1)  # B,1024
        #########  GOAT END  ##########
        else:
            total_feature = ops.cat((feature_1, feature_2), 0).mean(1)  # 2B,1024
            feature_1 = total_feature[:total_feature.shape[0] // 2]  # B,1024
            feature_2 = total_feature[total_feature.shape[0] // 2:]  # B,1024

        combined_feature = ops.cat((feature_2, feature_1, label[0] / theta), 1)  # 2 is exemplar N * 2049

        out_prob, delta = regressor(combined_feature)
        # evaluate result of training phase
        leaf_probs = out_prob[-1].reshape(combined_feature.shape[0], -1)
        relative_scores = group.inference(leaf_probs.numpy(), delta.numpy())
        if args.benchmark == 'MTL':
            if args.usingDD:
                score += (relative_scores + label_2) * diff
            else:
                score += relative_scores + label_2
        else:
            raise NotImplementedError()
    pred_scores.extend((score / len(feature_2_list)).numpy())


def save_checkpoint(base_model, regressor, optimizer, epoch, epoch_best, rho_best, L2_min, RL2_min, exp_name, args):
    ms.save_checkpoint({
        # 'base_model' : base_model.state_dict(),
        'regressor': regressor.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'epoch_best': epoch_best,
        'rho_best': rho_best,
        'L2_min': L2_min,
        'RL2_min': RL2_min,
    }, os.path.join(args.experiment_path, exp_name + '.pth'))


def save_outputs(pred_scores, true_scores, args):
    save_path_pred = os.path.join(args.experiment_path, 'pred.npy')
    save_path_true = os.path.join(args.experiment_path, 'true.npy')
    np.save(save_path_pred, pred_scores)
    np.save(save_path_true, true_scores)
