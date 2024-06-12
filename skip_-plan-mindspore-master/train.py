import sys

sys.path.append("..")  # 相对路径或绝对路径
import time
from collections import OrderedDict
import json
import pickle
import os
import shutil
import argparse

from callbacks import AverageMeter, Logger, set_save_path
from datasets.CrossTask_dataloader_ms import *

import mindspore.dataset as ds
import mindspore
import mindspore.ops.operations as P
from mindspore import nn
import mindspore.ops as ops
import mindspore.context as context

context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")  # 设置计算设备为GPU
# context.set_auto_tune(True)# 启用MindSpore的自动调优功能
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
from mindspore.communication.management import init, get_rank
from mindspore.context import ParallelMode


def set_device(args):
    """Set device and ParallelMode(if device_num > 1)"""
    rank = 0
    # set context and device
    device_target = args.device_target
    device_num = int(os.environ.get("DEVICE_NUM", 1))
    print("device_num is {}".format(device_num))

    if device_target == "GPU":
        if device_num > 1:
            init(backend_name="nccl")
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(
                device_num=device_num,
                parallel_mode=ParallelMode.DATA_PARALLEL,
                gradients_mean=True,
            )
            rank = get_rank()
        else:
            context.set_context(device_id=int(args.gpu))
    else:
        raise ValueError("Unsupported platform.")

    return rank


mode = {0: context.GRAPH_MODE, 1: context.PYNATIVE_MODE}

data_path = "/home/jingxuan/skipplan/datasets/CrossTask_assets"
parser = argparse.ArgumentParser()


parser.add_argument(
    "--data_path", type=str, default=data_path, help="default data path"
)

parser.add_argument(
    "--primary_path",
    type=str,
    default=os.path.join(data_path, "crosstask_release/tasks_primary.txt"),
    help="list of primary tasks",
)
parser.add_argument(
    "--related_path",
    type=str,
    default=os.path.join(data_path, "crosstask_release/tasks_related.txt"),
    help="list of related tasks",
)
parser.add_argument(
    "--annotation_path",
    type=str,
    default=os.path.join(data_path, "crosstask_release/annotations"),
    help="path to annotations",
)
parser.add_argument(
    "--video_csv_path",
    type=str,
    default=os.path.join(data_path, "crosstask_release/videos.csv"),
    help="path to video csv",
)
parser.add_argument(
    "--val_csv_path",
    type=str,
    default=os.path.join(data_path, "crosstask_release/videos_val.csv"),
    help="path to validation csv",
)
parser.add_argument(
    "--features_path",
    type=str,
    default=os.path.join(data_path, "crosstask_features"),
    help="path to features",
)
parser.add_argument(
    "--constraints_path",
    type=str,
    default=os.path.join(data_path, "crosstask_constraints"),
    help="path to constraints",
)
parser.add_argument(
    "--n_train", type=int, default=30, help="videos per task for training"
)

parser.add_argument(
    "--use_related",
    type=int,
    default=0,
    help="1 for using related tasks during training, 0 for using primary tasks only",
)
parser.add_argument(
    "--share",
    type=str,
    default="words",
    help="Level of sharing between tasks",
)
parser.add_argument(
    "--dataset",
    type=str,
    default="crosstask",
    help="Used dataset name for logging",
)
parser.add_argument(
    "--dataloader-type",
    type=str,
    default="ddn",
    help="The type of dataset processing loader: either ddn or plate",
)
parser.add_argument(
    "--label-type",
    type=str,
    default="ddn",
    help="The type of dataset processing loader: either ddn or plate",
)

parser.add_argument("--epochs", default=200, type=int, help="number of epochs")
parser.add_argument("--batch_size", default=32, type=int, help="batch size")
parser.add_argument("--max_traj_len", default=3, type=int, help="action number")
parser.add_argument("--gpu", default="3", type=str)
parser.add_argument("--dataset_root", default="./crosstask/")
parser.add_argument("--frameduration", default=3, type=int)
parser.add_argument("--dataset_mode", default="multiple")
parser.add_argument(
    "--ckpt",
    default="/home/jingxuan/skipplan/ckpt",
    help="folder to output checkpoints",
)
# parser.add_argument('--dataset_mode', default='multiple')
parser.add_argument("--num_workers", default=8, type=int)
parser.add_argument("--lr", default=0.02, type=float, help="initial learning rate")
parser.add_argument("--momentum", default=0.9, type=float, help="momentum")
parser.add_argument("--weight_decay", default=0.0001, type=float, help="weight decay")
parser.add_argument("--start_epoch", default=None, type=int)
parser.add_argument(
    "--lr_steps",
    default=[100, 150, 200, 250, 300, 350, 400, 450, 500, 700, 900],
    type=float,
)
parser.add_argument("--clip_gradient", default=5, type=float)
parser.add_argument(
    "--print_freq", "-p", default=100, type=int, help="print frequency (default: 20)"
)
parser.add_argument(
    "--log_freq",
    "-l",
    default=10,
    type=int,
    help="frequency to write in tensorboard (default: 10)",
)
parser.add_argument("--memory_size", default=128)
parser.add_argument(
    "--N", default=1, type=int, help="Number of layers in the temporal decoder"
)
parser.add_argument(
    "--H", default=16, type=int, help="Number of heads in the temporal decoder"
)
parser.add_argument("--d_model", default=1024, type=int)
parser.add_argument("--decoder_dropout", default=0, type=float)
parser.add_argument("--resume", default=None)
parser.add_argument("--seed", default=99999999, type=int)
parser.add_argument("--exist_datasplit", default=False, type=bool)
parser.add_argument("--dim_feedforward", default=1024, type=int)
parser.add_argument("--mlp_mid", default=512, type=int)
parser.add_argument("--gamma", default=1.5, type=float)
parser.add_argument("--smallmid_ratio", default=3, type=int)
parser.add_argument("--graph_mode", default=0, type=int)
parser.add_argument("--device_target", default="GPU", type=str)

# options
args = parser.parse_args()
args.query_length = args.max_traj_len + 1
args.memory_length = args.max_traj_len + 1
print(args)

best_loss = 1000000
best_acc = -np.inf
best_success_rate = -np.inf
best_miou = -np.inf

context.set_context(
    mode=mode[args.graph_mode], device_target=args.device_target
)
context.set_context(enable_graph_kernel=False)

rank = set_device(args)  # 分布式训练，多卡并行设置，返回当前卡号
ms.set_seed(args.seed + rank)

device_index = ms.get_context("device_id")
print("当前正在运行代码的设备索引:", device_index)
########################################
# Start Loading/Processing the dataset #
########################################

task_vids = get_vids(args.video_csv_path)
val_vids = get_vids(args.val_csv_path)
task_vids = {
    task: [vid for vid in vids if task not in val_vids or vid not in val_vids[task]]
    for task, vids in task_vids.items()
}
primary_info = read_task_info(args.primary_path)
test_tasks = set(primary_info["steps"].keys())
if args.use_related:
    related_info = read_task_info(args.related_path)
    task_steps = {**primary_info["steps"], **related_info["steps"]}
    n_steps = {**primary_info["n_steps"], **related_info["n_steps"]}
else:
    task_steps = primary_info["steps"]
    n_steps = primary_info["n_steps"]
all_tasks = set(n_steps.keys())
task_vids = {task: vids for task, vids in task_vids.items() if task in all_tasks}
val_vids = {task: vids for task, vids in val_vids.items() if task in all_tasks}

with open(os.path.join(args.data_path, "crosstask_release/cls_step.json"), "r") as f:
    step_cls = json.load(f)
with open(
    os.path.join(args.data_path, "crosstask_release/activity_step.json"), "r"
) as f:
    act_cls = json.load(f)

##################################
# If using existing data-split   #
##################################
if args.exist_datasplit:
    with open(
        "./checkpoints/CrossTask_t{}_datasplit_pre.pth".format(args.max_traj_len), "rb"
    ) as f:
        datasplit = pickle.load(f)
    trainset = CrossTaskDataset(
        task_vids,
        n_steps,
        args.features_path,
        args.annotation_path,
        step_cls,
        pred_h=args.max_traj_len,
        act_json=act_cls,
    )
    testset = CrossTaskDataset(
        task_vids,
        n_steps,
        args.features_path,
        args.annotation_path,
        step_cls,
        pred_h=args.max_traj_len,
        act_json=act_cls,
        train=False,
    )
    trainset.plan_vids = datasplit["train"]
    testset.plan_vids = datasplit["test"]

else:
    """Random Split dataset by video"""
    train_vids, test_vids = random_split(
        task_vids, test_tasks, args.n_train, seed=args.seed
    )

    trainset = CrossTaskDataset(
        train_vids,
        n_steps,
        args.features_path,
        args.annotation_path,
        step_cls,
        pred_h=args.max_traj_len,
        act_json=act_cls,
    )

    # Run random_split for eval/test sub-set
    # trainset.random_split()
    testset = CrossTaskDataset(
        test_vids,
        n_steps,
        args.features_path,
        args.annotation_path,
        step_cls,
        pred_h=args.max_traj_len,
        act_json=act_cls,
    )

    # Show stats of train/test dataset
print("Training dataset has {} samples".format(len(trainset)))
print("Testing dataset has {} samples".format(len(testset)))

#######################
# Run data whitening  #
#######################

mean_lang = 0.038948704
mean_vis = 0.000133333
var_lang = 33.063942
var_vis = 0.00021489676

trainset.mean_lan = mean_lang
trainset.mean_vis = mean_vis
trainset.var_lan = var_lang
trainset.var_vis = var_vis
testset.mean_lan = mean_lang
testset.mean_vis = mean_vis
testset.var_lan = var_lang
testset.var_vis = var_vis

#######################
# Init the DataLoader #
######################

train_loader = ds.GeneratorDataset(
    source=trainset,
    column_names=["vid", "task", "X", "C", "W", "T"],
    shuffle=True,
    num_parallel_workers=args.num_workers,
)
train_loader = train_loader.batch(batch_size=args.batch_size, drop_remainder=True)


test_loader = ds.GeneratorDataset(
    testset,
    ["vid", "task", "X", "C", "W", "T"],
    shuffle=False,
    num_parallel_workers=args.num_workers,
)
test_loader = test_loader.batch(batch_size=args.batch_size, drop_remainder=True)


"""Get all reference from test-set, for KL-Divgence, NLL, MC-Prec and MC-Rec"""
reference = [x[2] for x in testset.plan_vids]
all_ref = np.array(reference)

##################################
# Saving the data split to local #
##################################
if not args.exist_datasplit:
    datasplit = {}

    datasplit["train"] = trainset.plan_vids
    datasplit["test"] = testset.plan_vids

    with open("CrossTask_t{}_datasplit.pth".format(args.max_traj_len), "wb") as f:
        pickle.dump(datasplit, f)


def main():
    global best_loss, best_acc, best_success_rate, best_miou

    # create model

    from model.model_baseline_cont import Model

    model = Model(args)

    # optionally resume from a checkpoint
    if args.resume:  # path to latest checkpoint
        assert os.path.isfile(args.resume), "No checkpoint found at '{}'".format(
            args.resume
        )
        print("=> loading checkpoint '{}'".format(args.resume))

        checkpoint = mindspore.load_checkpoint(args.resume, net=model)
        mindspore.load_param_into_net(model, checkpoint)

    if args.start_epoch is None:
        args.start_epoch = 0

    num_param = sum(p.numel() for p in model.trainable_params())
    print("Total number of parameters: ", num_param)
    polynomial_decay_lr = nn.PolynomialDecayLR(args.lr, 0.002, 4, 0.5)
    optimizer = nn.SGD(
        params=model.trainable_params(),
        learning_rate=polynomial_decay_lr,
        momentum=args.momentum,
    )

    criterion = nn.FocalLoss(gamma=args.gamma)

    # training
    time_pre = time.strftime("%Y%m%d%H%M%S", time.localtime())
    logname = (
        "expcont_"
        + time_pre
        + "_"
        + str(args.dataset_mode)
        + "_"
        + str(args.max_traj_len)
    )
    args.logname = logname
    tb_logdir = os.path.join("./logs", logname)
    if not (os.path.exists(tb_logdir)):
        os.makedirs(tb_logdir)
    tb_logger = Logger(tb_logdir)

    log, writer = set_save_path(tb_logdir)

    for epoch in range(args.start_epoch, args.epochs):
        # adjust_learning_rate(args, optimizer, epoch, args.lr_steps)
        epoch_starttime = time.time()
        model.set_train()
        train_loss, train_state_loss, train_acc, train_success_rate, train_miou = train(
            args, train_loader, model, optimizer, epoch, criterion, tb_logger
        )
        model.set_train(False)
        loss, acc, success_rate, miou = validate(
            args, test_loader, model, criterion, epoch, tb_logger
        )

        epoch_endtime = time.time()

        oneepoch_time = epoch_endtime - epoch_starttime

        print("one epoch time:", oneepoch_time)
        print("t/T=", oneepoch_time * epoch, "/", oneepoch_time * args.epochs)
        print("SSSSSSSSSSSSSSETTING", args)

        is_best_sr = success_rate > best_success_rate
        if is_best_sr:
            best_loss = loss
            best_acc = acc
            best_success_rate = success_rate
            best_miou = miou
        print(
            "Epoch {}: Best evaluation - "
            "accuracy: {:.2f}, success rate: {:.2f}, miou: {:.2f}".format(
                epoch, best_acc, best_success_rate, best_miou
            )
        )
        if not os.path.exists(args.ckpt):
            os.makedirs(args.ckpt)
        save_checkpoint(
            model,
            {
                "epoch": epoch + 1,
                "best_loss": best_loss,
                "best_success_rate": best_success_rate,
            },
            is_best_sr,  # is_best,
            os.path.join(tb_logdir, "{}".format(logname)),
        )

        log_info = ["epoch {}/{}".format(epoch, args.epochs)]
        log_info.append("train: train_loss={:.4f}".format(train_loss))
        log_info.append("train_acc={:.4f}".format(train_acc))
        log_info.append("train_success_rate={:.4f}".format(train_success_rate))
        log_info.append("train_MIoU={:.4f}".format(train_miou))

        log_info.append("val: val_loss={:.4f}".format(loss))
        log_info.append("val_acc={:.4f}".format(acc))
        log_info.append("val_success_rate={:.4f}".format(success_rate))
        log_info.append("val_MIoU={:.4f}".format(miou))

        log_info.append("best: best_loss={:.4f}".format(best_loss))
        log_info.append("best_acc={:.4f}".format(best_acc))
        log_info.append("best_success_rate={:.4f}".format(best_success_rate))
        log_info.append("best_MIoU={:.4f}".format(best_miou))
        # writer.flush()
        log(", ".join(log_info))

        if epoch == 1:
            tb_logger.log_info(args)


def train(args, trainset, model, optimizer, epoch, criterion, tb_logger):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    state_losses = AverageMeter()
    acc_meter = AverageMeter()
    success_rate_meter = AverageMeter()
    miou_meter = AverageMeter()

    model.set_train()
    end = time.time()

    reshape_op = P.Reshape()
    expand_dims_op = P.ExpandDims()

    # Define forward function
    def forward_fn(frames, lowlevel_labels):
        output = model(frames)  # out (32, 3, 106)
        output_reshaped = reshape_op(output, (-1, output.shape[-1]))
        lowlevel_labels_reshaped = reshape_op(lowlevel_labels, (-1,))
        lowlevel_labels_reshaped = expand_dims_op(lowlevel_labels_reshaped, 1)
        loss = criterion(output_reshaped, lowlevel_labels_reshaped.long())
        return loss, output_reshaped

    # Get gradient function
    grad_fn = ms.value_and_grad(
        forward_fn, None, weights=optimizer.parameters, has_aux=True
    )

    # Define function of one-step training
    def train_step(data, label):
        (loss, logits), grads = grad_fn(data, label)

        if args.clip_gradient is not None:
            grads = ops.clip_by_global_norm(grads, args.clip_gradient)
        optimizer(grads)
        return loss, logits

    for i, (_, _, frames, _, lowlevel_labels, _) in enumerate(trainset):
        model.set_train()
        data_time.update(time.time() - end)
        loss, output_reshaped = train_step(frames, lowlevel_labels)

        loss = float(loss.asnumpy())

        lowlevel_labels_reshaped = reshape_op(lowlevel_labels, (-1,))
        lowlevel_labels_reshaped = expand_dims_op(lowlevel_labels_reshaped, 1)
        acc, success_rate, _ = accuracy(
            output_reshaped, lowlevel_labels_reshaped, args.max_traj_len
        )

        out = reshape_op(output_reshaped, (32, 3, 106))
        _, output_r = out.topk(1, 2, True, True)
        gt = ms.ops.Squeeze(axis=-1)(output_r).asnumpy().astype("int")
        rst = lowlevel_labels.asnumpy().astype("int")
        miou = acc_iou(rst, gt, False)
        miou = miou.mean()

        losses.update(loss, frames.shape[0])
        acc_meter.update(acc.item().asnumpy(), frames.shape[0])
        success_rate_meter.update(success_rate.item().asnumpy(), frames.shape[0])
        miou_meter.update(miou, frames.shape[0] // args.max_traj_len)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(
                "Epoch: [{0}][{1}/{2}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "State Loss {state_loss.val:.4f} ({state_loss.avg:.4f})\t"
                "Train Acc {acc_meter.val:.2f} ({acc_meter.avg:.2f})\t"
                "Train Success Rate {success_rate_meter.val:.2f} ({success_rate_meter.avg:.2f})\t"
                "Train_MIoU {miou_meter.val:.2f} ({miou_meter.avg:.2f})\t".format(
                    epoch,
                    i,
                    len(trainset),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    state_loss=state_losses,
                    acc_meter=acc_meter,
                    success_rate_meter=success_rate_meter,
                    miou_meter=miou_meter,
                )
            )
        # log training data into tensorboard
        if tb_logger is not None and i % args.log_freq == 0:
            logs = OrderedDict()
            logs["Train/IterLoss"] = losses.val
            logs["Train/Acc"] = acc_meter.val
            logs["Train/Success_Rate"] = success_rate_meter.val
            logs["Train/MIoU"] = miou_meter.val
            # how many iterations we have trained
            iter_count = epoch * len(train_loader) + i
            for key, value in logs.items():
                tb_logger.log_scalar(value, key, iter_count)

            # tb_logger.log_info = ['epoch {}/{}'.format(epoch, args.epoch)]
            tb_logger.flush()
    return (
        losses.avg,
        state_losses.avg,
        acc_meter.avg,
        success_rate_meter.avg,
        miou_meter.avg,
    )


def validate(args, testset, model, criterion, epoch, tb_logger):
    batch_time = AverageMeter()
    losses = AverageMeter()
    state_losses = AverageMeter()
    acc_meter = AverageMeter()
    success_rate_meter = AverageMeter()
    miou_meter = AverageMeter()

    end = time.time()

    all_rst = []
    all_gt = []
    reshape_op = P.Reshape()
    expand_dims_op = P.ExpandDims()
    model.set_train(False)

    for i, (_, _, frames, _, lowlevel_labels, _) in enumerate(testset):
        output = model(frames)
        output_reshaped = reshape_op(output, (-1, output.shape[-1]))
        lowlevel_labels_reshaped = reshape_op(lowlevel_labels, (-1,))
        lowlevel_labels_reshaped = expand_dims_op(lowlevel_labels_reshaped, 1)

        loss = criterion(output_reshaped, lowlevel_labels_reshaped.long())
        acc, success_rate, _ = accuracy(
            output_reshaped, lowlevel_labels_reshaped, args.max_traj_len
        )
        _, output_r = output.topk(1, 2, True, True)
        gt = ms.ops.Squeeze(axis=-1)(output_r).asnumpy().astype("int")
        rst = lowlevel_labels.asnumpy().astype("int")
        miou = acc_iou(rst, gt, False)
        miou = miou.mean()

        all_rst.append(rst)
        all_gt.append(gt)

        losses.update(loss.asnumpy(), frames.shape[0])
        acc_meter.update(acc.item().asnumpy(), frames.shape[0])
        success_rate_meter.update(success_rate.item().asnumpy(), frames.shape[0])
        miou_meter.update(miou, frames.shape[0] // args.max_traj_len)

        batch_time.update(time.time() - end)

        if i % args.print_freq == 0 or i + 1 == len(testset):
            print(
                "Test: [{0}/{1}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Val Acc {acc_meter.val:.2f} ({acc_meter.avg:.2f})\t"
                "Val Success Rate {success_rate_meter.val:.2f} ({success_rate_meter.avg:.2f})\t"
                "Val MIoU {miou_meter.val:.1f} ({miou_meter.avg:.2f})\t".format(
                    i,
                    len(testset),
                    batch_time=batch_time,
                    loss=losses,
                    acc_meter=acc_meter,
                    success_rate_meter=success_rate_meter,
                    miou_meter=miou_meter,
                )
            )

    if epoch is not None and tb_logger is not None:
        logs = OrderedDict()
        logs["Val/EpochLoss"] = losses.avg
        logs["Val/Acc"] = acc_meter.val
        logs["Val/Success_Rate"] = success_rate_meter.val
        logs["Val/MIoU"] = miou_meter.val

        # how many iterations we have trained
        for key, value in logs.items():
            tb_logger.log_scalar(value, key, epoch + 1)

        tb_logger.flush()

    return losses.avg, acc_meter.avg, success_rate_meter.avg, miou_meter.avg


def save_checkpoint(state, dic, is_best, filename):
    print("save checkpoint to {}".format(filename))
    mindspore.save_checkpoint(
        save_obj=state, append_dict=dic, ckpt_file_name=filename + "_latest.ckpt"
    )

    if is_best:
        best_filename = filename + "_best.ckpt"
        shutil.copyfile(filename + "_latest.ckpt", best_filename)


def adjust_learning_rate(args, optimizer, epoch, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10"""
    decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
    lr = args.lr * decay
    for param_group in optimizer._parameters:
        param_group["lr"] = lr


def mean_category_acc(pred, gt):
    """required format
    Action space is a single integer
    pred: List [batch * seq]
    gt  : List [batch * seq]
    """

    from sklearn.metrics import precision_score

    rst = precision_score(gt, pred, average="macro")
    return rst


def accuracy(output, target, max_traj_len=0):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    # Token Accuracy
    batch_size = target.shape[0]
    _, pred = output.topk(1, 1, True, True)

    pred = pred.squeeze()
    target = target.view(-1)

    correct_1 = pred == target
    expand_dims_op = P.ExpandDims()
    correct_1 = expand_dims_op(correct_1, 1)

    # Instruction Accuracy
    instruction_correct = correct_1.all(axis=1)
    # print('correct1',correct_1.shape) # correct1 (96, 1)
    # print('instruction_correct', instruction_correct.sum()) # instruction_correct (96,)
    instruction_accuracy = (
        instruction_correct.sum() * 100.0 / instruction_correct.shape[0]
    )

    # Success Rate
    trajectory_success = instruction_correct.view(-1, max_traj_len).all(axis=1)
    trajectory_success_rate = (
        trajectory_success.sum() * 100.0 / trajectory_success.shape[0]
    )

    # MIoU
    pred_inst = pred
    pred_inst_set = set()
    target_inst = target.view(correct_1.shape[0], -1)
    target_inst_set = set()
    pred_inst = pred_inst.reshape(pred_inst.shape[0], 1)
    # print('pred_inst',pred_inst.shape)  (96, 1)

    for i in range(pred_inst.shape[0]):
        pred_inst_set.add(tuple(pred_inst[i].asnumpy().tolist()))
        target_inst_set.add(tuple(target_inst[i].asnumpy().tolist()))

    MIoU = (
        100.0
        * len(pred_inst_set.intersection(target_inst_set))
        / len(pred_inst_set.union(target_inst_set))
    )

    return instruction_accuracy, trajectory_success_rate, MIoU


def acc_iou(pred, gt, aggregate=True):
    """required format
    Action space is a single integer
    pred: Numpy [batch, seq]
    gt  : Numpy [batch, seq]
    """

    epsn = 1e-6

    if aggregate:
        intersection = (pred & gt).sum((0, 1))
        union = (pred | gt).sum((0, 1))
    else:
        intersection = (pred & gt).sum((1))
        union = (pred | gt).sum((1))

    return 100 * ((intersection + epsn) / (union + epsn))


# class WeightDecay(nn.Cell):
#     def __init__(self, initial_lr=args.lr, decay_rate=0.1, decay_steps=args.lr_steps):
#         super().__init__()
#         self.initial_lr = initial_lr
#         self.decay_rate = decay_rate
#         self.decay_steps = decay_steps

#     def construct(self, global_step):
#         decay = self.decay_rate ** (sum(global_step >= np.array(self.decay_steps)))
#         lr = self.initial_lr * decay
#         return lr


if __name__ == "__main__":
    main()
