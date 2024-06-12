
# Model Architecture

BCANet is a two-stage target detection network,This network uses a region proposal network (RPN), which can share the convolution features of the whole image with the detection network, so that the calculation of region proposal is almost cost free. The whole network further combines RPN and FastRcnn into a network by sharing the convolution features.

# Environment Requirements

- Hardware（Ascend/GPU/CPU）

    - Prepare hardware environment with Ascend processor.

- Docker base image

    - [Ascend Hub](ascend.huawei.com/ascendhub/#/home)

- Install [MindSpore](https://www.mindspore.cn/install/en).

    1. If coco dataset is used. **Select dataset to coco when run script.**
        Install Cython and pycocotool, and you can also install mmcv to process data.

        ```pip
        pip install Cython

        pip install pycocotools

        pip install mmcv==0.2.14
        ```

        And change the COCO_ROOT and other settings you need in `default_config.yaml、default_config_101.yaml or default_config_152.yaml`. The directory structure is as follows:

        ```path
        .
        └─cocodataset
          ├─annotations
            ├─instance_train2017.json
            └─instance_val2017.json
          ├─val2017
          └─train2017

        ```

    2. If your own dataset is used. **Select dataset to other when run script.**
        Organize the dataset information into a TXT file, each row in the file is as follows:

        ```log
        train2017/0000001.jpg 0,259,401,459,7 35,28,324,201,2 0,30,59,80,2
        ```

        Each row is an image annotation which split by space, the first column is a relative path of image, the others are box and class information of the format [xmin,ymin,xmax,ymax,class]. We read image from an image path joined by the `IMAGE_DIR`(dataset directory) and the relative path in `ANNO_PATH`(the TXT file path), `IMAGE_DIR` and `ANNO_PATH` are setting in `default_config_50.yaml、default_config_101.yaml or default_config_152.yaml`.

# Quick Start

After installing MindSpore via the official website, you can start training and evaluation as follows:

Note:

1. the first run will generate the mindeocrd file, which will take a long time.
2. pretrained model is a resnet50 checkpoint that trained over ImageNet2012.you can train it with [resnet50](https://gitee.com/mindspore/models/tree/master/official/cv/ResNet) scripts in modelzoo, and use src/convert_checkpoint.py to get the pretrain model.
3. BACKBONE_MODEL is a checkpoint file trained with [resnet50](https://gitee.com/mindspore/models/tree/master/official/cv/ResNet) scripts in modelzoo.PRETRAINED_MODEL is a checkpoint file after convert.VALIDATION_JSON_FILE is label file. CHECKPOINT_PATH is a checkpoint file after trained.

## Run on Ascend

```shell

# convert checkpoint
python -m src.convert_checkpoint --ckpt_file=[BACKBONE_MODEL]

# standalone training
bash run_standalone_train_ascend.sh [PRETRAINED_MODEL] [BACKBONE] [COCO_ROOT] [DEVICE_ID] [MINDRECORD_DIR](optional)

# distributed training
bash run_distribute_train_ascend.sh [RANK_TABLE_FILE] [PRETRAINED_MODEL] [BACKBONE] [COCO_ROOT] [MINDRECORD_DIR](optional)

# eval
bash run_eval_ascend.sh [VALIDATION_JSON_FILE] [CHECKPOINT_PATH] [BACKBONE] [COCO_ROOT] [DEVICE_ID] [MINDRECORD_DIR](optional)

# inference (the values of IMAGE_WIDTH and IMAGE_HEIGHT must be set or use default at the same time.)
bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [ANN_FILE_PATH] [IMAGE_WIDTH](optional) [IMAGE_HEIGHT](optional) [DEVICE_ID](optional)
```

## Run on GPU

```shell

# convert checkpoint
python -m src.convert_checkpoint --ckpt_file=[BACKBONE_MODEL]

# standalone training
bash run_standalone_train_gpu.sh [PRETRAINED_MODEL] [BACKBONE] [COCO_ROOT] [DEVICE_ID] [MINDRECORD_DIR](optional)

# distributed training
bash run_distribute_train_gpu.sh [DEVICE_NUM] [PRETRAINED_MODEL] [BACKBONE] [COCO_ROOT] [MINDRECORD_DIR](optional)

# eval
bash run_eval_gpu.sh [VALIDATION_JSON_FILE] [CHECKPOINT_PATH] [BACKBONE] [COCO_ROOT] [DEVICE_ID] [MINDRECORD_DIR](optional)
```

## Run on CPU

```shell

# standalone training
bash run_standalone_train_cpu.sh [PRETRAINED_MODEL] [BACKBONE] [COCO_ROOT] [MINDRECORD_DIR](optional)

# eval
bash run_eval_cpu.sh [VALIDATION_JSON_FILE] [CHECKPOINT_PATH] [BACKBONE] [COCO_ROOT] [MINDRECORD_DIR](optional)
```

## Run in docker

1. Build docker images

```shell
# build docker
docker build -t BCANet:20.1.0 . --build-arg FROM_IMAGE_NAME=ascend-mindspore-arm:20.1.0
```

2. Create a container layer over the created image and start it

```shell
# start docker
bash scripts/docker_start.sh BCANet:20.1.0 [DATA_DIR] [MODEL_DIR]
```

3. Train

```shell
# standalone training
bash run_standalone_train_ascend.sh [PRETRAINED_MODEL] [BACKBONE] [COCO_ROOT] [DEVICE_ID] [MINDRECORD_DIR](optional)

# distributed training
bash run_distribute_train_ascend.sh [RANK_TABLE_FILE] [PRETRAINED_MODEL] [BACKBONE] [COCO_ROOT] [MINDRECORD_DIR](optional)
```

4. Eval

```shell
# eval
bash run_eval_ascend.sh [VALIDATION_JSON_FILE] [CHECKPOINT_PATH] [BACKBONE] [COCO_ROOT] [DEVICE_ID] [MINDRECORD_DIR](optional)
```

5. Inference

```shell
# inference
bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [ANN_FILE] [IMAGE_WIDTH](optional) [IMAGE_HEIGHT](optional) [DEVICE_ID](optional)
```




## Training Process

### Usage

#### on Ascend

```shell
# standalone training on ascend
bash run_standalone_train_ascend.sh [PRETRAINED_MODEL] [BACKBONE] [COCO_ROOT] [DEVICE_ID] [MINDRECORD_DIR](optional)

# distributed training on ascend
bash run_distribute_train_ascend.sh [RANK_TABLE_FILE] [PRETRAINED_MODEL] [BACKBONE] [COCO_ROOT] [MINDRECORD_DIR](optional)
```

#### on GPU

```shell
# standalone training on gpu
bash run_standalone_train_gpu.sh [PRETRAINED_MODEL] [BACKBONE] [COCO_ROOT] [DEVICE_ID] [MINDRECORD_DIR](optional)

# distributed training on gpu
bash run_distribute_train_gpu.sh [DEVICE_NUM] [PRETRAINED_MODEL] [BACKBONE] [COCO_ROOT] [MINDRECORD_DIR](optional)
```

#### on CPU

```shell
# standalone training on cpu
bash run_standalone_train_cpu.sh [PRETRAINED_MODEL] [BACKBONE] [COCO_ROOT] [MINDRECORD_DIR](optional)
```

Notes:

1. Rank_table.json which is specified by RANK_TABLE_FILE is needed when you are running a distribute task. You can generate it by using the [hccl_tools](https://gitee.com/mindspore/models/tree/master/utils/hccl_tools).
2. As for PRETRAINED_MODEL，it should be a trained ResNet50 checkpoint. If you need to load Ready-made pretrained BCANet checkpoint, you may make changes to the train.py script as follows.

```python
# Comment out the following code
#   load_path = args_opt.pre_trained
#    if load_path != "":
#        param_dict = load_checkpoint(load_path)
#        for item in list(param_dict.keys()):
#            if not item.startswith('backbone'):
#                param_dict.pop(item)
#        load_param_into_net(net, param_dict)

# Add the following codes after optimizer definition since the BCANet checkpoint includes optimizer parameters：
    lr = Tensor(dynamic_lr(config, rank_size=device_num), mstype.float32)
    opt = SGD(params=net.trainable_params(), learning_rate=lr, momentum=config.momentum,
              weight_decay=config.weight_decay, loss_scale=config.loss_scale)

    if load_path != "":
        param_dict = load_checkpoint(load_path)
        for item in list(param_dict.keys()):
            if item in ("global_step", "learning_rate") or "rcnn.reg_scores" in item or "rcnn.cls_scores" in item:
                param_dict.pop(item)
        load_param_into_net(opt, param_dict)
        load_param_into_net(net, param_dict)
```

3. The original dataset path needs to be in the default_config_50.yaml、default_config_101.yaml、default_config_152.yaml,you can select "coco_root" or "image_dir".

### Result

Training result will be stored in the example path, whose folder name begins with "train" or "train_parallel". You can find checkpoint file together with result like the following in loss_rankid.log.

```log
# distribute training result(8p)
739 s | epoch: 1 step: 1335 total_loss: 0.18823  lr: 0.004989
1197 s | epoch: 2 step: 1335 total_loss: 0.10421  lr: 0.004927
1565 s | epoch: 3 step: 1335 total_loss: 0.10069  lr: 0.004810
1896 s | epoch: 4 step: 1335 total_loss: 0.06817  lr: 0.004641
2267 s | epoch: 5 step: 1335 total_loss: 0.09090  lr: 0.004425
2636 s | epoch: 6 step: 1335 total_loss: 0.07419  lr: 0.004166
2994 s | epoch: 7 step: 1335 total_loss: 0.14894  lr: 0.003870
3368 s | epoch: 8 step: 1335 total_loss: 0.07765  lr: 0.003542
3713 s | epoch: 9 step: 1335 total_loss: 0.09182  lr: 0.003192
4099 s | epoch: 10 step: 1335 total_loss: 0.22383  lr: 0.002826
4473 s | epoch: 11 step: 1335 total_loss: 0.03834  lr: 0.002453
4842 s | epoch: 12 step: 1335 total_loss: 0.08823  lr: 0.002081
5232 s | epoch: 13 step: 1335 total_loss: 0.05278  lr: 0.001719
5605 s | epoch: 14 step: 1335 total_loss: 0.04989  lr: 0.001373
5997 s | epoch: 15 step: 1335 total_loss: 0.03753  lr: 0.001053
6370 s | epoch: 16 step: 1335 total_loss: 0.13075  lr: 0.000766
6774 s | epoch: 17 step: 1335 total_loss: 0.04877  lr: 0.000517
7150 s | epoch: 18 step: 1335 total_loss: 0.18740  lr: 0.000312
7539 s | epoch: 19 step: 1335 total_loss: 0.07818  lr: 0.000156
7905 s | epoch: 20 step: 1335 total_loss: 0.05880  lr: 0.000053
```

## Evaluation Process

### Usage

#### on Ascend

```shell
# eval on ascend
bash run_eval_ascend.sh [VALIDATION_JSON_FILE] [CHECKPOINT_PATH] [BACKBONE] [COCO_ROOT] [DEVICE_ID] [MINDRECORD_DIR](optional)
```

#### on GPU

```shell
# eval on GPU
bash run_eval_gpu.sh [VALIDATION_JSON_FILE] [CHECKPOINT_PATH] [BACKBONE] [COCO_ROOT] [DEVICE_ID] [MINDRECORD_DIR](optional)
```

#### on CPU

```shell
# eval on CPU
bash run_eval_cpu.sh [VALIDATION_JSON_FILE] [CHECKPOINT_PATH] [BACKBONE] [COCO_ROOT] [MINDRECORD_DIR](optional)
```

> checkpoint can be produced in training process.
>
> Images size in dataset should be equal to the annotation size in VALIDATION_JSON_FILE, otherwise the evaluation result cannot be displayed properly.

### Result

Eval result will be stored in the example path, whose folder name is "eval". Under this, you can find result like the following in log.

```log
Result on garbage_dump dataset
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.394
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.583
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.388
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.528
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.375
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.587
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.560
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.560
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.633
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.570

```

