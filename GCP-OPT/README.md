# Towards A Deeper Understanding of Global Covariance Pooling in Deep Learning: An Optimization Perspective

This is an code implementation base on Mindspore1.9 of TPAMI2023 paper (Towards A Deeper Understanding of Global Covariance Pooling in Deep Learning: An Optimization Perspective), created by Qilong Wang and Zhaolin Zhang.

## Introduction
Global covariance pooling (GCP) as an effective alternative to global average pooling has shown good capacity to improve deep convolutional neural networks (CNNs) in a variety of vision tasks. Although promising performance, it is still an open problem on how GCP (especially its post-normalization) works in deep learning. In this paper, we make the effort towards understanding the effect of GCP on deep learning from an optimization perspective. Specifically, we first analyze behavior of GCP with matrix power normalization on optimization loss and gradient computation of deep architectures. Our findings show that GCP can improve Lipschitzness of optimization loss and achieve flatter local minima, while improving gradient predictiveness and functioning as a special pre-conditioner on gradients. Then, we explore the effect of post-normalization on GCP from the model optimization perspective, which encourages us to propose a simple yet effective normalization, namely DropCov. Based on above findings, we point out several merits of deep GCP that have not been recognized previously or fully explored, including faster convergence, stronger model robustness and better generalization across tasks. Extensive experimental results using both CNNs and vision transformers on diversified vision tasks provide strong support to our findings while verifying the effectiveness of our method.

## Citation

```
@ARTICLE{10269023,
  author={Wang, Qilong and Zhang, Zhaolin and Gao, Mingze and Xie, Jiangtao and Zhu, Pengfei and Li, Peihua and Zuo, Wangmeng and Hu, Qinghua},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Towards a Deeper Understanding of Global Covariance Pooling in Deep Learning: An Optimization Perspective}, 
  year={2023},
  volume={45},
  number={12},
  pages={15802-15819},
  doi={10.1109/TPAMI.2023.3321392}}
```

## Usage

### Environments
●OS：18.04  
●CUDA：11.6  
●Toolkit：mindspore1.9  
●GPU:GTX 3090 


### Install
●First, Install the driver of NVIDIA  
●Then, Install the driver of CUDA  
●Last, Install cudnn

create virtual enviroment mindspore
conda create -n mindspore python=3.7.5 -y
conda activate mindspore
CUDA 10.1 
```bash
conda install mindspore-gpu cudatoolkit=10.1 -c mindspore -c conda-forge
```
CUDA 11.1 
```bash
conda install mindspore-gpu cudatoolkit=11.1 -c mindspore -c conda-forge
```
validataion 
```bash
python -c "import mindspore;mindspore.run_check()"
```

### Data preparation
Download and extract ImageNet train and val images from http://image-net.org/. (https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder), and the training and validation data is expected to be in the `train/` folder and `val/` folder respectively:


```
/path/to/imagenet/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  val/
    class1/
      img3.jpeg
    class/2
      img4.jpeg
```
### Evaluation
To evaluate a pre-trained model on ImageNet val with GPUs run:

```
CUDA_VISIBLE_DEVICES={device_ids}  python eval.py --data_path={IMAGENET_PATH} --checkpoint_file_path={CHECKPOINT_PATH} --device_target="GPU" --config_path={CONFIG_FILE} &> log &
```

### Training

#### Train with ResNet

You can run the `main.py` to train as follow:

```
mpirun --allow-run-as-root -n {RANK_SIZE} --output-filename log_output --merge-stderr-to-stdout python train.py  --config_path={CONFIG_FILE} --run_distribute=True --device_num={DEVICE_NUM} --device_target="GPU" --data_path={IMAGENET_PATH}  --output_path './output' &> log &
```
For example:

```bash
mpirun --allow-run-as-root -n 4 --output-filename log_output --merge-stderr-to-stdout python train.py  --config_path="./config/resnet50_imagenet2012_config.yaml" --run_distribute=True --device_num=4 --device_target="GPU" --data_path=./imagenet --output_path './output' &> log &
```

## Main Results
|Method           | Acc@1(%) | #Params.(M) | FLOPs(G) | Checkpoint                                                          |
| ------------------ | ----- | ------- | ----- | ------------------------------------------------------------ |
| ResNet-18   |  70.53 |  11.7   |   1.81  |               |
| ResNet-18+Fast-MPN(Ours)   | 74.64  |   19.6  |  3.11   |[Download](https://drive.google.com/file/d/1kpBpbYpWQfSzSav7v8Ms7dn1SaUlOlMk/view?usp=drive_link)|
| ResNet-18+Drop-COV(Ours)   | 73.8  |   19.6  |  3.11   |[Download](https://drive.google.com/file/d/1zVDDmmQWQ-CDDoxjaolkcjI3MACE-rxx/view?usp=drive_link)|
| ResNet-34   |  73.68 |  21.8   |   3.66  |               |
| ResNet-34+Fast-MPN(Ours)   |  76.27 |   29.7  |  5.56   |[Download](https://drive.google.com/file/d/1T0YCzz-A-V2GI1tjihCHIjug93nFSQ91/view?usp=drive_link)|
| ResNet-34+Drop-COV(Ours)   | 76.13  |   29.7  |  5.56   |[Download](https://drive.google.com/file/d/1-gvogrLlRSnpzigvevLPV1GKAHF0vr2K/view?usp=drive_link)|
| ResNet-50   |  76.07 |  25.6   |   3.86  |               |
| ResNet-50+Fast-MPN(Ours)   | 77.71  |   32.3  |  6.19   |[Download](https://drive.google.com/file/d/1aF-By_pMhbCu82Gl73sFyz4qlNS25ayJ/view?usp=drive_link)|
| ResNet-50+Drop-COV(Ours)   | 77.77  |   32.0  |  6.19   |[Download](https://drive.google.com/file/d/1PBy8evHi-xiJHiTWgqrUs8jTH58hJM2n/view?usp=share_link)|


## Contact
If you have any questions or suggestions, please feel free to contact us: qlwang@tju.edu.cn; 914767787@qq.com.
