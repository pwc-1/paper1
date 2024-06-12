import mindspore_hub as mshub
from mindspore import context
import mindspore.nn as nn
from mindspore import Tensor
import numpy as np
from models2.model_MS import TimeSformer


context.set_context(mode=context.GRAPH_MODE,
                    device_target="GPU",
                    device_id=0)


model_path = "mindspore/1.9/resnet18_cifar10"
Resnet18 = mshub.load(model_path)
Resnet18.set_train(False)


class GenerateModel(nn.Cell):
    def __init__(self):
        super(GenerateModel, self).__init__()
        # self.backbone = resnet18()
        self.backbone = Resnet18
        self.STTransformer = TimeSformer(num_frames=16,
                                         img_size=7,
                                         patch_size=1,
                                         attention_type='divided_space_time',
                                         use_learnable_pos_emb=True,
                                         return_cls_token=True)
        self.fc = nn.Dense(512, 7)

    def construct(self, x):
        x = self.backbone(x)
        x = self.STTransformer(x)
        x = self.fc(x)
        return x



if __name__ == '__main__':
    # input = torch.rand(2, 16, 3, 112, 112)
    input_data = np.random.rand(2, 16, 3, 112, 112).astype(np.float32)
    input = Tensor(input_data)
    
    model = GenerateModel()
    print(model(input).shape)