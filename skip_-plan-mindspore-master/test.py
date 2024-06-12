import mindspore as ms
import mindspore.nn as nn
logits = ms.Tensor([[0.8, 1.4], [0.5, 0.9], [1.2, 0.9]], ms.float32)
labels = ms.Tensor([[1], [1], [0]], ms.int32)
focalloss = nn.FocalLoss(weight=ms.Tensor([1, 2]), gamma=2.0, reduction='mean')
output = focalloss(logits, labels)
print(output)
print(labels.shape)
print(logits.shape)
print(output.(shape))
print()
print