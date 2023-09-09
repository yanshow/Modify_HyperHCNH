import torch
import torch.nn as nn
# 假设有一个模型输出和对应的标签
model_output = torch.tensor([[0.2, 0.8, 0.3], [0.7, 0.1, 0.9]])
target_labels = torch.tensor([1, 2])
loss_function = nn.CrossEntropyLoss()
loss = loss_function(model_output, target_labels)
print(loss)
