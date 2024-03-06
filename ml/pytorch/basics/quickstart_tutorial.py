# From https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from dotenv import load_dotenv

load_dotenv()

######################################################################
# PyTorch提供了领域特定的库, 例如:
# `TorchText <https://pytorch.org/text/stable/index.html>`_,
# `TorchVision <https://pytorch.org/vision/stable/index.html>`_,
# `TorchAudio <https://pytorch.org/audio/stable/index.html>`_,
# 这些库包含了各种数据集。在本教程中，我们将用到一个TorchVision数据集。
#
# ``torchvision.datasets`` 模块包含了许多现实世界的视觉数据集，
# 如CIFAR、COCO（`完整列表请参考这里 <https://pytorch.org/vision/stable/datasets.html>`_）。
# 在本教程中，我们使用FashionMNIST数据集。每个TorchVision ``Dataset`` 包括两个参数：``transform`` 和 ``target_transform``，
# 分别用于修改样本和标签。

# 第一步：下载封装好的 Dataset, 如果已经下载就不会重新下载
print("第一步：下载封装好的 Dataset, 如果已经下载就不会重新下载")
# 用统一的目录存放数据集，方便管理
DATASETS_ROOT = os.getenv('DATASETS_ROOT')
# 从开放数据集中下载训练数据。
training_data = datasets.FashionMNIST(
    root=DATASETS_ROOT,
    train=True,
    download=True,
    transform=ToTensor(),
)
# 从开放数据集中下载测试数据。
test_data = datasets.FashionMNIST(
    root=DATASETS_ROOT,
    train=False,
    download=True,
    transform=ToTensor(),
)

# 第二步：加载数据集：使用 DataLoader 生成一个迭代器
print("第二步：加载数据集：使用 DataLoader 生成一个迭代器")
# 我们将``Dataset``作为参数传递给``DataLoader``。它将我们的数据集包装成一个可迭代对象，并支持自动批处理、采样、乱序和多进程加载。
# 在这里，我们定义batch_size为64，即数据加载器中的每个元素将返回一批包含64个特征和标签。

batch_size = 64

# 创建数据加载器
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

# 第三步：创建模型
print("第三步：创建模型")
# 在PyTorch中定义神经网络，创建一个继承自 `nn.Module <https://pytorch.org/docs/stable/generated/torch.nn.Module.html>`_ 的类。
# 在 ``__init__`` 函数中定义网络的层，并在 ``forward`` 函数中指定数据如何通过网络。
# 为了加速神经网络的运算，将其移动到GPU或MPS上（如果可用）。

# 获取用于训练的CPU、GPU或MPS设备。
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


# 定义模型
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork().to(device)
print(model)

# 优化模型参数
# 要训练模型，我们需要一个 `损失函数 <https://pytorch.org/docs/stable/nn.html#loss-functions>`_
# 和一个 `优化器 <https://pytorch.org/docs/stable/optim.html>`_。
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


# 第四步：训练模型
print("第四步：训练模型")
# 在单个训练循环中，模型对训练数据集进行了多次迭代。每次迭代，它都会计算模型的预测，并比较预测结果与实际标签。
# 使用损失函数计算损失，并使用优化器来调整模型的权重，以降低损失。以下代码块演示了单个训练循环的过程。

# 训练循环
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # 将张量移动到设备上
        X, y = X.to(device), y.to(device)

        # 计算模型的误差
        pred = model(X)
        # 计算损失
        loss = loss_fn(pred, y)

        # 反向传播
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # 每N个 batch，打印 loss 和 acc
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


# 我们还希望在每个训练周期结束时测试模型的准确性。我们将计算模型在测试数据集上的准确率，并将其用作训练循环的输出。
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


# 训练过程经过多次迭代（*epochs*）。在每个时期，模型学习参数以做出更好的预测。
# 在每个epoch打印模型的准确性和损失；我们希望看到精度逐渐增加，而损失逐渐减少。
epochs = 5
for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")


# 第五步: 保存模型
print("第五步: 保存模型")
model_path = os.path.join(os.getenv('MODELS_ROOT'), 'model.pth')
# 保存模型的一种常见方法是序列化内部状态字典（包含模型参数）
torch.save(model.state_dict(), model_path)
print("Saved PyTorch Model State to model.pth")


# 第六步: 使用训练好的模型
print("第六步: 使用训练好的模型")
# 加载训练好的模型
# 加载模型的过程包括重新创建模型结构并将状态字典加载到其中。
model = NeuralNetwork().to(device)
model.load_state_dict(torch.load(model_path))

# 这个模型现在可以用来进行预测。
classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    x = x.to(device)
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')
