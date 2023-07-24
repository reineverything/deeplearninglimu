from mxnet import autograd, np, npx
from d2l import mxnet as d2l
import torch
from torch import nn

npx.set_np()


# 卷积操作
def corr2d(X, K):  # @save
    """计算二维互相关运算"""
    h, w = K.shape
    Y = np.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y


X = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
K = np.array([[0.0, 1.0], [2.0, 3.0]])
print(corr2d(X, K))


#############################################################
# 卷积层
class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))  # 生成kernel*kernel的二维向量
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias


# 边缘检测
X = torch.ones((6, 8))
X[:, 2:6] = 0
print(X)
K = torch.tensor([[1.0, -1.0]])  # 卷积核，没有变化时是0，变化时是1或者-1
Y = corr2d(X, K)
print(Y)

print(corr2d(X.t(), K))  # 此时转置变成水平边缘，无法检测

#################################################################
# 学习由X生成Y的卷积核
# 构造一个二维卷积层，它具有1个输出通道和形状为（1，2）的卷积核
conv2d = nn.Conv2d(1, 1, kernel_size=(1, 2), bias=False)

# 这个二维卷积层使用四维输入和输出格式（批量大小、通道、高度、宽度），
# 其中批量大小和通道数都为1
X = X.reshape((1, 1, 6, 8))  # 1-灰度图通道数，1-reshape-样本维度即样本数目
Y = torch.from_numpy(Y.reshape((1, 1, 6, 7)).asnumpy())  # torch中类型查看type,numpy中是dtype
lr = 3e-2  # 学习率

for i in range(20):
    Y_hat = conv2d(X)
    l = (Y_hat - Y) ** 2  # 均方误差MSE = 1/n * ∑(xi - yi)²
    conv2d.zero_grad()  # 计算梯度时是累积计算的，每一个batch的梯度应该单独计算
    l.sum().backward()  # backward() 方法实现了一种自动微分的计算过程，可以自动
    # 求解标量相对于任意输入节点（包括模型参数、输入数据等）的梯度，并将结果存储在节点的 .grad 属性中。
    # 迭代卷积核
    conv2d.weight.data[:] -= lr * conv2d.weight.grad
    if (i + 1) % 2 == 0:
        print(f'epoch {i + 1}, loss {l.sum():.3f}')

print(conv2d.weight.data)