import torch
from torch import nn
from torch.nn import functional as F

x = torch.arange(4)
torch.save(x, 'x-file')
x2 = torch.load('x-file')
print(x2)

y = torch.zeros(4)
torch.save([x, y], 'x-files')
x2, y2 = torch.load('x-files')
print((x2, y2))

mydict = {'x': x, 'y': y}
torch.save(mydict, 'mydict')
mydict2 = torch.load('mydict')
print(mydict2)


#########################################################################################
# 加载和保存模型
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))


net = MLP()
X = torch.randn(size=(2, 20))
Y = net(X)
torch.save(net.state_dict(), 'mlp.params')  # 其中包含了深度学习模型net所有可学习参数（如权重和偏置）的名称和对应的张量。
clone = MLP()  # 创建时已经被随机初始化
clone.load_state_dict(torch.load('mlp.params'))
print(clone.eval())  # 返回一个模型的副本，其中包含原始模型的参数和结构

Y_clone = clone(X)
print(Y_clone == Y)