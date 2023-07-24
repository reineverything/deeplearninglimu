import torch
from torch import nn

net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
X = torch.rand(size=(2, 4))
print(net(X))

# 查看参数 net[0]第一层参数
print(net[2].state_dict())
print(type(net[2].bias))
print(net[2].bias)
print(net[2].bias.data)

# 查看所有参数
print(*[(name, param.shape) for name, param in net[0].named_parameters()])
print(*[(name, param.shape) for name, param in net.named_parameters()])
# 有了name后以通过名字来访问
print(net.state_dict()['2.bias'].data)


# 嵌套块
def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                         nn.Linear(8, 4), nn.ReLU())


def block2():
    net = nn.Sequential()
    for i in range(4):
        # 在这里嵌套
        net.add_module(f'block {i}', block1())  # 向网络中添加新的子模块，第一个参数是设置name，以后可以通过这个name访问每层
    return net


rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
print(rgnet)


# 参数初始化
def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)  # 均值为0，方差0.01初始化
        nn.init.zeros_(m.bias)  # 偏置设为0


net.apply(init_normal)  # 对net中所有的layer作为参数传入init_normal中
print(net[0].weight.data[0], net[0].bias.data[0])
print(net[2].weight.data[0], net[2].bias.data[0])


def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1)  # 全部设置为1
        nn.init.zeros_(m.bias)


net.apply(init_constant)


# 均匀分布初始化
def init_xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)


def init_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 42)


print(net)
net[0].apply(init_xavier)
net[2].apply(init_42)
print(net[0].weight.data[0])
print(net[2].weight.data)


# 自定义初始化
def my_init(m):
    if type(m) == nn.Linear:
        print("Init", *[(name, param.shape)
                        for name, param in m.named_parameters()][0])
        nn.init.uniform_(m.weight, -10, 10)  # 均匀初始化
        m.weight.data *= m.weight.data.abs() >= 5  # 绝对值大于5的，True，小于5的false。python中true-1,false-0.将绝对值小于5的元素置为0，而大于等于5的元素保留原值。


net.apply(my_init)
print(net[0].weight[:2])
net[0].weight.data[:] += 1  # 所有值+1
net[0].weight.data[0, 0] = 42  # 第一个参数设为42
print(net[0].weight.data[0])

# ##################################################################################################
# 参数绑定
# 我们需要给共享层一个名称，以便可以引用它的参数，即设置一层，同时使用两次
shared = nn.Linear(8, 8)
print(shared.weight.data[0])
print(shared.weight.data[0, 0])
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                    shared, nn.ReLU(),
                    shared, nn.ReLU(),
                    nn.Linear(8, 1))
print(net)
# 检查参数是否相同
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0, 0] = 100
# 确保它们实际上是同一个对象，而不只是有相同的值
print(net[2].weight.data[0] == net[4].weight.data[0])
