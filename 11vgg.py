import torch
from d2l.torch import d2l
from torch import nn


##################################################################
# 思想：用几个相同架构的块来替代多个卷积层，相对于AlexNet的升级是用多个小的卷积层代替一个大的卷积层

# vgg网络
def vgg(conv_arch):  # 存放n和channel的列表((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
    layers = []
    in_channels = 1  # 第一通道
    for (num_convs, out_channels) in conv_arch:
        layers.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels
    return nn.Sequential(
        *layers, nn.Flatten(1, 3),
        nn.Linear(6272, 4096),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.ReLU(),
        nn.Linear(4096, 10)
    )


# vgg_block
def vgg_block(n, in_channels, out_channels):  # n-卷积层数
    layers = []
    for i in range(n):
        layers.append(nn.Conv2d(in_channels, out_channels, 3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(2, stride=2))
    return nn.Sequential(*layers)


conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))  # 跑不动。。。
ratio = 4
small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
vgg11 = vgg(small_conv_arch)

image = torch.rand((1, 1, 224, 224))
output = vgg11(image)
print(output.shape)

batch_size = 20
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=227)


def evaluate_accuracy_gpu(net, data_iter, device=None):  # @save
    """使用GPU计算模型在数据集上的精度"""
    if isinstance(net, nn.Module):
        net.eval()  # 设置为评估模式，对于验证集应该这样设置
        if not device:
            device = next(iter(net.parameters())).device  # net.parameters()获得模型的所有参数，
            # 再放入迭代器中，获取第一个参数所在迭代器
    # 正确预测的数量，总预测的数量
    metric = d2l.Accumulator(2)
    with torch.no_grad():  # 关闭自动求导
        for X, y in data_iter:
            if isinstance(X, list):
                # BERT微调所需的（之后将介绍）
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(d2l.accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


# @save
def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    """用GPU训练模型(在第六章定义)"""

    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],  # xlim-x的范围
                            legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = d2l.Timer(), len(train_iter)  # 定时器类
    for epoch in range(num_epochs):
        # 训练损失之和，训练准确率之和，样本数
        metric = d2l.Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))

        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    animator.show()
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')


lr, num_epochs = 0.1, 10
train_ch6(vgg11, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())

# 结果
# loss 0.139, train acc 0.948, test acc 0.918
# 419.4 examples/sec on cuda:0
