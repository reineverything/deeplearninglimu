from torch import nn
import torch
from d2l import torch as d2l


class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.Conv1 = nn.Conv2d(1, 96, 11, 4, padding=0)
        self.relu = nn.ReLU()
        self.Pooling1 = nn.MaxPool2d(3, 2)
        self.Conv2 = nn.Conv2d(96, 256, 5, 1, 2)
        self.Pooling2 = nn.MaxPool2d(3, 2, 0)
        self.Conv3 = nn.Conv2d(256, 384, 3, 1, 1)
        self.Conv4 = nn.Conv2d(384, 384, 3, 1, 1)
        self.Conv5 = nn.Conv2d(384, 256, 3, 1, 1)
        self.Pooling5 = nn.MaxPool2d(3, 2, 0)
        self.flatten = nn.Flatten(1,3)
        self.linear1 = nn.Linear(256 * 6 * 6, 4096)
        self.dropout = nn.Dropout(0.4)
        self.linear2 = nn.Linear(4096, 4096)
        self.linear3 = nn.Linear(4096, 10)
        self.softmax = nn.Softmax(0)

    def forward(self, X):
        X = self.Conv1(X)
        X = self.relu(X)
        X = self.Pooling1(X)
        X = self.Conv2(X)
        X = self.relu(X)
        X = self.Pooling2(X)
        X = self.Conv3(X)
        X = self.relu(X)
        X = self.Conv4(X)
        X = self.relu(X)
        X = self.Conv5(X)
        X = self.relu(X)
        X = self.Pooling5(X)
        X = self.flatten(X)
        X = self.linear1(X)
        X = self.dropout(X)
        X = self.linear2(X)
        X = self.dropout(X)
        X = self.linear3(X)
        return X


batch_size = 50
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=227)

# image = torch.rand((2, 1, 227, 227))  # 第一个维度表示batch，第二个表示channel
# print(image)
model = AlexNet()
# print(model)
# output = model(image)
# print(output.shape)


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
train_ch6(model, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())