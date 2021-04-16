import torch
import time
import os
import matplotlib.pyplot as plt
import torch.optim as optim
import math

LR = 0.0005    # 设置学习率
EPOCH_NUM = 10  # 训练轮次

# 导入 定义的 ResNet50 和 导入的数据
from ResNet50 import resnet50
from dataload import new_train_loader, new_test_loader


def time_since(since):
    s = time.time() - since
    m = math.floor(s/60)
    s -= m*60
    return '%dm %ds' % (m, s)

model = resnet50()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

train_data = new_train_loader
test_data = new_test_loader

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

def train(epoch, loss_list):
    running_loss = 0.0
    for batch_idx, data in enumerate(new_train_loader, 0):
        inputs, target = data[0], data[1]
        inputs, target = inputs.to(device), target.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        loss_list.append(loss.item())
        running_loss += loss.item()
        if batch_idx % 100 == 99:
            print(f'[{time_since(start)}] Epoch {epoch}', end='')
            print('[%d, %5d] loss:%.3f' % (epoch + 1, batch_idx + 1, running_loss / 100))
            running_loss = 0.0

    return loss_list

def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for _, data in enumerate(new_test_loader, 0):
            inputs, target = data[0], data[1]
            inputs, target = inputs.to(device), target.to(device)
            outputs = model(inputs)
            _, prediction = torch.max(outputs.data, dim=1)

            total += target.size(0)
            correct += (prediction == target).sum().item()
        print('Accuracy on test set: (%d/%d)%d %%' % (correct, total, 100 * correct / total))
        with open("test.txt", "a") as f:
            f.write('Accuracy on test set: (%d/%d)%d %% \n' % (correct, total, 100 * correct / total))

if __name__ == '__main__':
    start = time.time()

    with open("test.txt", "a") as f:
        f.write('Start write!!! \n')

    loss_list = []
    for epoch in range(EPOCH_NUM):
        train(epoch, loss_list)
        test()
    torch.save(model.state_dict(), 'Model.pth')

    x_ori = []
    for i in range(len(loss_list)):
        x_ori.append(i)
    plt.title("Graph")
    plt.plot(x_ori, loss_list)
    plt.ylabel("Y")
    plt.xlabel("X")
    plt.show()


