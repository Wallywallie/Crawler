import re
import matplotlib.pyplot as plt


def draw_1():
    # 定义正则表达式来提取数据
    epoch_pattern = re.compile(r'Epoch (\d+)/\d+')
    loss_pattern = re.compile(r'loss: ([\d.]+)')
    acc_pattern = re.compile(r'acc: ([\d.]+)')

    # 初始化列表存储数据
    epochs = []
    losses = []
    accuracies = []

    # 读取文件并解析数据
    with open('loss.log', 'r') as file:
        for line in file:
            epoch_match = epoch_pattern.search(line)
            loss_match = loss_pattern.search(line)
            acc_match = acc_pattern.search(line)
            print( epoch_match )
            
            # 如果找到对应数据，添加到列表中
            if epoch_match:
                epochs.append(int(epoch_match.group(1)))
            if loss_match:
                losses.append(float(loss_match.group(1)))
            if acc_match:
                accuracies.append(float(acc_match.group(1)))

    print(epochs)
    # 绘制 Loss 曲线
    plt.figure(figsize=(12, 5))

    # Loss 图
    plt.subplot(1, 2, 1)
    plt.plot(epochs, losses, marker='o', color='b', label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Epoch vs Loss')
    plt.legend()
    plt.grid(True)

    # Accuracy 图
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracies, marker='o', color='g', label='Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Epoch vs Accuracy')
    plt.legend()
    plt.grid(True)

    # 显示图表
    plt.tight_layout()
    plt.show()


def draw_loss_epoch():


    # 数据
    data = [
    {'loss': 1.1875, 'epoch': 0.86},
    {'loss': 0.1599, 'epoch': 2.0},
    {'loss': 0.1228, 'epoch': 2.86},
    {'loss': 0.0564, 'epoch': 4.0},
    {'loss': 0.0357, 'epoch': 4.86},
    {'loss': 0.0158, 'epoch': 6.0},
    {'loss': 0.0098, 'epoch': 6.86},
    {'loss': 0.0121, 'epoch': 8.0},
    {'loss': 0.0079, 'epoch': 8.86},
    {'loss': 0.0078, 'epoch': 10.0},
    {'loss': 0.003, 'epoch': 10.86},
    {'loss': 0.0011, 'epoch': 12.0},
    {'loss': 0.0008, 'epoch': 12.86},
    {'loss': 0.0005, 'epoch': 14.0},
    {'loss': 0.0001, 'epoch': 14.86},
    {'loss': 0.0001, 'epoch': 16.0},
    {'loss': 0.0001, 'epoch': 16.86},
    {'loss': 0.0, 'epoch': 18.0},
    {'loss': 0.0, 'epoch': 18.86},
    {'loss': 0.0, 'epoch': 20.0},
    {'loss': 0.0, 'epoch': 20.86},
    {'loss': 0.0, 'epoch': 21.43},
    ]

    # 提取数据
    epochs = [point['epoch'] for point in data]
    losses = [point['loss'] for point in data]

    # 绘制图表
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses, marker='o', label='Loss')
    plt.title("Loss vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    #plt.yscale("log")  # 对数坐标轴
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()

draw_loss_epoch()    
