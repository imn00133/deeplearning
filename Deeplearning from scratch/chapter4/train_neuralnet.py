import sys, os
sys.path.append(os.pardir)

import numpy as np
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

(train_x, train_t), (test_x, test_t) = load_mnist(normalize=True, one_hot_label=True)
train_loss_list = []
train_acc_list = []
test_acc_list = []

# 하이퍼파라미터
iters_num = 10000
train_size = train_x.shape[0]
batch_size = 100
learning_rate = 0.1

# 1에폭당 반복 수
iter_per_epoch = max(train_size / batch_size, 1)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

for i in range(iters_num):
    # 미니배치 만들기
    batch_mask = np.random.choice(train_size, batch_size)
    batch_x = train_x[batch_mask]
    batch_t = train_t[batch_mask]

    # 기울기 계산
    grad = network.numerical_gradient(batch_x, batch_t)

    # 매개변수 갱신
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    # 학습 경과 기록
    loss = network.loss(batch_x, batch_t)
    train_loss_list.append(loss)

    # 1에폭당 정확도 계산
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(train_x, train_t)
        test_acc = network.accuracy(test_x, test_t)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc: %f, %f" %(train_acc, test_acc))

    print("%d번째" %i)
