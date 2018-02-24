import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
import pickle
from dataset.mnist import load_mnist


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a-c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y


def get_data():
    (train_x, train_t), (test_x, test_t) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return test_x, test_t


def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y


test_x, test_t = get_data()
network = init_network()

batch_size = 100
accuracy_cnt = 0

for i in range(0, len(test_x), batch_size):
    x_batch = test_x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    # 확률이 가장 높은 원소의 인덱스를 얻는다.
    p = np.argmax(y_batch, axis=1)
    accuracy_cnt += np.sum(p == test_t[i:i+batch_size])

print("Accuracy:" + str(float(accuracy_cnt) / len(test_x)))
