import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
import pickle
from dataset.mnist import load_mnist
from PIL import Image


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


network = init_network()
accuracy_cnt = 0

img = Image.open('test/test.jpg')
# 사진의 이미지를 흑백으로 바꾼다.
img = img.convert('L')
img = np.array(list(img.getdata()), dtype="float32")
# 정규화 한다.
img = img/255
y = predict(network, img)
print(np.argmax(y))