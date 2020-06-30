import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist

(train_x, train_t), (test_x, test_t) = load_mnist(flatten=True, normalize=False)

print(train_x.shape)
print(train_t.shape)
print(test_x.shape)
print(test_t.shape)
