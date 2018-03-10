import sys, os
sys.path.append(os.pardir)

import numpy as np
from dataset.mnist import load_mnist


(train_x, train_t), (test_x, test_t) = load_mnist(normalize=True, one_hot_label=True)

train_size = train_x.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)
x_batch = train_x[batch_mask]
t_batch = train_t[batch_mask]

