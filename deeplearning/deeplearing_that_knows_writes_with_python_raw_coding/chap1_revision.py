import numpy as np
import csv
import time
from typing import Callable, Generator

np.random.seed(1234)
RND_MEAN = 0
RND_STD = 0.0030
LEARNING_RATE = 0.001

INPUT_CNT, OUTPUT_CNT = 10, 1


def randomize():
    np.random.seed(time.time())


# https://www.kaggle.com/maik3141/abalone
def load_abalone_dataset() -> np.ndarray:
    with open('train_data/chap1_abalone.csv') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader, None)
        rows = [row for row in csv_reader]

    data = np.zeros([len(rows), INPUT_CNT + OUTPUT_CNT])

    for row_index, row in enumerate(rows):
        if row[0] == 'I':
            data[row_index, 0] = 1
        if row[0] == 'M':
            data[row_index, 1] = 1
        if row[0] == 'F':
            data[row_index, 2] = 1
        data[row_index, 3:] = row[1:]
    return data


def init_model():
    global weight, bias
    weight = np.random.normal(RND_MEAN, RND_STD, [INPUT_CNT, OUTPUT_CNT])
    bias = np.zeros([OUTPUT_CNT])


def save_train_set(data: np.ndarray, shuffle_map: np.ndarray, mb_size: int, step_count: int) \
        -> Callable[[], Generator[np.ndarray, np.ndarray]]:
    def new_train_set() -> Generator[np.ndarray, np.ndarray]:
        nonlocal shuffle_map
        np.random.shuffle(shuffle_map)

        for step in range(step_count):
            yield get_x_y(data[shuffle_map[mb_size * step:mb_size * (step + 1)]])
        # return (get_x_y(data[shuffle_map[mb_size * step:mb_size * (step + 1)]]) for step in range(step_count))
    return new_train_set


def get_x_y(data: np.ndarray) -> (np.ndarray, np.ndarray):
    return data[:, :-OUTPUT_CNT], data[:, -OUTPUT_CNT:]


def get_test_train_data(data: np.ndarray, mb_size: int) \
        -> ((np.ndarray, np.ndarray), Callable[[], Generator[np.ndarray, np.ndarray]]):
    shuffle_map = np.arange(data.shape[0])
    np.random.shuffle(shuffle_map)
    step_count = int(data.shape[0] * 0.8) // mb_size
    test_begin_idx = step_count * mb_size
    test_data = data[shuffle_map[test_begin_idx:]]
    return get_x_y(test_data), save_train_set(data, shuffle_map[:test_begin_idx], mb_size, step_count)


def forward_neuralnet(x):
    global weight, bias
    # p.79 파이썬 인터프리터 x - numpy의 broadcasting연산
    output = np.matmul(x, weight) + bias
    return output, x


def backprop_neuralnet(g_output, x):
    global weight, bias
    g_output_w = x.transpose()

    g_w = np.matmul(g_output_w, g_output)
    g_b = np.sum(g_output, axis=0)

    weight -= LEARNING_RATE * g_w
    bias -= LEARNING_RATE * g_b


def forward_postproc(ouput, y):
    diff = ouput - y
    square = np.square(diff)
    loss = np.mean(square)
    return loss, diff


def backprop_postproc(g_loss, diff):
    shape = diff.shape

    g_loss_square = np.ones(shape) / np.prod(shape)
    g_square_diff = 2 * diff
    g_diff_output = 1

    g_square = g_loss_square * g_loss
    g_diff = g_square_diff * g_square
    g_output = g_diff_output * g_diff

    return g_output


def eval_accuracy(output, y):
    mdiff = np.mean(np.abs((output - y) / y))
    return 1 - mdiff


def run_train(x, y):
    output, aux_nn = forward_neuralnet(x)
    loss, aux_pp = forward_postproc(output, y)
    accuracy = eval_accuracy(output, y)

    g_loss = 1.0
    g_output = backprop_postproc(g_loss, aux_pp)
    backprop_neuralnet(g_output, aux_nn)

    return loss, accuracy


def run_test(x, y):
    output, _ = forward_neuralnet(x)
    accuracy = eval_accuracy(output, y)
    return accuracy


def train_and_test(data: np.array, epoch_count: int, mb_size: int, report: int):
    (test_x, test_y), new_train_set = get_test_train_data(data, mb_size)

    for epoch in range(epoch_count):
        losses, accs = [], []

        for train_x, train_y in new_train_set():
            loss, acc = run_train(train_x, train_y)
            losses.append(loss)
            accs.append(acc)

        if report > 0 and (epoch + 1) % report == 0:
            acc = run_test(test_x, test_y)
            print(f"Epoch {epoch + 1}: loss={np.mean(losses):5.3f}, accuracy={np.mean(accs):5.3f}/{acc:5.3f}")

    final_acc = run_test(test_x, test_y)
    print(f"\nFinal Test: final accuracy = {final_acc:5.3f}")


def abalone_exec(epoch_count=10, mb_size=10, report=1):
    data = load_abalone_dataset()
    init_model()
    train_and_test(data, epoch_count, mb_size, report)


if __name__ == "__main__":
    abalone_exec()
