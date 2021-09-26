import numpy as np
import csv
import time

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

    global data
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


def arrange_data(mb_size: int):
    global data, shuffle_map, test_begin_idx
    shuffle_map = np.arange(data.shape[0])
    np.random.shuffle(shuffle_map)
    step_count = int(data.shape[0] * 0.8) // mb_size
    test_begin_idx = step_count * mb_size
    return step_count


def get_test_data():
    global data, shuffle_map, test_begin_idx
    test_data = data[shuffle_map[test_begin_idx:]]
    return test_data[:, :-OUTPUT_CNT], test_data[:, -OUTPUT_CNT:]


def get_train_data(mb_size: int, step: int):
    global data, shuffle_map, test_begin_idx
    if step == 0:
        np.random.shuffle(shuffle_map[:test_begin_idx])
    train_data = data[shuffle_map[mb_size * step: mb_size * (step + 1)]]
    return train_data[:, :-OUTPUT_CNT], train_data[:, -OUTPUT_CNT:]


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


def train_and_test(epoch_count: int, mb_size: int, report: int):
    step_count = arrange_data(mb_size)
    test_x, test_y = get_test_data()

    for epoch in range(epoch_count):
        losses, accs = [], []

        for step in range(step_count):
            train_x, train_y = get_train_data(mb_size, step)
            loss, acc = run_train(train_x, train_y)
            losses.append(loss)
            accs.append(acc)

        if report > 0 and (epoch + 1) % report == 0:
            acc = run_test(test_x, test_y)
            print(f"Epoch {epoch + 1}: loss={np.mean(losses):5.3f}, accuracy={np.mean(accs)/acc:5.3f}")

    final_acc = run_test(test_x, test_y)
    print(f"\nFinal Test: final accuracy = {final_acc:5.3f}")


def abalone_exec(epoch_count=10, mb_size=10, report=1):
    load_abalone_dataset()
    init_model()
    train_and_test(epoch_count, mb_size, report)


if __name__ == "__main__":
    abalone_exec()
