def f2(x):
    print("f2")
    print(x)


def f3():
    print("f3")
    return f2(x)


def f1(f):
    pass


# f3가 return값으로 f2(x)를 주는데, x의 변수값은 이미 스코프에서 찾아 넘겨주는 것으로 보인다.
x = 1
f1(f3())
