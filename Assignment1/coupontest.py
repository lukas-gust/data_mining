import numpy as np
import random
import time
import matplotlib.pyplot as plt


def coupon(n):
    collected = np.full(n, False)
    k = 0
    unique = 0
    while unique < n:
        num = int(n*random.random())
        if not collected[num]:
            unique += 1
            collected[num] = True
        k += 1

    return k


def coupon_experiement_loop(m, n):
    ks = np.zeros(m)
    for i in range(m):
        ks[i] = coupon(n)


if __name__ == '__main__':
    ns = np.linspace(300, 20000, num=5, dtype='int')
    ms = np.linspace(400, 5000, num=3, dtype='int')

    runtimes = np.zeros((3, len(ns)))

    for i, m in enumerate(ms):
        for j, n in enumerate(ns):
            start = time.time()
            coupon_experiement_loop(m, n)
            end = time.time()
            runtimes[i, j] = end - start

    plt.figure(figsize=(12, 5))
    plt.plot(ns, runtime[:, 0], label='m = 400')
    plt.plot(ns, runtime[:, 1], label='m = 2300')
    plt.plot(ns, runtime[:, 2], label='m = 5000')
    plt.xticks(ns)
    plt.xlabel('Domain [n]')
    plt.ylabel('Runtime [seconds]')
    plt.title('Runtime vs n')
    plt.legend()
    plt.savefig('coupon-runtime.png')