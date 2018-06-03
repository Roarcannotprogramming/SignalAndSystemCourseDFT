import numpy as np
from matplotlib import pyplot as plt


class DFT_calc:
    def __init__(self, lambda_func, N, fs):
        # lambda_func 为函数映射
        # x_list 为函数横坐标列表
        self.func = lambda_func
        self.N = N
        self.x_list = [n for n in range(int(-N/2), int(N/2), 1)]
        x_list_sample = [n / fs for n in self.x_list]
        self.y_list = list(map(self.func, x_list_sample))
        self.x_list_plus = []

    def normal_calc(self):
        # 常规方法计算DFT
        self.x_list_plus = [num-min(self.x_list) for num in self.x_list]
        for k in self.x_list_plus:
            sum_calc = 0
            for xn, n in zip(self.y_list, self.x_list):
                sum_calc += xn * np.exp(-2j * np.pi * k * n / self.N)
            yield sum_calc  # 生成X(k)

    def fft_calc(self):
        return list(np.fft.fft(np.array(self.y_list)))


if __name__ == '__main__':
    a = DFT_calc(lambda x: np.exp(-np.pi * (x ** 2)), 1000, 1)
    list_a = list(a.normal_calc())
    list_b = a.fft_calc()
    plt.plot(a.x_list_plus, list_b)
    plt.show()
    plt.plot(a.x_list_plus, list_a)
    plt.show()
    # for elem in list_b:
    #     print(elem)
