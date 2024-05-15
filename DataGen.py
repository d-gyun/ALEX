import numpy as np
from enum import Enum


class Distribution(Enum):
    LINEAR = 0
    LOGNORMAL = 1
    NORMAL = 2
    RANDOM = 3


class DataGen:
    def __init__(self, distribution, size, range_size):
        self.data = None
        self.distribution = distribution
        self.size = size
        self.range_size = range_size

    def generate(self):
        if self.distribution == Distribution.LINEAR:
            self.data = self.linear_data()
        elif self.distribution == Distribution.LOGNORMAL:
            self.data = self.log_normal()
        elif self.distribution == Distribution.NORMAL:
            self.data = self.normal()
        elif self.distribution == Distribution.RANDOM:
            self.data = self.rand_data()
        else:
            print("I can't recognize the given distribution please give one of those(linear, normal, random)")
            return None
        return self.data

    def linear_data(self):
        a = np.random.randint(10)
        b = np.random.randint(10)
        print(f"Creating linear data according to the {a}x+{b} line.")
        data = []
        for i in range(self.size):
            data.append(a*i+b)
        return data

    def log_normal(self):
        data = []
        for i in range(self.size):
            mu, sigma = 0, 0.1
            sample = np.random.lognormal(mean=mu, sigma=sigma, size=100)
            data.append(np.prod(sample))
        data = np.array(data) / np.min(data)
        return np.rint(data)

    def normal(self):
        mean_val = self.size*50  # 평균값
        std_dev = self.size*20  # 표준편차

        # 원하는 범위 설정
        lower_limit = 0
        upper_limit = mean_val*2

        # 정규 분포를 따르는 난수 생성 및 범위 제한
        random_data = []
        while len(random_data) < self.size:
            value = np.rint(np.random.normal(mean_val, std_dev)).astype(int)
            if lower_limit <= value <= upper_limit:
                random_data.append(value)

        data = np.array(random_data)
        return data

    def rand_data(self):
        """
            Generates an array of randomly distributed values.
        """
        data = []
        for i in range(self.size):
            sample = np.random.randint(self.range_size)
            data.append(sample)
        data = np.array(data)
        return data
