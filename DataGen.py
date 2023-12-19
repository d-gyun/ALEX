import numpy as np
from enum import Enum


class Distribution(Enum):
    LINEAR = 0
    NORMAL = 1
    RANDOM = 2


class DataGen:
    def __init__(self, distribution, size):
        self.data = None
        self.distribution = distribution
        self.size = size

    def generate(self):
        if self.distribution == Distribution.LINEAR:
            self.data = self.linear_data()
        elif self.distribution == Distribution.NORMAL:
            self.data = self.log_normal()
        elif self.distribution == Distribution.RANDOM:
            self.data = self.rand_data()
        else:
            print("I can't recognize the given distribution please give one of those(linear, normal, random)")
            return None
        return self.data

    def linear_data(self):
        """
        Generates linear data and labels.

        Parameters:
        amount (int): The number of data points to generate.
        a (float): The slope of the linear relationship.
        b (float): The y-intercept of the linear relationship.

        Returns:
        numpy.ndarray, numpy.ndarray: Arrays containing generated data and labels.
        """
        a = np.random.randint(10)
        b = np.random.randint(10)
        print(f"Creating linear data according to the {a}x+{b} line.")
        data = []
        for i in range(self.size):
            data.append(a*i+b)
        return data

    def log_normal(self):
        """
           Generates an array of log-normal distributed values and scales them.

           Parameters:
           amount (int): The number of log-normal samples to generate.

           Returns:
           numpy.ndarray: An array of log-normal samples, scaled by the minimum value.
        """
        data = []
        for i in range(self.size):
            sample = 10. + np.random.standard_normal(100)
            data.append(np.prod(sample))
        data = np.array(data) / np.min(data)
        return np.rint(data)

    def rand_data(self):
        """
            Generates an array of randomly distributed values.

            Returns:
            numpy.ndarray: An array of random samples.
        """
        data = []
        for i in range(self.size):
            sample = np.random.randint(10000)
            data.append(sample)
        data = np.array(data)
        return data
