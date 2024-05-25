import numpy as np
import osmnx as ox
from enum import Enum

class Distribution(Enum):
    LINEAR = 0
    LOGNORMAL = 1
    NORMAL = 2
    RANDOM = 3
    LONGITUDES = 4
    LONGLAT = 5

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
        elif self.distribution == Distribution.LONGITUDES:
            self.data = self.generate_longitudes()
        elif self.distribution == Distribution.LONGLAT:
            self.data = self.generate_longlat()
        else:
            print("I can't recognize the given distribution please give one of those(linear, normal, random, lognormal, longitudes, longlat)")
            return None
        return self.data

    def linear_data(self):
        a = np.random.randint(10)
        b = np.random.randint(10)
        data = [a * i + b for i in range(self.size)]
        return data

    def log_normal(self):
        mu, sigma = 0, 1
        data = np.random.lognormal(mean=mu, sigma=sigma, size=self.size * 10)
        data = np.unique(data)
        while len(data) < self.size:
            additional_data = np.random.lognormal(mean=mu, sigma=sigma, size=self.size)
            data = np.unique(np.concatenate((data, additional_data)))
        data = np.sort(data[:self.size])
        data = data / np.max(data) * self.range_size
        return np.rint(data)

    def normal(self):
        mean_val = self.size * 50
        std_dev = self.size * 20
        lower_limit = 0
        upper_limit = mean_val * 2
        random_data = []
        while len(random_data) < self.size:
            value = np.rint(np.random.normal(mean_val, std_dev)).astype(int)
            if lower_limit <= value <= upper_limit:
                random_data.append(value)
        data = np.array(random_data)
        return data

    def rand_data(self):
        data = [np.random.randint(self.range_size) for _ in range(self.size)]
        return np.array(data)

    def generate_longitudes(self):
        place_name = "Manhattan, New York, USA"
        graph = ox.graph_from_place(place_name, network_type='all')
        nodes, edges = ox.graph_to_gdfs(graph)
        longitudes = nodes['x'].values
        np.random.shuffle(longitudes)
        longitudes = longitudes[:self.size]
        return np.sort(longitudes)

    def generate_longlat(self):
        place_name = "Manhattan, New York, USA"
        graph = ox.graph_from_place(place_name, network_type='all')
        nodes, edges = ox.graph_to_gdfs(graph)
        longlat_data = list(zip(nodes['x'].values, nodes['y'].values))
        np.random.shuffle(longlat_data)
        longlat_data = longlat_data[:self.size]
        longlat_data.sort()
        return longlat_data
