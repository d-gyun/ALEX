import numpy as np
from helpers import minmax
from enum import Enum
from sklearn.linear_model import LinearRegression

class Regression(Enum):
    LINEAR = 0

class RMI:
    def __init__(self, reg, data):
        self.gain = 10  # stage별 각 노드가 처리할 데이터 축소 비율
        self.leafNodeSize = 100
        self.data = np.array(data)
        self.data = np.sort(self.data)
        self.root = LearnedIndexNode(reg, 0, self.data, 0)
        self.nodeList = [self.root]
        self.stageIdx = [0]

        self.nodeSize = len(self.data)
        print("RMI 시작")
        while self.nodeSize > self.leafNodeSize:
            self.stageIdx.append(self.stageIdx[-1]+1)
            self.interval = len(self.data)//self.gain**(self.stageIdx[-1])
            for i in range(0, len(self.data)//self.interval):
                if self.interval*(i+1) < len(self.data):
                    childNode = LearnedIndexNode(reg, self.stageIdx[-1], self.data[self.interval*i:self.interval*(i+1)], idx=i)
                else:
                    childNode = LearnedIndexNode(reg, self.stageIdx[-1], self.data[self.interval*i:], idx=i)
                self.nodeList.append(childNode)
            self.nodeSize //= self.gain
        print("RMI STAGE = " + str(self.stageIdx[-1]))

    def find(self, key):
        """
           Search for a key in an indexed data structure using a predictive model.

           Parameters:
               key: The value to search for in the data structure.

           Returns:
               (position, error): A tuple containing the position of the key in the data structure and the number of checks performed.
               If the key is not found after max_checks, the function returns (-1, -1).
        """
        upper_bound = len(self.root.index) - 1
        lower_bound = 0
        pos = minmax(lower_bound, upper_bound, np.rint(self.nodeList[0].model.predict([[key]])[0][0]))
        print(f"Model in rootNode predicted that the requested key{key} is in position {pos}")

        idx = 1
        for node in self.nodeList[1:]:
            if node.stage == idx and pos in node.labels:
                pos = minmax(node.labels[0], node.labels[-1], np.rint(node.model.predict([[key]])[0][0]))
                print(f"Model in stage{idx} predicted that the requested key is in position {pos}")
                idx += 1
            if idx > self.stageIdx[-1]:
                self.node = node
                break

        error = 0
        while self.node.index[pos] != key:
            error += 1
            # Escape if you overextend in key.
            pos += 1 if self.node.index[pos] < key else -1

            if pos < self.node.labels[0]:
                print(f"pos:{pos}, node.labels[-1]:{node.labels[-1]} 이므로 이전 노드로 이동합니다.")
                self.node = self.nodeList[self.nodeList.index(self.node)-1]
                print(f"이전 노드의 범위는 {self.node.labels[0]}부터 {self.node.labels[-1]}입니다.")
            elif pos > self.node.labels[-1]:
                print(f"pos:{pos}, node.labels[-1]:{node.labels[-1]} 이므로 다음 노드로 이동합니다.")
                self.node = self.nodeList[self.nodeList.index(self.node)+1]
                print(f"다음 노드의 범위는 {self.node.labels[0]}부터 {self.node.labels[-1]}입니다.")

            if (pos < 0 or pos > (len(self.root.index)-1)):
                print(f"After making {error + 1} checks I figured that the key doesn't exist!")
                return -1, -1
        print(f"Found {key} in position {pos} after making {error + 1} checks")
        return pos, error

    def find_all(self):
        stats = np.zeros(100)
        for i in range(len(self.data)):
            pos, err = self.find(self.data[i])
            if pos == -1:
                stats[-1] += 1
            stats[err] += 1
        return stats

class LearnedIndexNode:
    def __init__(self, reg, stage, data, idx):
        self.reg = reg
        self.stage = stage
        self.model = None
        self.index = None
        # print(self.data)
        self.labels = np.where(~np.isnan(data))[0] + len(data)*idx  #Create Labels
        self.keys = data[np.where(~np.isnan(data))[0]]
        self.build()

    def build(self):
        self.index = {}
        for KEY, POS in zip(self.keys, self.labels):
            self.index[POS] = KEY
        if self.reg == Regression.LINEAR:
            X = self.keys.reshape(-1, 1)
            Y = self.labels.reshape(-1, 1)
            self.model = LinearRegression()
            print(self.index)
            self.model.fit(X, Y)