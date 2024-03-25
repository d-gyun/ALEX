import math
import numpy as np
from helpers import minmax
from enum import Enum
from sklearn.linear_model import LinearRegression

class Regression(Enum):
    LINEAR = 0

class RMI:
    def __init__(self, reg, data):
        self.fanout = 8  # node의 fanout
        self.leafNodeSize = 100
        self.data = np.array(data)
        self.data = np.sort(self.data)
        self.root = LearnedIndexNode(reg, 0, self.data,0)
        self.nodeList = [self.root]
        self.stageIdx = [0]

        self.nodeSize = len(self.data)
        print("RMI 시작")
        while self.nodeSize > self.leafNodeSize:
            self.stageIdx.append(self.stageIdx[-1]+1)
            # internal node
            if self.nodeSize // self.fanout > self.leafNodeSize:
                self.interval = len(self.data)//self.fanout**(self.stageIdx[-1])
            # data node
            else:
                self.interval = math.ceil(self.leafNodeSize*(0.6))

            for i in range(0, len(self.data)//self.interval + (1 if (len(self.data)%self.interval) else 0)):
                if self.interval*(i+1) < len(self.data):
                    childNode = LearnedIndexNode(reg, self.stageIdx[-1], self.data[self.interval*i:self.interval*(i+1)], self.interval*i)
                else:
                    childNode = LearnedIndexNode(reg, self.stageIdx[-1], self.data[self.interval*i:], self.interval*i)
                self.nodeList.append(childNode)
            self.nodeSize //= self.fanout
            '''
            # data node (GAP create)
            else:
                self.interval = math.ceil(self.leafNodeSize*(0.6))
                for i in range(0, len(self.data) // self.interval):
                    if self.interval * (i + 1) < len(self.data):
                        childNode = LearnedIndexNode(reg, self.stageIdx[-1],
                                                self.data[self.interval * i:self.interval * (i + 1)], idx=i)
                    else:
                        childNode = LearnedIndexNode(reg, self.stageIdx[-1], self.data[self.interval * i:], idx=i)
                    self.nodeList.append(childNode)
            '''
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
        upper_bound = np.float64(len(self.root.index) - 1)
        lower_bound = 0.0
        pos = minmax(lower_bound, upper_bound, np.rint(self.nodeList[0].model.predict([[key]])[0][0]))
        # print(f"Model in rootNode predicted that the requested key{key} is in position {pos}")

        idx = 1
        for node in self.nodeList[1:]:
            if node.stage == idx and pos in node.labels:
                pos = minmax(lower_bound, upper_bound, np.rint(node.model.predict([[key]])[0][0]))
                # print(f"Model in stage{idx} predicted that the requested key is in position {pos}")
                idx += 1
            if idx > self.stageIdx[-1]:
                self.node = node
                # 최종 리프노드 선정
                while True:
                    if pos < self.node.labels[0]:
                        # print(f"pos:{pos}, node.labels[0]:{node.labels[0]} 이므로 이전 노드로 이동합니다.")
                        self.node = self.nodeList[self.nodeList.index(self.node) - 1]
                        # print(f"이전 노드의 범위는 {self.node.labels[0]}부터 {self.node.labels[-1]}입니다.")
                    elif pos > self.node.labels[-1]:
                        # print(f"pos:{pos}, node.labels[-1]:{node.labels[-1]} 이므로 다음 노드로 이동합니다.")
                        self.node = self.nodeList[self.nodeList.index(self.node) + 1]
                        # print(f"다음 노드의 범위는 {self.node.labels[0]}부터 {self.node.labels[-1]}입니다.")
                    else:
                        break
                break

        error = 0
        l, r = 0, 0 # binary search를 위한 range
        i = 0
        # exponential search
        if self.node.index[pos] < key:
            while self.node.index[pos] < key:
                error += 1
                l = int(pos + 1)
                pos += 2 ** i
                if pos > (len(self.root.index) - 1):
                    pos = len(self.root.index) - 1
                if pos > self.node.labels[-1]:
                    # print(f"pos:{pos}, node.labels[-1]:{self.node.labels[-1]} 이므로 다음 노드로 이동합니다.")
                    self.node = self.nodeList[self.nodeList.index(self.node) + 1]
                    # print(f"다음 노드의 범위는 {self.node.labels[0]}부터 {self.node.labels[-1]}입니다.")
                i += 1

            if self.node.index[pos] == key:
                print(f"Found {key} in position {pos} after making {error + 1} checks")
                return pos, error
            elif self.node.index[pos] > key:
                r = int(pos)
                if l < self.node.labels[0]:
                    self.node = self.nodeList[self.nodeList.index(self.node) - 1]
                    for chk in range(l, self.node.labels[-1] + 1):
                        error += 1
                        if self.node.index[chk] == key:
                            pos = chk
                            print(f"Found {key} in position {pos} after making {error + 1} checks")
                            return pos, error
                    self.node = self.nodeList[self.nodeList.index(self.node) + 1]
                    for chk in range(self.node.labels[0], r):
                        error += 1
                        if self.node.index[chk] == key:
                            pos = chk
                            print(f"Found {key} in position {pos} after making {error + 1} checks")
                            return pos, error
                else:
                    for chk in range(l, r):
                        error += 1
                        if self.node.index[chk] == key:
                            pos = chk
                            print(f"Found {key} in position {pos} after making {error + 1} checks")
                            return pos, error
        else:
            while self.node.index[pos] > key:
                error += 1
                r = int(pos)
                pos -= 2 ** i
                if pos < 0:
                    pos = 0
                if pos < self.node.labels[0]:
                    print(f"pos:{pos}, node.labels[0]:{self.node.labels[0]} 이므로 이전 노드로 이동합니다.")
                    self.node = self.nodeList[self.nodeList.index(self.node) - 1]
                    print(f"이전 노드의 범위는 {self.node.labels[0]}부터 {self.node.labels[-1]}입니다.")
                i += 1

            if self.node.index[pos] == key:
                print(f"Found {key} in position {pos} after making {error + 1} checks")
                return pos, error
            elif self.node.index[pos] < key:
                l = int(pos + 1)
                if r > self.node.labels[-1]:
                    for chk in range(l, self.node.labels[-1] + 1):
                        error += 1
                        if self.node.index[chk] == key:
                            pos = chk
                            print(f"Found {key} in position {pos} after making {error + 1} checks")
                            return pos, error
                    self.node = self.nodeList[self.nodeList.index(self.node) + 1]
                    for chk in range(self.node.labels[0], r):
                        error += 1
                        if self.node.index[chk] == key:
                            pos = chk
                            print(f"Found {key} in position {pos} after making {error + 1} checks")
                            return pos, error
                else:
                    for chk in range(l, r):
                        error += 1
                        if self.node.index[chk] == key:
                            pos = chk
                            print(f"Found {key} in position {pos} after making {error + 1} checks")
                            return pos, error
        if (pos == 0 or pos == (len(self.root.index) - 1)):
            print(f"After making {error + 1} checks I figured that the key doesn't exist!")
            return -1, -1
        print(f"Found {key} in position {pos} after making {error + 1} checks")

    def find_all(self):
        stats = np.zeros(100)
        for i in range(len(self.data)):
            pos, err = self.find(self.data[i])
            if pos == -1:
                stats[-1] += 1
            stats[err] += 1
        return stats

    def predict_pos(self, key):
        upper_bound = np.float64(len(self.root.index) - 1)
        lower_bound = 0.0
        print(self.nodeList[0].model.predict([[key]])[0][0])
        pos = minmax(lower_bound, upper_bound, np.rint(self.nodeList[0].model.predict([[key]])[0][0]))
        print(f"Model in rootNode predicted that the requested key{key} is inserted in position {pos}")

        idx = 1
        for node in self.nodeList[1:]:
            if node.stage == idx and pos in node.labels:
                pos = minmax(lower_bound, upper_bound, np.rint(self.nodeList[0].model.predict([[key]])[0][0]))
                print(f"Model in stage{idx} predicted that the requested key is inserted in position {pos}")
                idx += 1
            if idx > self.stageIdx[-1]:
                self.node = node
                # 최종 리프노드 선정
                while True:
                    if pos < self.node.labels[0]:
                        # print(f"pos:{pos}, node.labels[0]:{node.labels[0]} 이므로 이전 노드로 이동합니다.")
                        self.node = self.nodeList[self.nodeList.index(self.node) - 1]
                        # print(f"이전 노드의 범위는 {self.node.labels[0]}부터 {self.node.labels[-1]}입니다.")
                    elif pos > self.node.labels[-1]:
                        # print(f"pos:{pos}, node.labels[-1]:{node.labels[-1]} 이므로 다음 노드로 이동합니다.")
                        self.node = self.nodeList[self.nodeList.index(self.node) + 1]
                        # print(f"다음 노드의 범위는 {self.node.labels[0]}부터 {self.node.labels[-1]}입니다.")
                    else:
                        break
                break

        l, r = 0, 0  # binary search를 위한 range
        i = 0
        while self.node.index[pos] != key:
            # exponential search
            if self.node.index[pos] < key:
                l = int(pos + 1)
                pos += 2 ** i
                if pos > (len(self.root.index) - 1):
                    pos = len(self.root.index) - 1
                if pos > self.node.labels[-1]:
                    # print(f"pos:{pos}, node.labels[-1]:{node.labels[-1]} 이므로 다음 노드로 이동합니다.")
                    self.node = self.nodeList[self.nodeList.index(self.node) + 1]
                    # print(f"다음 노드의 범위는 {self.node.labels[0]}부터 {self.node.labels[-1]}입니다.")
                if self.node.index[pos] > key:
                    r = int(pos)
                    if l < self.node.labels[0]:
                        self.node = self.nodeList[self.nodeList.index(self.node) - 1]
                        for chk in range(l, self.node.labels[-1] + 1):
                            if self.node.index[chk] > key:
                                pos = chk
                                print(f"Model recommend {key} is inserted in position {pos}")
                                return pos
                        self.node = self.nodeList[self.nodeList.index(self.node) + 1]
                        for chk in range(self.node.labels[0], r):
                            if self.node.index[chk] > key:
                                pos = chk
                                print(f"Model recommend {key} is inserted in position {pos}")
                                return pos
                    else:
                        for chk in range(l, r):
                            if self.node.index[chk] > key:
                                pos = chk
                                print(f"Model recommend {key} is inserted in position {pos}")
                                return pos
            else:
                r = int(pos)
                pos -= 2 ** i
                if pos < 0:
                    pos = 0
                if pos < self.node.labels[0]:
                    # print(f"pos:{pos}, node.labels[0]:{node.labels[0]} 이므로 이전 노드로 이동합니다.")
                    self.node = self.nodeList[self.nodeList.index(self.node) - 1]
                    # print(f"이전 노드의 범위는 {self.node.labels[0]}부터 {self.node.labels[-1]}입니다.")
                if self.node.index[pos] < key:
                    l = int(pos + 1)
                    if r > self.node.labels[-1]:
                        for chk in range(l, self.node.labels[-1] + 1):
                            if self.node.index[chk] > key:
                                pos = chk
                                print(f"Model recommend {key} is inserted in position {pos}")
                                return pos
                        self.node = self.nodeList[self.nodeList.index(self.node) + 1]
                        for chk in range(self.node.labels[0], r):
                            if self.node.index[chk] > key:
                                pos = chk
                                print(f"Model recommend {key} is inserted in position {pos}")
                                return pos
                    else:
                        for chk in range(l, r):
                            if self.node.index[chk] > key:
                                pos = chk
                                print(f"Model recommend {key} is inserted in position {pos}")
                                return pos
            i += 1
            # 전체 data 범위를 벗어난 값에 대하여 어떻게 insert할 지 구현 필요

    def insert(self, key):
        print("inserting key", key)
        insert_pos = int(self.predict_pos(key))
        self.node.labels = np.append(self.node.labels, self.node.labels[-1]+1)
        self.node.keys = np.insert(self.node.keys, insert_pos%self.node.interval, key)
        LearnedIndexNode.build(self.node)

    # def split_node(self, node):



class LearnedIndexNode:
    def __init__(self, reg, stage, data, offset):
        self.reg = reg
        self.stage = stage
        self.model = None
        self.index = None
        # print(self.data)
        self.labels = np.where(~np.isnan(data))[0] + offset #Create Labels
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

'''
class DataNode:
    def __init__(self, reg, stage, data, idx):
        self.reg = reg
        self.stage = stage
        self.data = create_gapped_array(data)
        self.idx = idx
        self.model = None
        self.index = None
        self.labels = np.where(~np.isnan(self.data) | np.isnan(self.data))[0] + len(self.data)*idx  #Create Labels
        self.keys = self.data[np.where(~np.isnan(self.data) | np.isnan(self.data))[0]]
        self.build()

    def build(self):
        self.index = {}
        train_labels = np.where(~np.isnan(self.data))[0] + len(self.data)*self.idx
        train_keys = self.data[np.where(~np.isnan(self.data))[0]]
        for KEY, POS in zip(self.keys, self.labels):
            self.index[POS] = KEY
        if self.reg == Regression.LINEAR:
            X = train_keys.reshape(-1, 1)
            Y = train_labels.reshape(-1, 1)
            self.model = LinearRegression()
            print(self.index)
            self.model.fit(X, Y)
'''