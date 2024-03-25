import numpy as np
from helpers import minmax
from enum import Enum
from sklearn.linear_model import LinearRegression

class RMI:
    def __init__(self, data):
        self.fanout = 8  # node의 fanout(공유자식트리에서 node의 fanout은 의미가 없지만 각 레벨별 노드 수의 비율이라고 선언)
        self.leafNodeSize = 100 # leafNode(dataNode)의 max node size
        self.data = np.array(data)
        self.data = np.sort(self.data)
        self.root = LearnedIndexNode(0, self.data, 0)
        self.nodeList = [self.root]
        self.stageIdx = [0]

        self.nodeSize = len(self.data)
        print("Start RMI")
        while self.nodeSize > self.leafNodeSize:
            self.stageIdx.append(self.stageIdx[-1]+1)
            self.interval = len(self.data)//self.fanout**(self.stageIdx[-1]) # 각 node의 size
            for i in range(0, len(self.data)//self.interval + (1 if (len(self.data)%self.interval) else 0)):
                if self.interval*(i+1) < len(self.data): # internal node
                    childNode = LearnedIndexNode(self.stageIdx[-1], self.data[self.interval*i:self.interval*(i+1)], self.interval*i)
                else: #data node
                    childNode = LearnedIndexNode(self.stageIdx[-1], self.data[self.interval*i:], self.interval*i)
                self.nodeList.append(childNode)
            self.nodeSize //= self.fanout
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
        # RMI 탐색을 통한 data node 선정
        for node in self.nodeList[1:]:
            if node.stage == idx and pos in node.labels:
                pos = minmax(lower_bound, upper_bound, np.rint(node.model.predict([[key]])[0][0]))
                # print(f"Model in stage{idx} predicted that the requested key is in position {pos}")
                idx += 1
            if idx > self.stageIdx[-1]:
                self.node = node
                break

        error = 0
        # 최종 리프노드 선정
        while True:
            if pos < self.node.labels[0]:
                # print(f"pos:{pos}, node.labels[0]:{node.labels[0]} 이므로 이전 노드로 이동합니다.")
                self.node = self.nodeList[self.nodeList.index(self.node) - 1]
                error += 1
                # print(f"이전 노드의 범위는 {self.node.labels[0]}부터 {self.node.labels[-1]}입니다.")
            elif pos > self.node.labels[-1]:
                # print(f"pos:{pos}, node.labels[-1]:{node.labels[-1]} 이므로 다음 노드로 이동합니다.")
                self.node = self.nodeList[self.nodeList.index(self.node) + 1]
                error += 1
                # print(f"다음 노드의 범위는 {self.node.labels[0]}부터 {self.node.labels[-1]}입니다.")
            else:
                break
        # Real Pos 탐색
        while self.node.index[pos] != key:
            error += 1
            pos += 1 if self.node.index[pos] < key else -1

            if pos < self.node.labels[0]:
                # print(f"pos:{pos}, node.labels[0]:{node.labels[0]} 이므로 이전 노드로 이동합니다.")
                self.node = self.nodeList[self.nodeList.index(self.node)-1]
                # print(f"이전 노드의 범위는 {self.node.labels[0]}부터 {self.node.labels[-1]}입니다.")
            elif pos > self.node.labels[-1]:
                # print(f"pos:{pos}, node.labels[-1]:{node.labels[-1]} 이므로 다음 노드로 이동합니다.")
                self.node = self.nodeList[self.nodeList.index(self.node)+1]
                # print(f"다음 노드의 범위는 {self.node.labels[0]}부터 {self.node.labels[-1]}입니다.")

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
    def __init__(self, stage, data, offset):
        self.stage = stage
        self.model = None
        self.index = None
        self.labels = np.where(~np.isnan(data))[0] + offset  #Create Labels
        self.keys = data[np.where(~np.isnan(data))[0]]
        self.build()

    def build(self):
        self.index = {}

        for KEY, POS in zip(self.keys, self.labels):
            self.index[POS] = KEY

        X = self.keys.reshape(-1, 1)
        Y = self.labels.reshape(-1, 1)
        self.model = LinearRegression()
        print(self.index)
        self.model.fit(X, Y)