import math
import numpy as np
from helpers import minmax
from sklearn.linear_model import LinearRegression

class RMI:
    def __init__(self, data, density):
        self.fanout = 8  # node의 fanout
        self.maxNodeSize = 100
        self.data = np.sort(data)
        self.root = LearnedIndexNode(0, self.data,0)
        self.nodeList = [self.root]
        self.dataNode = None
        self.stageIdx = [0]
        self.nodeSize = len(self.data)
        self.leafLevel = False
        self.node_density = density
        self.split_cnt = 0
        self.build_tree()

    def build_tree(self):
        self.stageIdx.append(self.stageIdx[-1]+1)
        # internal node
        if self.nodeSize // self.fanout > self.maxNodeSize:
            self.interval = len(self.data)//self.fanout**(self.stageIdx[-1])
            self.nodeSize //= self.fanout
        # data node
        else:
            self.interval = math.ceil(self.maxNodeSize*(self.node_density))
            self.leafLevel = True

        for i in range(0, len(self.data)//self.interval + (1 if (len(self.data)%self.interval) else 0)):
            if self.interval*(i+1) < len(self.data):
                Node = LearnedIndexNode(self.stageIdx[-1], self.data[self.interval*i:self.interval*(i+1)], self.interval*i)
            else:
                Node = LearnedIndexNode(self.stageIdx[-1], self.data[self.interval*i:], self.interval*i)
            self.nodeList.append(Node)
        print("RMI STAGE = " + str(self.stageIdx[-1]))

        if self.leafLevel:
            return
        else:
            self.build_tree()

    def find(self, key):
        upper_bound = len(self.root.index) - 1
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
                    self.dataNode = node # 최종 리프노드 선정
                    break

        error = 0
        while True:
            if pos < self.dataNode.labels[0]:
                # print(f"pos:{pos}, node.labels[0]:{node.labels[0]} 이므로 이전 노드로 이동합니다.")
                self.dataNode = self.nodeList[self.nodeList.index(self.dataNode) - 1]
                # error += 1
                # print(f"이전 노드의 범위는 {self.node.labels[0]}부터 {self.node.labels[-1]}입니다.")
            elif pos > self.dataNode.labels[-1]:
                # print(f"pos:{pos}, node.labels[-1]:{node.labels[-1]} 이므로 다음 노드로 이동합니다.")
                self.dataNode = self.nodeList[self.nodeList.index(self.dataNode) + 1]
                # error += 1
                # print(f"다음 노드의 범위는 {self.node.labels[0]}부터 {self.node.labels[-1]}입니다.")
            else:
                break

        if self.dataNode.index[pos] == key:
            print(f"Found {key} in position {pos} after making {error + 1} checks")
            return pos, error
        else:
            return self.exponential_search(self.dataNode, pos, key, error)

    def exponential_search(self, dataNode, pos, key, error):
        l, r = 0, 0  # binary search를 위한 range
        i = 0
        # exponential search
        if dataNode.index[pos] < key:
            while dataNode.index[pos] < key:
                error += 1
                l = int(pos + 1)
                pos += 2 ** i
                r = int(pos)
                if pos > self.nodeList[-1].labels[-1]:
                    pos = self.nodeList[-1].labels[-1]
                    r = int(pos)
                if pos > dataNode.labels[-1]:
                    if dataNode.keys[-1] == key:
                        print(f"Found {key} in position {pos} after making {error + 1} checks")
                        return pos, error
                    elif dataNode.keys[-1] > key:
                        r = dataNode.labels[-1] + 1
                        pos = dataNode.labels[-1]
                        break
                    else:
                        dataNode = self.nodeList[self.nodeList.index(dataNode) + 1]
                        l = dataNode.labels[0]
                if dataNode.index[pos] == key:
                    print(f"Found {key} in position {pos} after making {error + 1} checks")
                    return pos, error
                i += 1

            for chk in range(l, r+1):
                error += 1
                if dataNode.index[chk] == key:
                    pos = chk
                    print(f"Found {key} in position {pos} after making {error + 1} checks")
                    return pos, error
        else:
            while dataNode.index[pos] > key:
                error += 1
                r = int(pos)
                pos -= 2 ** i
                l = int(pos)
                if pos < 0:
                    pos = 0
                    l = int(pos) + 1
                if pos < dataNode.labels[0]:
                    if dataNode.keys[0] == key:
                        print(f"Found {key} in position {pos} after making {error + 1} checks")
                        return pos, error
                    elif dataNode.keys[0] < key:
                        l = dataNode.labels[0]
                        pos = dataNode.labels[0]
                        break
                    else:
                        dataNode = self.nodeList[self.nodeList.index(dataNode) - 1]
                        r = dataNode.labels[-1]
                if dataNode.index[pos] == key:
                    print(f"Found {key} in position {pos} after making {error + 1} checks")
                    return pos, error
                i += 1

            for chk in range(l, r+1):
                error += 1
                if dataNode.index[chk] == key:
                    pos = chk
                    print(f"Found {key} in position {pos} after making {error + 1} checks")
                    return pos, error
        print(f"After making {error + 1} checks I figured that the key {key} doesn't exist!")
        return -1, -1

    def find_all(self, data):
        stats = np.zeros(100)
        for i in range(len(data)):
            pos, err = self.find(data[i])
            if pos == -1:
                stats[-1] += 1
            stats[err] += 1
        return stats

    def split(self, split_node):
        for i, node in enumerate(self.nodeList):
            if node.stage == self.stageIdx[-1] and split_node.offset == node.offset:
                left = LearnedIndexNode(node.stage, node.keys[:len(node.keys)//2], node.labels[0])
                right = LearnedIndexNode(node.stage, node.keys[len(node.keys)//2:], node.labels[len(node.keys)//2])
                self.nodeList.remove(node)
                self.nodeList.insert(i, left)
                self.nodeList.insert(i+1, right)

                # for j, parent_node in enumerate(self.nodeList):
                #     if parent_node.stage == self.stageIdx[-1]-1 and parent_node.labels[0] <= node.offset <= parent_node.labels[-1]:
                #         self.nodeList.remove(parent_node)
                #         parent_node = LearnedIndexNode(parent_node.stage, , parent_node.labels[0])
                #         self.nodeList.insert(j, parent_node)
                break

    def insert(self, data):
        self.model_predict(data)

        while data > self.dataNode.keys[-1] and self.nodeList.index(self.dataNode) + 1 < len(self.nodeList):
            self.dataNode = self.nodeList[self.nodeList.index(self.dataNode) + 1]
        while data < self.dataNode.keys[0] and self.nodeList.index(self.dataNode) - 1 >= 0:
            self.dataNode = self.nodeList[self.nodeList.index(self.dataNode) - 1]

        self.dataNode.labels = np.append(self.dataNode.labels, self.dataNode.labels[-1] + 1)
        self.dataNode.keys = np.sort(np.append(self.dataNode.keys, data))

        for KEY, POS in zip(self.dataNode.keys, self.dataNode.labels):
            self.dataNode.index[POS] = KEY

        if len(self.dataNode.keys) > self.maxNodeSize * 0.8:
            self.split_cnt += 1
            self.split(self.dataNode)

    def bulk_load(self, data):
        insert_data = np.sort(data)
        for data in insert_data:
            self.insert(data)

    def model_predict(self, key):
        upper_bound = len(self.root.index) - 1
        lower_bound = 0.0
        pos = minmax(lower_bound, upper_bound, np.rint(self.nodeList[0].model.predict([[key]])[0][0]))

        idx = 1
        for node in self.nodeList[1:]:
            if node.stage == idx and pos in node.labels:
                pos = minmax(lower_bound, upper_bound, np.rint(node.model.predict([[key]])[0][0]))
                idx += 1
                if idx > self.stageIdx[-1]:
                    self.dataNode = node  # 최종 리프노드 선정
                    break

    def leafNodeCnt(self):
        cnt = 0
        leafnodestage = self.stageIdx[-1]
        for node in self.nodeList:
            if node.stage == leafnodestage:
                cnt += 1
        return cnt

class LearnedIndexNode:
    def __init__(self, stage, data, offset):
        self.stage = stage
        self.model = None
        self.index = None
        self.offset = offset
        self.labels = np.where(~np.isnan(data))[0] + offset #Create Labels
        self.keys = data[np.where(~np.isnan(data))[0]]
        self.build()

    def build(self):
        self.index = {}
        for KEY, POS in zip(self.keys, self.labels):
            self.index[POS] = KEY

        X = self.keys.reshape(-1, 1)
        Y = self.labels.reshape(-1, 1)
        self.model = LinearRegression()
        # print(self.index)
        self.model.fit(X, Y)
