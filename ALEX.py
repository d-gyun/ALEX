import math
import numpy as np
from helpers import minmax
from sklearn.linear_model import LinearRegression

leafNodeList = []

# EnhancedALEX.py

def get_cdf_based_splits(data, threshold=10000):
    if len(data) == 0:
        return [0, len(data)]

    cdf = np.arange(1, len(data) + 1) / len(data)

    # 데이터 내에서 동일한 값을 제거하여 분할할 수 있도록 함
    unique_data, unique_indices = np.unique(data, return_index=True)

    if len(unique_data) < 2:
        return [0, len(data)]

    # Check if indices are within bounds
    max_index = len(cdf) - 1
    valid_indices = unique_indices[unique_indices <= max_index]

    # 1차 도함수 (기울기) 계산
    cdf_gradients = np.gradient(cdf[valid_indices], unique_data[:len(valid_indices)])

    # 기울기 변화율 계산 (1차 도함수의 절대값)
    gradient_changes = np.abs(np.diff(cdf_gradients))

    # 기울기 변화율이 특정 임계값 이상인 지점 찾기
    change_points = np.where(gradient_changes > threshold)[0]

    # 인접한 change_points 병합
    merged_change_points = []
    if len(change_points) > 0:
        prev_cp = change_points[0]
        merged_change_points.append(prev_cp)
        for cp in change_points[1:]:
            if cp - prev_cp > 1:
                merged_change_points.append(cp)
            prev_cp = cp

    split_points = [0] + list(merged_change_points) + [len(data)]
    return split_points

class RMI:
    def __init__(self, data):
        self.leafNodeSize = 100  # Maximum size of leaf node
        self.data = np.sort(data)
        self.root = LearnedIndexNode(self.data, 0, 0, False, None)  # Initialize root node
        self.dataNode = None
        self.split_cnt = 0

        self.root.split_cdf_based(self.leafNodeSize)

    def find(self, key):
        def search_node(node, key):
            if node.is_leaf:
                error = 0
                pred_index = int(node.model.predict([[key]])[0])
                local_pred_pos = minmax(0, len(node.data) - 1, pred_index)

                if node.data[local_pred_pos] == key:
                    print(f"Found {key} in position {node.offset + local_pred_pos} after making {error + 1} checks")
                    return node.offset + local_pred_pos, error  # Return position and error count

                # Exponential search around the predicted position
                else:
                    # print(f"local_pred_pos is {local_pred_pos}")
                    return self.exponential_search(node, local_pred_pos, key, error)
            else:
                pred_index = minmax(0, len(node.data) - 1, int(node.model.predict([[key]])[0]))
                for child in node.children:
                    if child.offset <= pred_index + node.offset < (child.offset + len(child.data)):
                        # print(f"Next node is in position {child.offset} and {child.offset + len(child.data)}")
                        return search_node(child, key)

        return search_node(self.root, key)

    def exponential_search(self, dataNode, pos, key, error):
        l, r = 0, 0  # binary search를 위한 range
        i = 0
        # exponential search
        if dataNode.data[pos] < key:
            while dataNode.data[pos] < key:
                error += 1
                l = int(pos + 1)
                pos += 2 ** i
                r = int(pos)
                if dataNode == leafNodeList[-1] and pos >= len(dataNode.data):
                    pos = len(dataNode.data)-1
                    r = int(pos)
                if pos >= len(dataNode.data):
                    if dataNode.data[-1] == key:
                        print(f"Found {key} in position {dataNode.offset + pos} after making {error + 1} checks")
                        return dataNode.offset + pos, error
                    elif dataNode.data[-1] > key:
                        r = len(dataNode.data)
                        pos = len(dataNode.data)-1
                        break
                    else:
                        dataNode = leafNodeList[leafNodeList.index(dataNode) + 1]
                        pos = 0
                        l = 0
                if dataNode.data[pos] == key:
                    print(f"Found {key} in position {dataNode.offset + pos} after making {error + 1} checks")
                    return dataNode.offset + pos, error
                i += 1

            for chk in range(l, min(r + 1, len(dataNode.data))):
                error += 1
                if dataNode.data[chk] == key:
                    pos = chk
                    print(f"Found {key} in position {dataNode.offset + pos} after making {error + 1} checks")
                    return dataNode.offset + pos, error
        else:
            while dataNode.data[pos] > key:
                error += 1
                r = int(pos)
                pos -= 2 ** i
                l = int(pos)
                if dataNode == leafNodeList[0] and pos < 0:
                    pos = 0
                    l = int(pos) + 1
                if pos < 0:
                    if dataNode.data[0] == key:
                        print(f"Found {key} in position {dataNode.offset + pos} after making {error + 1} checks")
                        return dataNode.offset + pos, error
                    elif dataNode.data[0] < key:
                        l =0
                        pos = 0
                        break
                    else:
                        dataNode = leafNodeList[leafNodeList.index(dataNode) - 1]
                        pos = len(dataNode.data)-1
                        r = len(dataNode.data)
                if dataNode.data[pos] == key:
                    print(f"Found {key} in position {dataNode.offset + pos} after making {error + 1} checks")
                    return dataNode.offset + pos, error
                i += 1

            for chk in range(l, min(r + 1, len(dataNode.data))):
                error += 1
                if dataNode.data[chk] == key:
                    pos = chk
                    print(f"Found {key} in position {dataNode.offset + pos} after making {error + 1} checks")
                    return dataNode.offset + pos, error
        print(f"After making {error + 1} checks I figured that the key {key} doesn't exist!")
        return -1, -1

    def find_all(self, data):
        stats = np.zeros(300)
        for i in range(len(data)):
            pos, errors = self.find(data[i])
            if pos == -1:
                stats[-1] += 1
            else:
                stats[errors] += 1
        return stats

    def insert(self, data):
        self.model_predict(data)

        while data > self.dataNode.data[-1] and leafNodeList.index(self.dataNode) + 1 < len(leafNodeList):
            self.dataNode = leafNodeList[leafNodeList.index(self.dataNode) + 1]
        while data < self.dataNode.data[0] and leafNodeList.index(self.dataNode) - 1 >= 0:
            self.dataNode = leafNodeList[leafNodeList.index(self.dataNode) - 1]

        self.dataNode.data = np.sort(np.append(self.dataNode.data, data))

        # print(len(self.dataNode.data))
        if len(self.dataNode.data) > math.ceil(self.leafNodeSize * (0.8)):
            self.split_cnt += 1
            self.dataNode.split_side()

    def bulk_load(self, data):
        insert_data = np.sort(data)
        for data in insert_data:
            self.insert(data)

    def model_predict(self, key):
        def search_node(node, key):
            if node.is_leaf:
                self.dataNode = node
            else:
                pred_index = minmax(0, len(node.data) - 1, int(node.model.predict([[key]])[0]))
                for child in node.children:
                    if child.offset <= pred_index + node.offset < (child.offset + len(child.data)):
                        return search_node(child, key)

        return search_node(self.root, key)

    def leafNodeCnt(self):
        return len(leafNodeList)

class LearnedIndexNode:
    def __init__(self, data, offset, depth, is_leaf, parent):
        self.data = data
        self.offset = offset
        self.depth = depth
        self.is_leaf = is_leaf
        self.children = []
        self.parent = parent
        self.max_children = 8
        self.model = LinearRegression()

    def train_model(self):
        if len(self.data) > 0:
            X = self.data.reshape(-1, 1)
            y = np.arange(len(self.data))
            self.model.fit(X, y)

    def split_cdf_based(self, leafNodeSize):
        if len(self.data) <= math.ceil(leafNodeSize * (0.8)):
            self.is_leaf = True
            self.train_model()  # 리프 노드에서 바로 모델을 학습
            leafNodeList.append(self)
            return

        split_points = get_cdf_based_splits(self.data)
        num_children = len(split_points) - 1

        if num_children <= 1:
            if len(self.data) > leafNodeSize:
                num_children = min(self.max_children, math.ceil(len(self.data) / math.ceil(leafNodeSize * (0.8))))
                if num_children <= 1:
                    self.is_leaf = True
                    self.train_model()  # 데이터 양이 적을 때 모델 학습
                    leafNodeList.append(self)
                    return

                split_size = len(self.data) // num_children
                for i in range(num_children):
                    start = i * split_size
                    end = (i + 1) * split_size if i < num_children - 1 else len(self.data)
                    child_data = self.data[start:end]
                    if len(child_data) > 0:
                        child = LearnedIndexNode(child_data, self.offset + start, self.depth + 1, False, self)
                        self.children.append(child)
            else:
                self.is_leaf = True
                self.train_model()  # 데이터 양이 적을 때 모델 학습
                leafNodeList.append(self)
                return
        else:
            for i in range(num_children):
                start = split_points[i]
                end = split_points[i + 1]
                child_data = self.data[start:end]
                if len(child_data) > 0:
                    child = LearnedIndexNode(child_data, self.offset + start, self.depth + 1, False, self)
                    self.children.append(child)

        # 내부 노드인 경우 자식 노드 생성 후 모델 학습
        if not self.is_leaf:
            self.train_model()

        for child in self.children:
            child.split_cdf_based(leafNodeSize)

    def split_side(self):
        split_size = len(self.data) // 2
        left = LearnedIndexNode(self.data[0:split_size], self.offset, self.depth, True, self.parent)
        right = LearnedIndexNode(self.data[split_size:len(self.data)], self.offset + split_size, self.depth, True, self.parent)
        left.train_model()
        right.train_model()

        node_pos = leafNodeList.index(self)
        leafNodeList.remove(self)
        leafNodeList.insert(node_pos, left)
        leafNodeList.insert(node_pos+1, right)

        if self.parent is not None:
            node_pos = self.parent.children.index(self)
            self.parent.children.remove(self)
            self.parent.children.insert(node_pos, left)
            self.parent.children.insert(node_pos+1, right)
