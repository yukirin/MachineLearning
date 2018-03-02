from math import log2
import numpy as np
from collections import Counter, deque
from sklearn.datasets import load_iris
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


def cross_entropy(datanum):
    l = len(datanum)
    c = Counter(datanum)

    total = 0
    for v in c.values():
        p = v / l
        total += p * log2(p)

    return -total


class Node:
    def __init__(self, datanum):
        self.datanum = datanum
        self.l = len(datanum)
        self.left = self.right = None
        self.split_value = None
        self.split_index = None
        self.depth = 0


class DecisionTree:
    def __init__(self, datanum, ans, min_size=4, max_depth=2, rand_features=2):
        self.min_size = min_size
        self.max_depth = max_depth
        self.rand_features = rand_features
        self.num_features = len(datanum[0])
        self.root = Node(
            np.concatenate([datanum, np.reshape(ans, (len(ans), 1))], axis=1))

    def predict(self, data):
        q = [self.root]
        while q:
            node = q.pop()

            if node.left is None:
                c = Counter(node.datanum[:, -1])
                return c.most_common(1)[0][0]

            if data[node.split_index] < node.split_value:
                q.append(node.left)
            else:
                q.append(node.right)

    def prune_check(self, node):
        if node.l < self.min_size:
            return True

        if node.depth > self.max_depth:
            return True
        return False

    def fit(self):
        q = deque([self.root])
        while q:
            node = q.popleft()

            if self.prune_check(node):
                continue

            result = self.division(node)

            if not result:
                continue

            node.left, node.right, node.split_value, node.split_index = result
            node.left.depth = node.right.depth = node.depth + 1

            q.append(node.left)
            q.append(node.right)

    def division(self, node):
        min_entropy = 9e10
        min_left = min_right = None
        min_split_value = None
        min_split_index = None

        if len(set(node.datanum[:, -1])) == 1:
            return ()

        for f_index in shuffle(range(self.num_features))[:self.rand_features]:
            for d in node.datanum:
                split_value = d[f_index]
                left = np.array(
                    [p for p in node.datanum if p[f_index] < split_value])
                right = np.array(
                    [p for p in node.datanum if p[f_index] >= split_value])

                if len(left) == 0 or len(right) == 0:
                    continue

                entropy = (len(left) / node.l) * cross_entropy(left[:, -1]) + (
                    len(right) / node.l) * cross_entropy(right[:, -1])

                if entropy < min_entropy:
                    min_entropy = entropy
                    min_left, min_right = left, right
                    min_split_value = split_value
                    min_split_index = f_index

        if not min_split_value:
            return ()

        return Node(min_left), Node(
            min_right), min_split_value, min_split_index


if '__main__' == __name__:
    iris = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(
        iris.data, iris.target, stratify=iris.target, random_state=10)

    tree = DecisionTree(x_train, y_train)
    tree.fit()

    total = 0
    for d, ans in zip(x_test, y_test):
        total += int(tree.predict(d) == ans)

    print(total / len(x_test))