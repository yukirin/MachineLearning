from collections import Counter
from decisiontree import DecisionTree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


class RandomForest:
    def __init__(self, num_tree=6, sample_data_rate=0.7, sample_features=2):
        self.num_tree = num_tree
        self.sample_data_rate = sample_data_rate
        self.sample_features = sample_features
        self.trees = []

    def fit(self, datanum, ans):
        for _ in range(self.num_tree):
            x_train, _, y_train, _ = train_test_split(
                datanum, ans, test_size=1.0 - self.sample_data_rate)

            tree = DecisionTree(
                x_train, y_train, rand_features=self.sample_features)
            tree.fit()

            self.trees.append(tree)

    def predict(self, data):
        result = [tree.predict(data) for tree in self.trees]
        return Counter(result).most_common(1)[0][0]


if '__main__' == __name__:
    iris = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(
        iris.data, iris.target, stratify=iris.target, random_state=10)

    forest = RandomForest()
    forest.fit(x_train, y_train)

    total = 0
    for d, ans in zip(x_test, y_test):
        total += int(forest.predict(d) == ans)

    print(total / len(x_test))