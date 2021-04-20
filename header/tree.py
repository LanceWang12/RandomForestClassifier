import numpy as np
from header.unit import Node
from header.speed_up import compute_class_weight, split

class DecisionTreeClassifier:
    def __init__(
        self, criterion='gini', splitter='best', max_depth=None, 
        min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, 
        max_features=None, max_leaf_nodes=None, min_impurity_decrease=0.0, 
        min_impurity_split=0, class_weight=None, ccp_alpha=0.0
        ):
        self.criterion, self.splitter = criterion, splitter
        self.max_depth, self.min_samples_split, self.min_samples_leaf = max_depth, min_samples_split, min_samples_leaf
        self.min_weight_fraction, self.max_features, self.max_leaf_nodes = min_weight_fraction_leaf, max_features, max_leaf_nodes
        self.min_impurity_decrease, self.min_impurity_split = min_impurity_decrease, min_impurity_split
        self.class_weight, self.ccp_alpha = class_weight, ccp_alpha

        self.tree_ = None

    def fit(self, x, y):
        x, y = np.array(x), np.array(y)

        # The number of features
        self.n_features_ = x.shape[1]

        # The number of classes
        self.n_classes_ = np.unique(y).shape[0]

        # set the class weight
        self.__cw = compute_class_weight(self.class_weight, self.n_classes_, y)

        # The structure of decision tree
        self.tree_ = self.__build_tree(x, y)

        # Let process return the tree
        

        return self

    def predict(self, X):
        if self.tree_ is None:
            raise NotImplementedError('Please fit the model first!!')
        return np.array([self.__predict(inputs) for inputs in X])

    def __predict(self, inputs):
        current = self.tree_

        # visit all nodes
        while current.left:
            if inputs[current.feature_index] >= current.decision_threshold:
                # if the feature >= threshold -> go to right way
                current = current.right
            else:
                current = current.left
                
        return current.node_class


    def __split(self, x, y):
        return split(
            x, y, self.min_samples_split, self.n_classes_, self.n_features_,
            self.max_features, self.__cw, self.min_impurity_decrease,
            self.min_impurity_split
        )

        

    def __build_tree(self, X, y, depth = 0):
        num_samples_per_class = np.array([np.sum(y == i) for i in range(self.n_classes_)])

        # The nodel class will belong to the class which have the most data in this node
        node = Node(node_class = np.argmax(num_samples_per_class))

        if depth < self.max_depth:
            idx, thr = self.__split(X, y)
            if idx is not None:
                indices_right = X[:, idx] >= thr
                # get data belong to right and left
                x_right, y_right = X[indices_right], y[indices_right]
                x_left, y_left = X[~indices_right], y[~indices_right]
                node.set_idx_thrs(feature_index = idx, decision_threshold = thr)
                
                # recursivees_right], y[~indices_right]

                node.left = self.__build_tree(x_left, y_left, depth + 1)
                node.right = self.__build_tree(x_right, y_right, depth + 1)
        return node