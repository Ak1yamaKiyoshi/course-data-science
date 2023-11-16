import numpy as np
# To make sure the target variable is ordinally encoded, use sklearn preprocessing
    # Can be commented out so algorithm functionality still works if y data is already in correct format
import warnings


class Node:
    def __init__(self, predicted_class, depth=0):
        # metadata
        self.depth = depth
        self.predicted_class = predicted_class
        self.threshold = None
        self.feature_index = 0

        # child nodes
        self.left = None
        self.right = None

        # position relative to parent node
        self.is_left = False
        self.is_right= True


class DecisionTreeClassifier():
    def __init__(self, max_depth=None):
        self.tree = None
        self.max_depth = max_depth

    def split(self, X, y):
        """[Helper function to locate the ideal feature and threshold to split on. Called within grow_tree()]

        Parameters
        ----------
        X : [NumPy Array]
            [Training data, rows of features values]
        y : [NumPy Array]
            [Training target data, class values of each row]

        Returns
        -------
        ideal_col: [int]
                     [Column index of ideal feature to split on]
        ideal_threshold: [int]
                         [Ideal threshold value of the best feature to split on]
        """
        ideal_col = None
        ideal_threshold = None

        num_observations = y.size
        y = y.reshape(num_observations,)

        # calculate count for each class
        count_in_parent = [np.count_nonzero(y == c) for c in range(self.num_classes)]

        # calculate gini impurity for parent node
        best_gini = self.gini(num_observations, count_in_parent)

        # right shape for concat
        temp_y = y.reshape(y.reshape[0], 1)

        # Loop trough all columns
        # Sort X and y by values in column
        for col in range(self.num_features):
            # right shape for concat
            temp_X = X[:, col].reshape(num_observations, 1)
            all_data = np.concatenate((temp_X, temp_y), axis=1)
            sorted_data = all_data[all_data[:, 0].argsort()]
            # split data back in X and y ( now sorted )
            thresholds, obs_classes = np.array.split(sorted_data, 2, axis=1)
            obs_classes = obs_classes.astype(int)

            num_left = [0] * self.num_classes
            num_right = count_in_parent.copy()

            # Loop trough all observations to find best split
            for i in range(1, num_observations):
                class_ = obs_classes[i - 1][0]
                num_left[class_] += 1
                num_right[class_] -= 1
                gini_left = 1.0 - sum((num_left[x] / i) ** 2 for x in range(self.num_classes))
                gini_right = 1.0 - sum((num_right[x] / (num_observations - i)) ** 2 for x in range(self.num_classes))
                gini = (i * gini_left + (num_observations - i) * gini_right) / num_observations

                # avoid split when values are equal
                if (thresholds[i][0] == thresholds[i - 1][0]):
                    continue

                if gini < best_gini:
                    best_gini = gini
                    ideal_col = col
                    ideal_threshold = (thresholds[i][0] + thresholds[i - 1][0]) / 2
        return ideal_col, ideal_threshold

    def grow_tree(self, X, y, depth=0):
        """[Grow tree function to continue adding splits in the tree if depth < max_depth]

        Parameters
        ----------
        X : [np.Array]
            [X/feature values of current node]
        y : [np.Array]
            [y/target values of current node]
        depth : int, optional
            [depth of current node], by default 0

        Returns
        -------
        [Node]
            [Root Node of the DT]
        """
        pop_per_class = [np.count_nonzero(y==i) for i in range(self.num_classes)]

        predicted_class = np.argmax(pop_per_class)
        node = Node(predicted_class, depth)
        node.samples = y.size

        # if depth >= max_depth leave
        if depth >= self.max_depth:
            return node

        col, threshold = self.split(X, y)
        if col and threshold:
            indicies_left =     X[:, col] < threshold
            X_left, y_left =    X[indicies_left], y[indicies_left]
            indicies_right =    X[:, col] >= threshold
            X_right, y_right =  X[indicies_right], y[indicies_right]
            node.feature_index = col
            node.threshold = threshold
            node.left = self.grow_tree(X_left, y_left, depth + 1)
            node.left.is_left = True
            node.right = self.grow_tree(X_right, y_right, depth + 1)
            node.right.is_right = True
        return node


    def gini(self, obs, iterator):
        return 1.0 - sum((n / obs) ** 2 for n in iterator)

    def fit(self, X, y):
        """[Function to fit a decision tree classifier]

        Parameters
        ----------
        X : [NumPy Array]
            [rows of feature values]
        y : [NumPy Array]
            [target class value for each row]

        Attributes (of DecisionTreeClassifier, defined within fit function )
        ----------
            [Num Classes]
                Number of unique classes in the target
            [Num Features]
                Number of features in the data(X)
            [Tree]
                The decision tree
        """
        self.num_classes = len(np.unique(y))
        self.num_features = X.shape[1]
        self.tree = self.grow_tree(X, y)

    def predict(self, X_test):
        """[Function to predict using new X input(s)]

        Parameters
        ----------
        X_test : [np.Array (shape of (number of rows/observations,X_train.shape[1])) or Python list/list of lists]
            [X/feature test data]

        Returns
        -------
        [np.Array of shape (observations, )]
            [Predicted target class(es) of the test data]
        """
        node = self.tree
        predictions = []
        for obs in X_test:
            # Have to reassign node to root node or else will make same predictions
            node = self.tree
            while node.left:
                if obs[node.feature_index] < node.threshold:
                    node = node.left
                else:
                    node = node.right
            predictions.append(node.predicted_class)
        return np.array(predictions)