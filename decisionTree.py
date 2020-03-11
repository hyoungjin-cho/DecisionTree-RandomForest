import numpy as np
class DecisionTree:

    class Node:
        def __init__(self, left, right, split_rule, label, depth):
            self.left, self.right = left, right
            self.split_rule = split_rule
            self.label = label
            self.depth = depth

    def __init__(self, max_depth, num_features = 'all'):
        """
        num_features = 'all' or 'sqrt'
        """
        self.max_depth = max_depth
        self.num_features = num_features

    @staticmethod
    def entropy(labels):
        values, counts = np.unique(labels, return_counts=True)
        if len(values) == 1:
            return 0
        priors = counts / float(len(labels))
        return -np.dot(priors, np.log2(priors))

    @staticmethod
    def weighted_avg_entropy(left_labels, right_labels, ):
        left_entropy = DecisionTree.entropy(left_labels)
        right_entropy = DecisionTree.entropy(right_labels)
        return (len(left_labels)*left_entropy + len(right_labels)*right_entropy) / float(len(left_labels) + len(right_labels))

    @staticmethod
    def majority_label(labels):
        values, counts = np.unique(labels, return_counts = True)
        return values[np.argmax(counts)]

    def split(self, S, labels, feature, threshold):
        mask = S[:, feature] <= threshold
        S_left, S_left_labels = S[mask,:], labels[mask]
        S_right, S_right_labels = S[~mask,:], labels[~mask]
        return S_left, S_right, S_left_labels, S_right_labels

    def segmenter(self, S, labels):
        def find_best_threshold_per_feature(feature):
            thresholds = sorted(set(S[:, feature]))
            if len(thresholds) == 1:
                _, _, S_left_labels, S_right_labels = self.split(S, labels, feature, thresholds)
                return S[0,feature], DecisionTree.weighted_avg_entropy(S_left_labels, S_right_labels)
            if len(thresholds) == 2: # Optimization for binary feature.
                th = sum(thresholds) / 2
                _, _, S_left_labels, S_right_labels = self.split(S, labels, feature, th)
                weighted_avg_entropy = DecisionTree.weighted_avg_entropy(S_left_labels, S_right_labels)
                return th, weighted_avg_entropy
            thresholds = np.add(thresholds[:-1], thresholds[1:]) / 2
            best_threshold = thresholds[0]
            min_weighted_avg_entropy = float("inf")
            for threshold in thresholds:
                S_left, S_right, S_left_labels, S_right_labels = self.split(S, labels, feature, threshold)
                current_weighted_avg_entropy = self.weighted_avg_entropy(S_left_labels, S_right_labels)
                if current_weighted_avg_entropy < min_weighted_avg_entropy:
                    best_threshold = threshold
                    min_weighted_avg_entropy = current_weighted_avg_entropy
            return best_threshold, min_weighted_avg_entropy
        best_threshold = None
        best_feature = None
        min_weighted_avg_entropy = float("inf")
        random.seed(seed)
        features = np.random.choice(S.shape[1], self.num_features, replace = False)
        for feature in features:
            current_threshold, current_weighted_avg_entropy = find_best_threshold_per_feature(feature)
            if current_weighted_avg_entropy < min_weighted_avg_entropy:
                min_weighted_avg_entropy = current_weighted_avg_entropy
                best_threshold = current_threshold
                best_feature = feature
        return best_feature, best_threshold

    def grow_tree(self, S, labels, depth):
        if len(set(labels)) == 1 or depth >= self.max_depth:
            return self.Node(left = None, right = None, split_rule = None, label = self.majority_label(labels), depth = depth)
        else:
            best_feature, best_threshold = self.segmenter(S, labels)
            S_left, S_right, S_left_labels, S_right_labels = self.split(S, labels, best_feature, best_threshold)
            if len(S_left) == 0 or len(S_right) == 0:
                return self.Node(left = None, right = None, split_rule = None, label = self.majority_label(labels), depth = depth)
            return self.Node(left = self.grow_tree(S_left, S_left_labels, depth+1), right = self.grow_tree(S_right, S_right_labels, depth+1), \
                            split_rule = (best_feature, best_threshold), label = None, depth = depth)

    def train(self, data, labels, feature_names = None):
        assert len(data) == len(labels), "len(data) must be equal to len(labels)."
        if not feature_names:
            feature_names = ["feature " + str(i) for i in range(data.shape[1])]
        self.feature_names = feature_names
        if self.num_features == 'sqrt':
            self.num_features = int(np.sqrt(data.shape[1]))
        else:
            self.num_features = data.shape[1]
        self.root = self.grow_tree(data, labels, 0)
        return self

    def predict(self, data, showPath = False):
        if data.ndim == 1:
            data = data.reshape((1, len(data)))
        prediction = []
        for i in range(len(data)):
            current_node = self.root
            while current_node.label == None:
                feature_idx, threshold = current_node.split_rule
                if data[i, feature_idx] <= threshold:
                    if showPath:
                        print(self.feature_names[feature_idx] + " <= " + str(threshold))
                    current_node = current_node.left
                else:
                    if showPath:
                        print(self.feature_names[feature_idx] + " > " + str(threshold))
                    current_node = current_node.right
            prediction.append(current_node.label)
        if showPath:
            print("label: ", current_node.label)
        return prediction

    def accuracy(self, prediction, true_label):
        assert len(prediction) == len(true_label), "len(prediction) must be equal to len(true_label)"
        return np.sum(prediction == true_label) / len(prediction)

    def showTree(self):
        def showTree_helper(node):
            padding = "\n" + ("\t" * node.depth)
            if node.label != None: # leaf node.
                return padding + "leaf node: " + str(node.label)
            return padding + "("+ str(self.feature_names[node.split_rule[0]]) + ", " + str(node.split_rule[1]) + ")" + showTree_helper(node.left) + showTree_helper(node.right)
        print(showTree_helper(self.root))
