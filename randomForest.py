import decisionTree
import import multiprocessing
from scipy import stats
class RandomForest():
    def __init__(self, num_trees = 1, num_samples = None, num_features = 'sqrt', max_depth = 5):
        self.num_trees = num_trees
        self.num_samples = num_samples
        self.num_features = num_features
        self.max_depth = max_depth
        self.trees = [DecisionTree(max_depth, num_features)] * num_trees

    def train(self, data, labels, feature_names = None, num_CPUs = 1):
        assert num_CPUs <= multiprocessing.cpu_count(), "you have " + str(multiprocessing.cpu_count()) + " cpus."
        multiple_results = []
        pool = multiprocessing.Pool(processes = num_CPUs)
        for tree in self.trees:
            random.seed(seed)
            ran_samples = np.random.choice(len(data), self.num_samples, replace = True)
            multiple_results.append(pool.apply_async(tree.train, args=(data[ran_samples,:], labels[ran_samples], feature_names)))
        self.trees = [result.get() for result in multiple_results]
        pool.close()

    def predict(self, data, showPath = False):
        predictions = []
        for i in range(len(self.trees)):
            if showPath:
                print("tree " + str(i))
            predictions.append(self.trees[i].predict(data, showPath))
        predictions = np.array(predictions)
        results = []
        for i in range(len(data)):
            results.append(DecisionTree.majority_label(predictions[:,i]))
        return results

    def accuracy(self, prediction, true_label):
        assert len(prediction) == len(true_label), "len(prediction) must be equal to len(true_label)"
        return np.sum(prediction == true_label) / len(prediction)
