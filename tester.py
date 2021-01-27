from MachineLearningAlgorithms.Classification.DecisionTree.DecisionTree import DecisionTree
from sklearn.datasets import load_iris
from sklearn import tree


dt = DecisionTree(None, None, None, None, None, None, None, None, None, None, None, None, None)
X, y = load_iris(return_X_y = True)
model = dt.fit(X, y, None, None)
print(model)
tree.plot_tree(model)
tree.export_graphviz(model,
                     out_file="tree.dot",
                     filled = True)