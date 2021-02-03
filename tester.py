from MachineLearningAlgorithms.Classification.DecisionTree.DecisionTree import DecisionTree
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import metrics

dt = DecisionTree()
X, y = load_iris(return_X_y = True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
model = dt.fit(X_train, y_train, None, None)
y_pred = model.predict(X_test)
print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))