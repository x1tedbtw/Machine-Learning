from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

data = load_iris()

X = data.data #features
y = data.target # class label

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=23)

clf = DecisionTreeClassifier()

clf.fit(X_train, y_train)

train_pred = clf.predict(X_train)
test_pred = clf.predict(X_test)

train_accuracy = accuracy_score(y_train, train_pred)
test_accuracy = accuracy_score(y_test, test_pred)
print("Accuracy for train data:", train_accuracy)
print("Accuracy for test data:", test_accuracy)

# The result for the train data shows high accuracy, that means it learned patterns of the data pretty good.
# However, it might be different with other data.

random_state_values = range(1,10)
test_accuracies = []

for random_state_value in random_state_values:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=random_state_value)
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    test_predictions = clf.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_predictions)
    test_accuracies.append(test_accuracy)

# print(test_accuracies)

mean_accuracy = np.mean(test_accuracies)
std_deviation = np.std(test_accuracies)
print("Mean accuracy:", mean_accuracy)
print("Standard deviation:", std_deviation)

split_ratios = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
accuracy_scores = []

for ratio in split_ratios:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ratio, random_state=42)
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    test_predictions = clf.predict(X_test)
    accuracy = accuracy_score(y_test, test_predictions)
    accuracy_scores.append(accuracy)

plt.plot(split_ratios, accuracy_scores, marker='o')
plt.title('Accuracy Score vs. Split Ratio')
plt.xlabel('Split Ratio')
plt.ylabel('Accuracy Score')
plt.grid(True)
plt.show()

# My model
iris = load_iris()
X = iris.data[:, [0, 2]]
y = iris.target


feature_names = ['sepal_length', 'petal_length']
class_names = iris.target_names

clf = DecisionTreeClassifier()
clf.fit(X, y)

plt.figure(figsize=(20, 10))
plot_tree(clf, filled=True, feature_names=feature_names, class_names=class_names)
plt.show()