import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report


mnist = fetch_openml('mnist_784', version=1)

data_shape = mnist.data.shape
num_samples = data_shape[0]
num_features = data_shape[1]


data_info = mnist.DESCR

print("MNIST dataset information:")
print(f"Number of samples: {num_samples}")
print(f"Number of features: {num_features}")
print("\nStructure of the data:")
print(data_info)

log_file_path = "mnist_dataset_log.txt"
with open(log_file_path, 'w') as log_file:
    log_file.write("MNIST dataset information:\n")
    log_file.write(f"Number of samples: {num_samples}\n")
    log_file.write(f"Number of features: {num_features}\n\n")
    log_file.write("Structure of the data:\n")
    log_file.write(data_info)

print(f"Information saved in {log_file_path}")

# few sample digits
fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(10, 6))

for i, ax in enumerate(axes.flat):
    ax.imshow(mnist.data.iloc[i].values.reshape(28, 28), cmap='binary')
    ax.set_title(f"Label: {mnist.target[i]}")
    ax.axis('off')

plt.tight_layout()
plt.show()

X_train, X_test, y_train, y_test = train_test_split(mnist.data, mnist.target, test_size=0.2, random_state=42)


naive_bayes = MultinomialNB()
naive_bayes.fit(X_train, y_train)

from sklearn.metrics import confusion_matrix
import seaborn as sns


y_pred = naive_bayes.predict(X_test)


conf_mat = confusion_matrix(y_test, y_pred)

# Confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

print(classification_report(y_test, y_pred))
