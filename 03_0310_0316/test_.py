import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree


# dataset
X_toy, y_toy = datasets.make_blobs(
    n_samples=150, n_features=2, centers=2, cluster_std=5, random_state=2
)

# Plotting
fig = plt.figure(figsize=(10, 8))
plt.plot(X_toy[:, 0][y_toy == 0], X_toy[:, 1][y_toy == 0], "r^")
plt.plot(X_toy[:, 0][y_toy == 1], X_toy[:, 1][y_toy == 1], "bs")
plt.xlabel("X1")
plt.ylabel("X2")
plt.title("Random Classification Data with 2 classes")

# training a classifier is pretty straight-forward
# 1. Define the classifier istance
# 2. Train over a set of examples
# 3. Predict a set of examples (in this case we use the training data, but we can use any kind of data)
# 4. We use an evaluaiton function
clf_knn = KNeighborsClassifier(n_neighbors=5)
clf_knn.fit(X_toy, y_toy)
y_pred = clf_knn.predict(X_toy)
print(accuracy_score(y_true=y_toy, y_pred=y_pred))

# ---

# we first split the training into training + validation set, and then testing set
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X_toy, y_toy, train_size=0.8, random_state=123
)

# we then compute then split training and validation partitions
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, train_size=0.8, random_state=123
)
print(
    f"Original size = {X_toy.shape[0]}\tTrain size = {X_train.shape[0]}\tVal size = {X_val.shape[0]}\tTest size = {X_test.shape[0]}"
)

# ---

# we define two models
clf_knn1 = KNeighborsClassifier(n_neighbors=2)
clf_knn2 = KNeighborsClassifier(n_neighbors=5)

# train the models
clf_knn1.fit(X_train, y_train)
clf_knn2.fit(X_train, y_train)

# we pick the best model by observing training and validation sets
y_train_pred_clf1 = clf_knn1.predict(X_train)
y_val_pred_clf1 = clf_knn1.predict(X_val)

y_train_pred_clf2 = clf_knn2.predict(X_train)
y_val_pred_clf2 = clf_knn2.predict(X_val)

print(f"Clf1 --->\t Train = {accuracy_score(y_train, y_train_pred_clf1):.4f}")
print(f"Clf1 --->\t Val = {accuracy_score(y_val, y_val_pred_clf1):.4f}")

print(f"Clf2 --->\t Train = {accuracy_score(y_train, y_train_pred_clf2):.4f}")
print(f"Clf2 --->\t Val = {accuracy_score(y_val, y_val_pred_clf2):.4f}")

# the best classifier is clf1 (based on the validation score)
# we use clf1 for the testing set
y_test_pred = clf_knn1.predict(X_test)
print(f"\n\nTesting performance = {accuracy_score(y_test, y_test_pred):.4f}")

# ---

# we define the hyper-parameters we want to search
params = {  # the hyper parameters can be found in Sklearn official documentation
    "max_depth": [1, 3, 5],
    "min_samples_split": [2, 3, 4],
}

# we define the target classifier (remember the seed to allow reproducibility)
clf = DecisionTreeClassifier(random_state=123)

# and finally the grid search object
gs = GridSearchCV(estimator=clf, param_grid=params, verbose=1, cv=5, refit=True)

# we can now fit --- remeber that we need to use the train + val partitions
gs.fit(X_train_val, y_train_val)

# the gs returns the best model that is already retrained (if refit = true) on
# the train + val partitiosn
# check the testin performance
y_test_pred = gs.predict(X_test)
accuracy_score(y_test, y_test_pred)

# ---

# ???
# gs.best_params_
plot_tree(gs.best_estimator_)
