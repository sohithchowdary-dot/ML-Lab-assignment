import math
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial.distance import minkowski

def dot_product(vector_a, vector_b):
    result = 0
    for i in range(len(vector_a)):
        result += vector_a[i] * vector_b[i]
    return result

def vector_length(vector):
    total = 0
    for value in vector:
        total += value * value
    return math.sqrt(total)

def mean(data):
    total = 0
    for value in data:
        total += value
    return total / len(data)


def variance(data):
    m = mean(data)
    total = 0
    for value in data:
        total += (value - m) ** 2
    return total / len(data)

def class_center(class_samples):
    return np.mean(class_samples, axis=0)


def euclidean_distance(point_a, point_b):
    return vector_length(point_a - point_b)


def minkowski_distance(point_a, point_b, p):
    total = 0
    for i in range(len(point_a)):
        total += abs(point_a[i] - point_b[i]) ** p
    return total ** (1 / p)

def split_dataset(features, labels):
    return train_test_split(features, labels, test_size=0.3, random_state=1)


def train_knn_model(train_X, train_y, k):
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(train_X, train_y)
    return knn_model


def calculate_accuracy(model, test_X, test_y):
    return model.score(test_X, test_y)


def make_prediction(model, test_X):
    return model.predict(test_X)



def custom_knn(train_X, train_y, test_sample, k):
    distances = []

    for i in range(len(train_X)):
        dist = vector_length(train_X[i] - test_sample)
        distances.append((dist, train_y[i]))

    distances.sort()
    nearest_neighbors = distances[:k]

    class0_count = 0
    class1_count = 0

    for _, label in nearest_neighbors:
        if label == 0:
            class0_count += 1
        else:
            class1_count += 1

    return 0 if class0_count > class1_count else 1

def confusion_matrix(actual_labels, predicted_labels):
    TP = TN = FP = FN = 0

    for i in range(len(actual_labels)):
        if actual_labels[i] == 1 and predicted_labels[i] == 1:
            TP += 1
        elif actual_labels[i] == 0 and predicted_labels[i] == 0:
            TN += 1
        elif actual_labels[i] == 0 and predicted_labels[i] == 1:
            FP += 1
        else:
            FN += 1

    return TP, TN, FP, FN


def linear_regression_weights(train_X, train_y):
    ones_column = np.ones((len(train_X), 1))
    X = np.hstack((ones_column, train_X))
    weights = np.linalg.inv(X.T @ X) @ X.T @ train_y
    return weights



def main():
    feature_data = np.array([
        [1, 2],
        [2, 3],
        [3, 3],
        [6, 5],
        [7, 7],
        [8, 6]
    ])

    label_data = np.array([0, 0, 0, 1, 1, 1])

    vector1 = np.array([1, 2, 3])
    vector2 = np.array([4, 5, 6])

    print("Dot product:", dot_product(vector1, vector2))
    print("Vector length:", vector_length(vector1))

    class0_samples = feature_data[label_data == 0]
    class1_samples = feature_data[label_data == 1]

    print("Distance between class centers:",
          euclidean_distance(class_center(class0_samples),
                             class_center(class1_samples)))

    print("Minkowski distance:",
          minkowski_distance(vector1, vector2, 2),
          minkowski(vector1, vector2, 2))

    X_train, X_test, y_train, y_test = split_dataset(feature_data, label_data)

    knn = train_knn_model(X_train, y_train, 3)
    predictions = make_prediction(knn, X_test)

    print("kNN Accuracy:", calculate_accuracy(knn, X_test, y_test))

    print("Custom kNN prediction:",
          custom_knn(X_train, y_train, X_test[0], 3))

    max_k = len(X_train)
    for k in range(1, max_k + 1):
        knn_k = train_knn_model(X_train, y_train, k)
        print("k =", k, "Accuracy =", calculate_accuracy(knn_k, X_test, y_test))

    TP, TN, FP, FN = confusion_matrix(y_test, predictions)
    print("TP TN FP FN:", TP, TN, FP, FN)

    print("Matrix method weights:",
          linear_regression_weights(X_train, y_train))


main()
