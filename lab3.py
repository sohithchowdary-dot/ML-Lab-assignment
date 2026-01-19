#A1
import numpy as np

def compute_dot(vec_x, vec_y):
    result = 0
    for idx in range(len(vec_x)):
        result += vec_x[idx] * vec_y[idx]
    return result

def compute_euclidean_length(vec):
    total = 0
    for val in vec:
        total += val ** 2
    return total ** 0.5

vector_X = [2, 4, 6]
vector_Y = [1, 3, 5]

np_vector_X = np.array(vector_X)
np_vector_Y = np.array(vector_Y)

manual_dot_product = compute_dot(vector_X, vector_Y)
manual_norm_X = compute_euclidean_length(vector_X)
manual_norm_Y = compute_euclidean_length(vector_Y)

numpy_dot_product = np.dot(np_vector_X, np_vector_Y)
numpy_norm_X = np.linalg.norm(np_vector_X)
numpy_norm_Y = np.linalg.norm(np_vector_Y)

print("Manual Dot Product:", manual_dot_product)
print("NumPy Dot Product:", numpy_dot_product)
print("Manual Euclidean Norm of X:", manual_norm_X)
print("NumPy Euclidean Norm of X:", numpy_norm_X)
print("Manual Euclidean Norm of Y:", manual_norm_Y)
print("NumPy Euclidean Norm of Y:", numpy_norm_Y)