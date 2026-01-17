import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns

# A1
def load_purchase_data(file_path):
    
    data = pd.read_excel(file_path, sheet_name=0)

    candies = data["Candies (#)"].values
    mangoes = data["Mangoes (Kg)"].values
    milk = data["Milk Packets (#)"].values

    X = np.column_stack((candies, mangoes, milk))
    y = data["Payment (Rs)"].values

    return X, y


def calculate_matrix_rank(feature_matrix):
    rank = np.linalg.matrix_rank(feature_matrix)
    return rank


def calculate_product_costs(X, y):
    pseudo_inverse = np.linalg.pinv(X)
    cost_vector = np.dot(pseudo_inverse, y)
    return cost_vector

# A2

def classify_customers(payment_vector):
    customer_class = []

    for amount in payment_vector:
        if amount > 200:
            customer_class.append("RICH")
        else:
            customer_class.append("POOR")

    return customer_class



# A3
def load_stock_data(file_path):
    
    return pd.read_excel(file_path, sheet_name=1)


def calculate_mean_manual(data):
    total = 0
    count = 0

    for value in data:
        total += value
        count += 1

    mean = total / count
    return mean


def calculate_variance_manual(data):
    mean = calculate_mean_manual(data)
    variance_sum = 0

    for value in data:
        variance_sum += (value - mean) ** 2

    variance = variance_sum / len(data)
    return variance


def calculate_execution_time(function, data):
    times = []

    for _ in range(10):
        start = time.time()
        function(data)
        end = time.time()
        times.append(end - start)

    average_time = sum(times) / len(times)
    return average_time


def probability_of_loss(change_percent):
    loss_count = 0

    for value in change_percent:
        if value < 0:
            loss_count += 1

    probability = loss_count / len(change_percent)
    return probability


def probability_of_profit_on_wednesday(stock_data):
    wednesday_data = stock_data[stock_data["Day"] == "Wednesday"]
    profit_days = wednesday_data[wednesday_data["Chg%"] > 0]

    probability = len(profit_days) / len(wednesday_data)
    return probability


# A4
def load_thyroid_data(file_path):
    
    return pd.read_excel(file_path, sheet_name=2)


def analyze_missing_values(df):
    missing_report = {}

    for column in df.columns:
        missing_report[column] = df[column].isnull().sum()

    return missing_report



# A5

def calculate_jaccard_coefficient(v1, v2):
    f11 = f10 = f01 = 0

    for i in range(len(v1)):
        if v1[i] == 1 and v2[i] == 1:
            f11 += 1
        elif v1[i] == 1 and v2[i] == 0:
            f10 += 1
        elif v1[i] == 0 and v2[i] == 1:
            f01 += 1

    jc = f11 / (f11 + f10 + f01)
    return jc


def calculate_smc(v1, v2):
    f11 = f00 = f10 = f01 = 0

    for i in range(len(v1)):
        if v1[i] == 1 and v2[i] == 1:
            f11 += 1
        elif v1[i] == 0 and v2[i] == 0:
            f00 += 1
        elif v1[i] == 1 and v2[i] == 0:
            f10 += 1
        else:
            f01 += 1

    smc = (f11 + f00) / (f11 + f00 + f10 + f01)
    return smc


# A6. COSINE SIMILARITY

def calculate_cosine_similarity(v1, v2):
    dot_product = 0
    mag_v1 = 0
    mag_v2 = 0

    for i in range(len(v1)):
        dot_product += v1[i] * v2[i]
        mag_v1 += v1[i] ** 2
        mag_v2 += v2[i] ** 2

    cosine_similarity = dot_product / (np.sqrt(mag_v1) * np.sqrt(mag_v2))
    return cosine_similarity



# A7. HEATMAP
def generate_cosine_heatmap(data):
    similarity_matrix = []

    for i in range(20):
        row = []
        for j in range(20):
            similarity = calculate_cosine_similarity(data[i], data[j])
            row.append(similarity)
        similarity_matrix.append(row)

    sns.heatmap(similarity_matrix)
    plt.title("Cosine Similarity Heatmap")
    plt.show()



# MAIN FUNCTION

def main():
    file_path = "Lab Session Data.xlsx"

    # A1
    X, y = load_purchase_data(file_path)
    print("Rank of Feature Matrix:", calculate_matrix_rank(X))
    print("Cost of Products:", calculate_product_costs(X, y))

    # A2
    customer_type = classify_customers(y)
    print("Customer Classification:", customer_type)

    # A3
    stock_data = load_stock_data(file_path)
    prices = stock_data["Price"].values
    print("Mean (Manual):", calculate_mean_manual(prices))
    print("Variance (Manual):", calculate_variance_manual(prices))

    # A4
    thyroid = load_thyroid_data(file_path)
    print("Missing Value Analysis:", analyze_missing_values(thyroid))

    # A5 & A6
    v1 = np.array([1, 0, 1, 1])
    v2 = np.array([1, 1, 0, 1])
    print("Jaccard:", calculate_jaccard_coefficient(v1, v2))
    print("SMC:", calculate_smc(v1, v2))
    print("Cosine:", calculate_cosine_similarity(v1, v2))


main()
