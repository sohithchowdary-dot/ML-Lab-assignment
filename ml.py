import pandas as pd

data = pd.read_excel(
    r"C:\Users\sohit\Downloads\lab2.ML\Lab Session Data.xlsx"
)


print(data.head())

X = data[["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)"]].values
y = data["Payment (Rs)"].values

print("Features (X):")
print(X)    
print("Payment (y):  ")
print(y)

rank_X = np.linalg.matrix_rank(X)

print("Rank of Feature Matrix:", rank_X)
X_pinv = np.linalg.pinv(X)

# Cost calculation
cost = X_pinv.dot(y)

print("Cost of Candies     :", cost[0])
print("Cost of Mangoes (Kg):", cost[1])
print("Cost of Milk Packets:", cost[2])