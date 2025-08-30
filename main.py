import pandas as pd

# Step 1: Load dataset
data = pd.read_csv("data.csv")
print("✅ Dataset Loaded Successfully")

# Step 2: Show first few rows
print("\n🔹 First 5 rows of dataset:\n", data.head())

# Step 3: Show dataset shape
print("\n🔹 Dataset Shape (rows, columns):", data.shape)

# Step 4: Show column names
print("\n🔹 Columns in dataset:", data.columns.tolist())

# Step 5: Show basic statistics of numerical columns
print("\n🔹 Dataset Statistics:\n", data.describe())
