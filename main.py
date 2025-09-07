
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ------------------ WEEK 1 ------------------
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

# ------------------ WEEK 2 ------------------
# Step 6: Preprocess Data
# Convert Date to numeric format
data["Date"] = pd.to_datetime(data["Date"])
data["Date_ordinal"] = data["Date"].map(lambda x: x.toordinal())

# Step 7: Select features (X) and target (y)
X = data[["Date_ordinal"]]        # input feature (time)
y = data["AQI Value"]             # target (AQI)

# Step 8: Split data into training & testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 9: Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 10: Make predictions
predictions = model.predict(X_test)

print("\n🔹 Sample Predictions:")
print("Predicted AQI values:", predictions[:5])
print("Actual AQI values:", y_test[:5].values)

# Step 11: Evaluate Model
mse = mean_squared_error(y_test, predictions)
rmse = mse**0.5
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\n✅ Model Evaluation Results:")
print("MAE:", mae)
print("RMSE:", rmse)
print("R² Score:", r2)

# Step 12: Plot Actual vs Predicted
plt.figure(figsize=(8,5))
plt.scatter(y_test, predictions, color="blue")
plt.xlabel("Actual AQI")
plt.ylabel("Predicted AQI")
plt.title("Actual vs Predicted AQI")
plt.show()
