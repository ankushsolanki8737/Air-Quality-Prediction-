import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# Step 1: Load dataset
data = pd.read_csv("data.csv")
print("Dataset Loaded Successfully ✅")
print(data.head(), "\n")

# Step 2: Preprocess Data
# Convert Date to numeric format
data["Date"] = pd.to_datetime(data["Date"])
data["Date_ordinal"] = data["Date"].map(lambda x: x.toordinal())

# Encode categorical variables (Country, Status)
data_encoded = pd.get_dummies(data[["Country", "Status"]], drop_first=True)

# Step 3: Select features (X) and target (y)
X = pd.concat([data[["Date_ordinal"]], data_encoded], axis=1)   # multiple features
y = data["AQI Value"]

# Step 4: Split data into training & testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Make predictions
predictions = model.predict(X_test)

print("Predicted AQI values:", predictions[:5])
print("Actual AQI values:", y_test[:5].values, "\n")

# Step 6.1: Evaluate performance
print("R² Score:", r2_score(y_test, predictions))
print("Mean Absolute Error:", mean_absolute_error(y_test, predictions))

# Step 7: Plot graph
plt.figure(figsize=(8,5))
plt.scatter(y_test, predictions, color="blue")
plt.xlabel("Actual AQI")
plt.ylabel("Predicted AQI")
plt.title("Actual vs Predicted AQI")
plt.show()
