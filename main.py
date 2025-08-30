import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Step 1: Load dataset
data = pd.read_csv("data.csv")
print("Dataset Loaded Successfully ✅")
print(data.head(), "\n")

# Step 2: Preprocess Data
# Convert Date to numeric format
data["Date"] = pd.to_datetime(data["Date"])
data["Date_ordinal"] = data["Date"].map(lambda x: x.toordinal())

# Step 3: Select features (X) and target (y)
X = data[["Date_ordinal"]]        # input feature (time)
y = data["AQI Value"]             # target (AQI)

# Step 4: Split data into training & testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Make predictions
predictions = model.predict(X_test)

print("Predicted AQI values:", predictions[:5])
print("Actual AQI values:", y_test[:5].values, "\n")

# Step 7: Plot graph
plt.figure(figsize=(8,5))
plt.scatter(y_test, predictions, color="blue")
plt.xlabel("Actual AQI")
plt.ylabel("Predicted AQI")
plt.title("Actual vs Predicted AQI")
plt.show()
