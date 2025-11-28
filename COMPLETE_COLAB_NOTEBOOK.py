# 🏠 Complete House Price Prediction Model - Google Colab

# ✅ STEP 1: Install & Import Libraries
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import pickle

# ✅ STEP 2: Load the dataset
# Make sure housing.csv is uploaded to your Colab
df = pd.read_csv("/content/housing.csv")

print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())

# ✅ STEP 3: Remove missing values
df = df.dropna()
print("After dropping NaN:", df.shape)

# ❗ Handle categorical column
if "ocean_proximity" in df.columns:
    df = pd.get_dummies(df, columns=["ocean_proximity"], drop_first=True)
    print("Processed categorical columns")

# ✅ STEP 4: Separate features & target
X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]

print("Features shape:", X.shape)
print("Target shape:", y.shape)

# ✅ STEP 5: Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training set size:", X_train.shape[0])
print("Test set size:", X_test.shape[0])

# ✅ STEP 6: Train Model
print("\n🤖 Training model...")
model = LinearRegression()
model.fit(X_train, y_train)
print("✅ Model trained!")

# ✅ STEP 7: Evaluate Model
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("\n📊 Model Evaluation:")
print(f"MSE: ${mse:,.2f}")
print(f"RMSE: ${rmse:,.2f}")
print(f"R² Score: {r2:.4f}")

# ✅ STEP 8: Save & Export Model
print("\n💾 Saving model...")

# Save with joblib
joblib.dump(model, "house_model.pkl")
print("✅ Model saved with joblib!")

# Also save with pickle as backup
with open("house_model.pkl", "wb") as f:
    pickle.dump(model, f, protocol=4)
print("✅ Model saved with pickle (protocol 4)!")

# Download the file
from google.colab import files
print("\n📥 Starting download...")
files.download("house_model.pkl")
print("✅ Download complete! Check your Downloads folder")

# Save model info
print("\n📝 Model Info:")
print(f"Number of features: {X.shape[1]}")
print(f"Feature names: {X.columns.tolist()}")
