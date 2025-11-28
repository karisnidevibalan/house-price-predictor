# ✅ STEP 7: Export Model for Deployment
# Run this AFTER your training code in Google Colab

import pickle
import joblib

# Save with joblib (standard method)
joblib.dump(model, "house_model.pkl")
print("✅ Model saved with joblib!")

# Also save with pickle as backup
with open("house_model.pkl", "wb") as f:
    pickle.dump(model, f, protocol=4)
print("✅ Model saved with pickle (protocol 4)!")

# Download the file to your computer
from google.colab import files
files.download("house_model.pkl")
print("✅ Download started! Check your Downloads folder")
