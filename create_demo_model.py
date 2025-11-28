from sklearn.linear_model import LinearRegression
import numpy as np
import joblib

# Create a simple demo model for testing
X = np.random.rand(100, 8) * 100
y = np.random.rand(100) * 500000

model = LinearRegression()
model.fit(X, y)

joblib.dump(model, 'house_model.pkl')
print('✅ Demo model created and saved!')
