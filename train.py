import numpy as np
from sklearn.linear_model import LinearRegression
import joblib

# Training data
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
y = np.array([30, 35, 42, 50, 58, 65, 72, 80, 88, 95])

# Train model
model = LinearRegression()
model.fit(X, y)

# Save the trained model to a file
joblib.dump(model, 'salary_model.pkl')
print("Model trained and saved as 'salary_model.pkl'!")