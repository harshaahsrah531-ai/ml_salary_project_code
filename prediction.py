import numpy as np
import joblib
import os

# 1. Check if the model exists, then load it
model_filename = 'salary_model.pkl'
if not os.path.exists(model_filename):
    print(f"Error: '{model_filename}' not found. Please run the training script first!")
    exit()

model = joblib.load(model_filename)
print("Model loaded successfully!\n--- Salary Prediction Tool ---")

# 2. Interactive prediction loop
while True:
    user_input = input("Enter years of experience (or type 'q' to quit): ")
    
    if user_input.lower() == 'q':
        print("Exiting. Goodbye!")
        break
        
    try:
        # Convert input to float
        experience = float(user_input)
        
        # Scikit-learn expects a 2D array for input features
        X_new = np.array([[experience]])
        
        # Make prediction
        predicted_salary = model.predict(X_new)
        
        print(f"-> Predicted Salary for {experience} years: ${predicted_salary[0]:.2f}k\n")
        
    except ValueError:
        print("Invalid input. Please enter a valid number for years of experience.\n")