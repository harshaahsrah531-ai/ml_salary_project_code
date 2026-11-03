import joblib
from fastapi import FastAPI
from pydantic import BaseModel

# Initialize our web application
app = FastAPI()

# Load the model once when the app starts
model_filename = './salary_model.pkl'
with open(model_filename, "rb") as file:
    model = joblib.load(file)

# Define what kind of data you expect people to send you
class ModelInput(BaseModel):
    features: list[float]

# Create a "route" (a webpage/doorway) for predictions
@app.post("/predict")
def make_prediction(input_data: ModelInput):
    # Pass the data into our model
    prediction = model.predict([input_data.features])

    # Send the answer back as JSON (universal computer language)
    return {"prediction": prediction.tolist()}