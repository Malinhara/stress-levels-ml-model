from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

# Load the trained XGBoost model
model_xgb = joblib.load('./model/xgb_model.pkl')

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware

# origins = [ "*",  # Allows all origins, use with caution
# ]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,  # List of allowed origins
#     allow_credentials=True,
#     allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
#     allow_headers=["*"],  # Allows all headers
# )

# Define input data schema using Pydantic
class InputData(BaseModel):
    anxiety_level: int
    self_esteem: int
    mental_health_history: int
    depression: int
    headache: int
    blood_pressure: int
    sleep_quality: int
    breathing_problem: int
    noise_level: int
    living_conditions: int
    safety: int
    basic_needs: int
    academic_performance: int
    study_load: int
    teacher_student_relationship: int
    future_career_concerns: int
    social_support: int
    peer_pressure: int
    extracurricular_activities: int
    bullying: int

@app.post("/predict")
def predict(input_data: InputData):
    # Convert the input data into a pandas DataFrame
    data = input_data.dict()
    sample_df = pd.DataFrame([data])

    # Predict using the loaded model
    prediction = model_xgb.predict(sample_df)

    # Convert numpy.int32 to Python int
    prediction_value = int(prediction[0])

    # Return the predicted stress level
    return {"Predicted Stress Level": prediction_value}