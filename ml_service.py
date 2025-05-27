from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import uvicorn
app = FastAPI(
    title="Football Match Prediction API",
    description="API for prediction football match results (H, D, A) using a trained model.",
    version="1.0.0",
)

model = None
preprocessor = None
label_encoder = None
def load_model():
    """Loads the trained model, preprocessor, and label encoder."""
    global model, preprocessor, label_encoder
    try:
        # --- ColumnTransformer pipeline that handles scaling and one-hot encoding of input features ---
        preprocessor = joblib.load("preprocessor.pkl")
        print("Preprocessor loaded successfully!")
        # --- Load trained model ---
        # Choose the correct loading method based on model type
        # For scikit-learn models (LogisticRegression, RandomForest, XGBoost from sklearn):
        model = joblib.load("model.pkl")
        print("Scikit-learn model loaded successfully!")
        # --- Load the LabelEncoder (used to convert numerical predictions back to 'H', 'D', 'A') ---
        label_encoder = joblib.load("label_encoder.pkl")
        print("Label encoder loaded successfully!")
    except FileNotFoundError:
        print("Error: Model, preprocessor, or label encoder file not found. Please ensure they are in the same directory as ml_service.py")
        
    except Exception as e:
        print(f"An unexpected error occurred during artifact loading: {e}")

# Call the loading function when the FastAPI app starts up
@app.on_event("startup")
async def startup_event():
    load_model()

class MatchDataInput(BaseModel):
    """
    Defines the expected structure of the input JSON for a single match prediction.
    """
    homeTeam_Form: float
    awayTeam_Form: float
    home_Goals_Avg: float
    away_Goals_Avg: float
    last_5_Meetings_HomeWins: int
    last_5_Meetings_AwayWins: int

class PredictionOutput(BaseModel):
    """
    Defines the structure of the output JSON for the prediction result.
    """
    predicted_label: str
    probabilites: dict[str, float]  # Dictionary of class labels and their probabilities
    
@app.post("/predict",response_model=PredictionOutput, summary="Predict match result (H, D, A)")
async def predict_match_result(match_data: MatchDataInput):
    """
    API endpoint to receive match data, make a prediction, and return the result.
    Expects a JSON payload for a single match data object.
    Example:
    {
        "homeTeam_Form": 0.8,
        "awayTeam_Form": 0.5,
        "home_Goals_Avg": 2.1,
        "away_Goals_Avg": 0.9,
        "last_5_Meetings_HomeWins": 4,
        "last_5_Meetings_AwayWins": 1,
        
    }
    """
    if model is None or preprocessor is None or label_encoder is None:
        raise HTTPException(status_code=500, detail="ML service artifacts not loaded. Please check server logs.")
    try:
        # Convert the input data to a DataFrame for processing
        input_data = pd.DataFrame([match_data.dict()])
        
        # Apply the preprocessor to transform the input data
        processed_data = preprocessor.transform(input_data)
        
        prediction_proba = model.predict_proba(processed_data)
        
        # Make prediction using the loaded model
        predicted_encoded_label = model.predict(processed_data)[0]
        
        predicted_label = str(predicted_encoded_label)
        # Decode the predicted label back to 'H', 'D', 'A'
        class_labels = label_encoder.classes_.tolist()
        
        
        
        response_data = {
            'predicted_label': predicted_label,
            'probabilites': {label: float(prob) for label, prob in zip(class_labels, prediction_proba[0])}
        }
        
        return PredictionOutput(**response_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
    

if __name__ == '__main__':
    # Run the FastAPI application using Uvicorn.
    # host='0.0.0.0' makes the server accessible from any IP address.
    # port=5000 is the port the ML service will listen on.
    # reload=True enables auto-reloading on code changes (for development).
    uvicorn.run(app, host="0.0.0.0", port=5000, reload=True)
    