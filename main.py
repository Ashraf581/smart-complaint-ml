# ============================================
# main.py
# FastAPI ML Microservice using Hugging Face
# Runs on http://localhost:8000
# ============================================

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ============================================
# STEP 1: Create FastAPI app
# ============================================
app = FastAPI(
    title="Smart Complaint ML Service",
    description="Predicts department and priority using Hugging Face AI",
    version="2.0.0"
)

# ============================================
# STEP 2: Hugging Face Configuration
# ============================================
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
HF_API_URL = "https://router.huggingface.co/hf-inference/models/facebook/bart-large-mnli"

HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}

# Department labels the model will classify into
DEPARTMENT_LABELS = [
    "water supply problem",
    "electricity problem",
    "road problem",
    "sanitation and garbage problem",
    "park and recreation problem"
]

# Mapping from model output to our department names
LABEL_TO_DEPARTMENT = {
    "water supply problem": "WATER",
    "electricity problem": "ELECTRICITY",
    "road problem": "ROADS",
    "sanitation and garbage problem": "SANITATION",
    "park and recreation problem": "PARKS"
}

# Priority keywords — we determine priority from text
HIGH_PRIORITY_KEYWORDS = [
    "urgent", "dangerous", "accident", "burst",
    "no water", "no electricity", "blocked", "emergency",
    "days", "week", "leaking", "broken", "damaged"
]

MEDIUM_PRIORITY_KEYWORDS = [
    "not working", "missing", "dirty",
    "smell", "noise", "complaint"
]

# ============================================
# STEP 3: Request and Response models
# ============================================
class ComplaintRequest(BaseModel):
    title: str
    description: str

    class Config:
        json_schema_extra = {
            "example": {
                "title": "No water supply",
                "description": "There has been no water supply in our area for 3 days"
            }
        }

class PredictionResponse(BaseModel):
    department: str
    priority: str
    confidence: str

# ============================================
# STEP 4: Helper Functions
# ============================================

def determine_priority(text: str) -> str:
    """
    Determines priority based on keywords in complaint text
    HIGH   → urgent/dangerous situations
    MEDIUM → moderate issues
    LOW    → minor complaints
    """
    text_lower = text.lower()

    for keyword in HIGH_PRIORITY_KEYWORDS:
        if keyword in text_lower:
            return "HIGH"

    for keyword in MEDIUM_PRIORITY_KEYWORDS:
        if keyword in text_lower:
            return "MEDIUM"

    return "LOW"


def classify_complaint(text: str) -> dict:
    """
    Calls Hugging Face API for zero-shot classification
    Returns department and confidence score
    """
    payload = {
        "inputs": text,
        "parameters": {
            "candidate_labels": DEPARTMENT_LABELS
        }
    }

    response = requests.post(
        HF_API_URL,
        headers=HEADERS,
        json=payload,
        timeout=30
    )

    if response.status_code != 200:
        raise HTTPException(
            status_code=500,
            detail=f"Hugging Face API error: {response.text}"
        )

    result = response.json()

    # Print response for debugging
    print(f"🔍 Hugging Face Response: {result}")

    # HF returns a LIST of objects like:
    # [{"label": "water supply problem", "score": 0.98}, ...]
    # First item is the highest scoring one!
    top_result = result[0]
    top_label = top_result["label"]    # singular "label" ✅
    top_score = top_result["score"]    # singular "score" ✅

    return {
        "department": LABEL_TO_DEPARTMENT[top_label],
        "confidence": f"{round(top_score * 100, 2)}%"
    }
# ============================================
# STEP 5: API Endpoints
# ============================================

@app.get("/")
def root():
    return {
        "service": "Smart Complaint ML Service",
        "version": "2.0.0",
        "model": "facebook/bart-large-mnli",
        "status": "running"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model": "facebook/bart-large-mnli",
        "api_token_loaded": HF_API_TOKEN is not None
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(complaint: ComplaintRequest):
    try:
        # Combine title and description
        full_text = f"{complaint.title}. {complaint.description}"

        # Call Hugging Face API for department
        classification = classify_complaint(full_text)

        # Determine priority from keywords
        priority = determine_priority(full_text)

        return PredictionResponse(
            department=classification["department"],
            priority=priority,
            confidence=classification["confidence"]
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )