# ============================================
# train_model.py
# This file trains our ML model and saves it
# Run this file ONCE to create the model
# ============================================

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

# ============================================
# STEP 1: Create Training Data
# These are examples our model learns from
# More examples = better predictions!
# ============================================

training_data = {
    "text": [
        # WATER complaints
        "no water supply in my area",
        "water pipe is leaking on the street",
        "dirty water coming from tap",
        "water shortage for 3 days",
        "broken water pipeline",
        "no drinking water available",
        "sewage water overflowing",
        "water meter is not working",
        "water supply disrupted",
        "contaminated water in our locality",

        # ELECTRICITY complaints
        "street light not working",
        "power cut since yesterday",
        "electric pole is damaged",
        "no electricity in our area",
        "transformer is making loud noise",
        "electric wire hanging dangerously",
        "frequent power outages",
        "electricity bill is too high",
        "short circuit in the area",
        "power fluctuation damaging appliances",

        # ROADS complaints
        "big pothole on main road",
        "road is damaged after rain",
        "no street lights on highway",
        "road construction blocking traffic",
        "broken footpath near school",
        "road flooding during rain",
        "speed breaker is too high",
        "road markings are not visible",
        "dangerous road condition near bridge",
        "road cave in near market",

        # SANITATION complaints
        "garbage not collected for a week",
        "garbage bin is overflowing",
        "open defecation near park",
        "dead animal on the road",
        "public toilet is very dirty",
        "drainage is blocked",
        "bad smell from garbage dump",
        "stray dogs near garbage area",
        "illegal dumping of waste",
        "sewage overflow on street",

        # PARKS complaints
        "park benches are broken",
        "children playground equipment damaged",
        "grass in park not maintained",
        "park lights not working at night",
        "trees in park need trimming",
        "park is being used for illegal activities",
        "no drinking water in park",
        "park is very dirty and unclean",
        "park gate is broken",
        "park flooded after rain",
    ],

    "department": [
        # WATER
        "WATER", "WATER", "WATER", "WATER", "WATER",
        "WATER", "WATER", "WATER", "WATER", "WATER",
        # ELECTRICITY
        "ELECTRICITY", "ELECTRICITY", "ELECTRICITY", "ELECTRICITY", "ELECTRICITY",
        "ELECTRICITY", "ELECTRICITY", "ELECTRICITY", "ELECTRICITY", "ELECTRICITY",
        # ROADS
        "ROADS", "ROADS", "ROADS", "ROADS", "ROADS",
        "ROADS", "ROADS", "ROADS", "ROADS", "ROADS",
        # SANITATION
        "SANITATION", "SANITATION", "SANITATION", "SANITATION", "SANITATION",
        "SANITATION", "SANITATION", "SANITATION", "SANITATION", "SANITATION",
        # PARKS
        "PARKS", "PARKS", "PARKS", "PARKS", "PARKS",
        "PARKS", "PARKS", "PARKS", "PARKS", "PARKS",
    ],

    "priority": [
        # WATER - high priority (essential service)
        "HIGH", "HIGH", "HIGH", "HIGH", "HIGH",
        "HIGH", "HIGH", "MEDIUM", "HIGH", "HIGH",
        # ELECTRICITY - high priority
        "MEDIUM", "HIGH", "HIGH", "HIGH", "HIGH",
        "HIGH", "MEDIUM", "LOW", "HIGH", "MEDIUM",
        # ROADS - medium priority
        "HIGH", "HIGH", "MEDIUM", "MEDIUM", "MEDIUM",
        "HIGH", "LOW", "LOW", "HIGH", "HIGH",
        # SANITATION - medium priority
        "HIGH", "MEDIUM", "HIGH", "HIGH", "MEDIUM",
        "HIGH", "MEDIUM", "MEDIUM", "MEDIUM", "HIGH",
        # PARKS - low priority
        "LOW", "MEDIUM", "LOW", "LOW", "LOW",
        "HIGH", "MEDIUM", "LOW", "LOW", "LOW",
    ]
}

# ============================================
# STEP 2: Create DataFrame
# A DataFrame is like an Excel table in Python
# ============================================
df = pd.DataFrame(training_data)
print("✅ Training data created!")
print(f"   Total examples: {len(df)}")
print(f"   Departments: {df['department'].unique()}")
print()

# ============================================
# STEP 3: Train Department Classifier
# Pipeline = TF-IDF + Naive Bayes combined
# TF-IDF    = converts text to numbers
# Naive Bayes = learns patterns and predicts
# ============================================
print("🔄 Training department classifier...")

X = df["text"]           # Input: complaint text
y_dept = df["department"] # Output: department label

# Split data: 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y_dept, test_size=0.2, random_state=42
)

# Create and train the pipeline
department_model = Pipeline([
    ("tfidf", TfidfVectorizer()),   # Step 1: text → numbers
    ("clf", MultinomialNB())         # Step 2: numbers → prediction
])

department_model.fit(X_train, y_train)
print("✅ Department model trained!")

# Test the model accuracy
y_pred = department_model.predict(X_test)
print("\n📊 Department Model Performance:")
print(classification_report(y_test, y_pred))

# ============================================
# STEP 4: Train Priority Classifier
# Same approach but predicts priority
# ============================================
print("🔄 Training priority classifier...")

y_priority = df["priority"]

X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X, y_priority, test_size=0.2, random_state=42
)

priority_model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", MultinomialNB())
])

priority_model.fit(X_train2, y_train2)
print("✅ Priority model trained!")

# ============================================
# STEP 5: Save the trained models to files
# joblib saves Python objects to disk
# We'll load these in main.py later
# ============================================
print("\n💾 Saving models...")

# Save to model/ folder
joblib.dump(department_model, "model/department_model.pkl")
joblib.dump(priority_model, "model/priority_model.pkl")

print("✅ Models saved!")
print("   → model/department_model.pkl")
print("   → model/priority_model.pkl")

# ============================================
# STEP 6: Test with sample complaints
# ============================================
print("\n🧪 Testing with sample complaints:")
print("-" * 45)

test_complaints = [
    "water pipe burst near my house",
    "no electricity since morning",
    "huge pothole damaged my car",
    "garbage not picked up for days",
    "park benches are all broken",
]

for complaint in test_complaints:
    dept = department_model.predict([complaint])[0]
    priority = priority_model.predict([complaint])[0]
    print(f"Complaint : {complaint}")
    print(f"Department: {dept}")
    print(f"Priority  : {priority}")
    print("-" * 45)