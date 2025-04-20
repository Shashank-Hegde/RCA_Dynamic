

# symptom_net/constants.py

CANON_KEYS = [
    # Demographics
    "age", "sex",
    # Symptom categories (flattened)
    "fever", "cough", "nausea", "vomiting", "headache", "dizziness",
    "chest_pain", "chest_tightness", "fatigue", "sore_throat",
    "hemoptysis", "abdominal_pain", "diarrhea", "rash",
    # Add more as needed for your ontology
    "symptom_duration_map"
]
