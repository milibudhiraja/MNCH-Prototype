import pickle, sys
import numpy as np
import pandas as pd
from pathlib import Path

# Load model
with open('/content/risk_model.pkl', 'rb') as f:
    _artifact = pickle.load(f)

def predict_risk(patient):
    model = _artifact['model']
    feature_names = _artifact['feature_names']
    fi = _artifact['feature_importances']
    row = {f: patient.get(f, np.nan) for f in feature_names}
    X = pd.DataFrame([row])
    prob = model.predict_proba(X)[0][1]
    label = 'High Risk' if prob >= 0.5 else 'Low Risk'
    top_factors = []
    if fi is not None:
        for feat, imp in fi.head(10).items():
            val = row.get(feat, np.nan)
            if not (np.isnan(val) if isinstance(val, float) else False):
                top_factors.append((feat.replace('_', ' ').title(), round(float(imp), 4)))
        top_factors = top_factors[:5]
    return {'risk_label': label, 'probability': round(float(prob), 4), 'top_factors': top_factors}

print('✅ Model loaded')
