# MNCH-Prototype: Pregnancy Risk Stratification

A machine learning model that predicts whether a pregnant woman is High Risk or Low Risk using antenatal care (ANC) records
---

## 📊 Dataset
- Source: Antenatal care records, Dodoma Regional Referral Hospital, Tanzania
- Size: 8,817 patients, 683 variables
- Target: Risk (High / Low)
- Coverage: Up to 8 ANC visits, delivery outcomes, postnatal follow-up

---

## 🤖 Model Performance

| Model | ROC-AUC | Accuracy |
|---|---|---|
| Logistic Regression | 0.922 | 86% |
| Random Forest | 0.954 | 92% |
| Gradient Boosting | 0.962 | 94% |

### Top Predictive Features
1. Haemoglobin at first visit
2. Number of pregnancies
3. Prior caesarean delivery
4. BMI
5. HIV status

---

## 📁 Project Structure
```
MNCH-Prototype/
├── src/
│   ├── preprocess.py       # Feature engineering (683 → 70 features)
│   ├── train.py            # Model training and evaluation
│   └── predict.py          # Inference function
├── app/
│   └── app.py              # Gradio interactive demo
├── models/
│   └── risk_model.pkl      # Trained model
├── Dodoma_Pregnancy_Risk.ipynb   # Full notebook (run in Google Colab)
└── requirements.txt
```

---

## 🚀 Run in Google Colab (no installation needed)

1. Open [Google Colab](https://colab.research.google.com)
2. Upload `Dodoma_Pregnancy_Risk.ipynb`
3. Run each cell top to bottom
4. Get a live demo link at the end

---

## ⚠️ Disclaimer
This tool is for research and decision-support only. It does not replace clinical judgement. Always follow local clinical guidelines.
