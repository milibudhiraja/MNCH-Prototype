# MNCH-Prototype: Pregnancy Risk Stratification

The Challenge -

Maternal health remains a critical challenge in many low-resource settings, where timely identification of high-risk pregnancies is often limited by gaps in clinical resources, inconsistent follow-up, and high patient volumes. Many complications—such as anemia, preeclampsia, or prior obstetric risks—can go unnoticed until they become severe, leading to preventable maternal and neonatal morbidity or mortality.

Solution -

This project develops a machine learning model that predicts whether a pregnant woman is "High Risk" or "Low Risk" using antenatal care (ANC) records. By leveraging historical clinical data across multiple visits, the model identifies patterns and risk factors that may not be immediately obvious during routine assessments.

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
