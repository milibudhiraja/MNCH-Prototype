"""
app.py
------
Gradio interactive demo for the Dodoma Pregnancy Risk Stratification Model.

Run locally:
    python app/app.py

Deploy to Hugging Face Spaces:
    1.  Create a new Space at https://huggingface.co/spaces (choose Gradio SDK)
    2.  Push this repo (the Space picks up app.py automatically)
    3.  Make sure requirements.txt includes: gradio, scikit-learn, pandas, numpy
"""

import sys
from pathlib import Path

# Make src importable
import gradio as gr

# ── Risk colour helper ────────────────────────────────────────────────────────
def risk_colour(label):
    return "🔴" if label == "High Risk" else "🟢"


# ── Main prediction function called by Gradio ─────────────────────────────────
def run_prediction(
    age, no_pregnancy, duration_weeks, height_cm, weight_kg,
    prior_deliveries, prior_stillbirths, prior_caesarean,
    hiv, diabetes_mellitus, hypertension, heart_problems,
    hemoglobin_v1, bp_systolic_v1,
    preterm_labour_1, prior_live_births,
    miscarriages, twin_pregnancy, anaemia_flag,
    intimate_partner_violence,
):
    # Derived features
    bmi = weight_kg / ((height_cm / 100) ** 2) if height_cm > 0 else float("nan")

    patient = {
        "age": age,
        "no_pregnancy": no_pregnancy,
        "duration_weeks": duration_weeks,
        "height_cm": height_cm,
        "weight_kg": weight_kg,
        "bmi": bmi,
        "prior_deliveries": prior_deliveries,
        "prior_stillbirths": prior_stillbirths,
        "prior_live_births": prior_live_births,
        "prior_caesarean_delivery": int(prior_caesarean),
        "hiv": int(hiv),
        "diabetes_mellitus": int(diabetes_mellitus),
        "hypertension": int(hypertension),
        "heart_problems": int(heart_problems),
        "hemoglobin_v1": hemoglobin_v1,
        "hemoglobin_min": hemoglobin_v1,   # use v1 as proxy if only one value
        "bp_systolic_v1": bp_systolic_v1,
        "bp_systolic_max": bp_systolic_v1,
        "bp_high_count": int(bp_systolic_v1 >= 140),
        "bp_flag_v1": int(bp_systolic_v1 >= 140),
        "preterm_labour_1": int(preterm_labour_1),
        "miscarriages_or_abortions": int(miscarriages),
        "twin_pregnancy": int(twin_pregnancy),
        "anaemia_flag": int(anaemia_flag),
        "intimate_partner_violence": int(intimate_partner_violence),
        "adolescent": int(age < 18),
        "grand_multipara": int(prior_deliveries >= 5),
        # Fill the rest with 0 / neutral
        "living_children": prior_live_births,
        "hiv_positive_v1": int(hiv),
        "malaria_positive_v1": 0,
        "proteinuria_v1": 0,
    }

    result = predict_risk(patient)
    label = result["risk_label"]
    prob = result["probability"]
    icon = risk_colour(label)

    # Format output
    risk_display = f"{icon} **{label}**"
    prob_display = f"Probability of High Risk: **{prob * 100:.1f}%**"

    factors_text = "**Key contributing factors:**\n"
    for name, imp in result["top_factors"]:
        bar = "█" * int(imp * 100)
        factors_text += f"- {name}: {bar} ({imp:.3f})\n"

    disclaimer = (
        "\n---\n"
        "⚠️ *This tool is for research and decision-support only. "
        "It does not replace clinical judgement. Always follow local clinical guidelines.*"
    )

    return risk_display, prob_display, factors_text + disclaimer


# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(title="Dodoma Pregnancy Risk Model", theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    # 🤰 Pregnancy Risk Stratification — Dodoma, Tanzania
    **ML model trained on antenatal care records from Dodoma.**
    Fill in the patient details below to predict whether the pregnancy is **High Risk** or **Low Risk**.
    """)

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 👤 Demographics & Pregnancy Info")
            age = gr.Slider(10, 55, value=25, step=1, label="Age (years)")
            no_pregnancy = gr.Slider(1, 12, value=1, step=1, label="Current pregnancy number")
            duration_weeks = gr.Slider(4, 42, value=20, step=1, label="Gestational age (weeks)")
            height_cm = gr.Slider(130, 200, value=158, step=1, label="Height (cm)")
            weight_kg = gr.Slider(35, 120, value=60, step=1, label="Weight (kg)")

            gr.Markdown("### 🩺 Obstetric History")
            prior_deliveries = gr.Slider(0, 10, value=1, step=1, label="Prior deliveries")
            prior_live_births = gr.Slider(0, 10, value=1, step=1, label="Prior live births")
            prior_stillbirths = gr.Slider(0, 5, value=0, step=1, label="Prior stillbirths")
            prior_caesarean = gr.Checkbox(label="Prior caesarean delivery")
            preterm_labour_1 = gr.Checkbox(label="Prior preterm labour")
            miscarriages = gr.Checkbox(label="Previous miscarriage / abortion")
            twin_pregnancy = gr.Checkbox(label="Current twin pregnancy")

        with gr.Column():
            gr.Markdown("### 🔬 Medical History")
            hiv = gr.Checkbox(label="HIV positive")
            diabetes_mellitus = gr.Checkbox(label="Diabetes mellitus")
            hypertension = gr.Checkbox(label="Hypertension")
            heart_problems = gr.Checkbox(label="Heart problems")
            intimate_partner_violence = gr.Checkbox(label="Intimate partner violence")

            gr.Markdown("### 📋 Visit 1 Clinical Findings")
            hemoglobin_v1 = gr.Slider(4.0, 18.0, value=11.5, step=0.1, label="Haemoglobin (g/dL)")
            bp_systolic_v1 = gr.Slider(70, 200, value=115, step=1, label="Systolic BP (mmHg)")
            anaemia_flag = gr.Checkbox(label="Haemoglobin < 11 g/dL (anaemia)")

    predict_btn = gr.Button("🔍 Predict Risk", variant="primary", size="lg")

    with gr.Row():
        out_label = gr.Markdown()
        out_prob = gr.Markdown()

    out_factors = gr.Markdown()

    predict_btn.click(
        fn=run_prediction,
        inputs=[
            age, no_pregnancy, duration_weeks, height_cm, weight_kg,
            prior_deliveries, prior_stillbirths, prior_caesarean,
            hiv, diabetes_mellitus, hypertension, heart_problems,
            hemoglobin_v1, bp_systolic_v1,
            preterm_labour_1, prior_live_births,
            miscarriages, twin_pregnancy, anaemia_flag,
            intimate_partner_violence,
        ],
        outputs=[out_label, out_prob, out_factors],
    )

    gr.Markdown("""
    ---
    **Model:** Gradient Boosting Classifier | **AUC-ROC:** 0.96 | **Accuracy:** 94%
    **Training data:** 8,817 antenatal records, Dodoma Regional Referral Hospital, Tanzania
    """)


if __name__ == "__main__":
    demo.launch(share=True)
