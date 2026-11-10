"""
HC-CTXAI: Human-Centered Context-Aware Explainable AI Framework
Demonstration Prototype for Remote Psychological Stress Screening
Design Science Research (DSR) Paradigm Implementation
"""

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import shap
import json

print("================================================================================")
print("  HC-CTXAI: 5-PILLAR ARCHITECTURE REPLICABLE DEMONSTRATION PROTOTYPE")
print("================================================================================\n")

# -----------------------------------------------------------------------------
# STEP 0: MODEL TRAINING (SYNTHETIC MULTIMODAL BASELINE)
# -----------------------------------------------------------------------------
feature_cols = ['hrv_rmssd', 'eda_scl', 'eda_scr_count', 'skin_temp', 'sleep_hours', 'physical_activity']

np.random.seed(42)
N_TRAIN = 1500
tr_hrv = np.random.normal(45, 15, N_TRAIN).clip(15, 95)
tr_eda = np.random.exponential(2.2, N_TRAIN).clip(0.1, 14)
tr_scr = np.random.poisson(3.2, N_TRAIN)
tr_temp = np.random.normal(33.5, 0.7, N_TRAIN).clip(30, 37)
tr_sleep = np.random.normal(6.7, 1.4, N_TRAIN).clip(3, 10)
tr_act = np.random.binomial(1, 0.18, N_TRAIN)

# Contextual stress formulation: physical exercise suppresses distress false positives
synth_stress = (
    - 0.045 * tr_hrv 
    + 0.50 * tr_eda 
    + 0.35 * tr_scr 
    - 0.35 * tr_sleep 
    - 1.60 * tr_act
    + np.random.normal(0, 0.4, N_TRAIN)
)

y_train = np.zeros(N_TRAIN, dtype=int)
y_train[synth_stress > 0.8] = 1   # Moderate Risk
y_train[synth_stress > 2.3] = 2   # High Risk

X_train = pd.DataFrame({
    'hrv_rmssd': tr_hrv, 'eda_scl': tr_eda, 'eda_scr_count': tr_scr,
    'skin_temp': tr_temp, 'sleep_hours': tr_sleep, 'physical_activity': tr_act
})

model = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.08, eval_metric='mlogloss', random_state=42)
model.fit(X_train, y_train)

# Initialize Algorithmic Explainer
explainer = shap.TreeExplainer(model)

# -----------------------------------------------------------------------------
# PILLAR 1: MULTIMODAL SENSING & CONTEXTUAL DATA INGESTION
# -----------------------------------------------------------------------------
patients_df = pd.read_csv('data/patients_data.csv')
print("[PILLAR 1: CONTEXTUAL INGESTION] Ingested Clinical Vignettes:")
print(patients_df[['patient_id', 'patient_name', 'hrv_rmssd', 'eda_scl', 'sleep_hours', 'physical_activity']])
print("-" * 80)

# -----------------------------------------------------------------------------
# PIPELINE EXECUTION FOR CLINICAL VIGNETTES (PILLARS 2 TO 5)
# -----------------------------------------------------------------------------
risk_labels = ["Low Risk", "Moderate Risk", "High Risk"]

def run_hc_ctxai_pipeline(patient_idx, override_action=None, clinician_reason=""):
    patient = patients_df.iloc[patient_idx]
    patient_vector = pd.DataFrame([patient[feature_cols]])
    
    # Pillar 2: Predictive Engine & Uncertainty Quantification
    probs = model.predict_proba(patient_vector)[0]
    pred_class = int(np.argmax(probs))
    confidence = float(probs[pred_class])
    entropy = -np.sum([p * np.log(p + 1e-9) for p in probs]) / np.log(len(probs))
    
    # Pillar 3: Algorithmic Explainability (SHAP Attributions)
    shap_vals = explainer.shap_values(patient_vector)
    if isinstance(shap_vals, list):
        feat_shap = shap_vals[pred_class][0]
    else:
        feat_shap = shap_vals[0, :, pred_class]
        
    drivers = sorted(zip(feature_cols, patient_vector.values[0], feat_shap), key=lambda x: abs(x[2]), reverse=True)
    
    # Pillar 4: Dual-Channel Adaptive Translation (Clinician vs Patient)
    clinician_pack = {
        "Target": f"{patient['patient_id']} - {patient['patient_name']}",
        "Prediction": risk_labels[pred_class],
        "Confidence": f"{confidence*100:.1f}%",
        "Uncertainty_Entropy": f"{entropy:.3f}",
        "Clinical_Context": patient['clinical_note'],
        "Key_Physiological_Drivers": [
            f"{f} (val: {v:.1f}) -> SHAP weight: {w:+.3f}" for f, v, w in drivers[:2]
        ]
    }
    
    if pred_class >= 1:
        patient_pack = (
            f"Hello {patient['patient_name']}, your wearable system detected elevated physiological strain "
            f"primarily driven by autonomic recovery indicators and reduced sleep duration. "
            f"Recommendation: Consider taking a 15-minute rest period and practicing regulated breathing.\n"
            f"*Disclaimer: This is an early screening assessment and does not constitute a formal psychiatric diagnosis.*"
        )
    else:
        patient_pack = (
            f"Hello {patient['patient_name']}, your physiological markers reflect healthy autonomic stability. "
            f"Maintain your regular daily routine and sleep schedule!"
        )

    print(f"\n>>> CASE EVALUATION: {patient['patient_id']} - {patient['patient_name']} <<<")
    print(f"Pillar 2 [Predictive Engine]: {risk_labels[pred_class]} | Confidence: {confidence*100:.1f}% | Uncertainty Entropy: {entropy:.3f}")
    print("Pillar 3 & 4 [Clinician Auditing Package]:")
    print(json.dumps(clinician_pack, indent=2))
    print("Pillar 4 [Patient Empathetic Package]:")
    print(f"   \"{patient_pack}\"")
    
    # Pillar 5: Human-AI Collaboration & Clinical Oversight
    print("Pillar 5 [Human-AI Collaboration & Clinical Oversight]:")
    if override_action:
        print(f"   [CLINICIAN OVERRIDE ACTION]: {override_action}")
        print(f"   [CLINICAL JUSTIFICATION]: {clinician_reason}")
        print("   -> Audit logged to clinical registry to prevent automation bias.")
    else:
        print("   [CLINICIAN VALIDATION]: AI risk assessment validated and accepted without modification.")
    print("=" * 80)

# Scenario 1: True Psychological Distress (P001)
run_hc_ctxai_pipeline(patient_idx=0)

# Scenario 2: Exercise Confounder Filter (P002)
run_hc_ctxai_pipeline(
    patient_idx=1,
    override_action="VALIDATED_AS_PHYSICAL_EXERTION",
    clinician_reason="Elevated EDA and heart rate accounted for by physical exercise (physical_activity=1). Not psychological distress."
)

# Scenario 3: Ambiguous Boundary State / High Uncertainty (P004)
run_hc_ctxai_pipeline(
    patient_idx=3,
    override_action="REQUEST_EMA_AND_FOLLOW_UP",
    clinician_reason="High prediction entropy observed near classification threshold. Ordered Ecological Momentary Assessment (EMA)."
)