"""
HC-CTXAI Research Demonstration Visualizer
Generates Academic Visual Artifacts for GitHub & Faculty Presentation
"""

import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs('assets', exist_ok=True)

# 1. VISUALIZATION: Algorithmic Explainability (SHAP Top Feature Drivers)
features = ['Sleep Duration', 'EDA Tonic (SCL)', 'HRV (RMSSD)', 'EDA Phasic (SCR)', 'Physical Activity', 'Skin Temp']
shap_impact = [-0.62, +0.54, -0.48, +0.31, -0.25, +0.08]  # Values for Case P001 (Distress)
colors = ['crimson' if x > 0 else 'royalblue' for x in shap_impact]

plt.figure(figsize=(9, 4.5))
bars = plt.barh(features, shap_impact, color=colors, edgecolor='black', alpha=0.85)
plt.axvline(0, color='gray', linestyle='--', linewidth=0.8)
plt.title("Pillar 3: Algorithmic Feature Attributions (SHAP) - Case P001", fontsize=12, fontweight='bold')
plt.xlabel("SHAP Attribution Weight (Impact on Distress Risk)")
plt.grid(axis='x', linestyle=':', alpha=0.6)

for bar in bars:
    w = bar.get_width()
    plt.text(w + (0.02 if w >= 0 else -0.06), bar.get_y() + bar.get_height()/4, f"{w:+.2f}", fontweight='bold')

plt.tight_layout()
plt.savefig('assets/shap_attributions.png', dpi=300)
plt.close()

# 2. VISUALIZATION: Human-AI Collaboration Audit Flow (Pillars 2 & 5)
cases = ['P001\n(Distress)', 'P002\n(Exercise)', 'P003\n(Baseline)', 'P004\n(Boundary)']
ai_confidence = [0.89, 0.78, 0.94, 0.52]
uncertainty_entropy = [0.24, 0.38, 0.12, 0.91]

x = np.arange(len(cases))
width = 0.35

fig, ax1 = plt.subplots(figsize=(8, 4))
rects1 = ax1.bar(x - width/2, ai_confidence, width, label='AI Confidence', color='#2b5c8f', edgecolor='black')
ax1.set_ylabel('Model Confidence Score (0 to 1.0)', color='#2b5c8f', fontweight='bold')
ax1.set_ylim(0, 1.1)

ax2 = ax1.twinx()
rects2 = ax2.bar(x + width/2, uncertainty_entropy, width, label='Uncertainty Entropy', color='#d95f02', edgecolor='black')
ax2.set_ylabel('Shannon Entropy Index (Uncertainty)', color='#d95f02', fontweight='bold')
ax2.set_ylim(0, 1.1)
ax2.axhline(0.6, color='red', linestyle='--', label='Uncertainty Threshold (Pillar 5 Trigger)')

ax1.set_xticks(x)
ax1.set_xticklabels(cases, fontweight='bold')
plt.title("Pillar 2 & 5: Prediction Confidence vs. Boundary Uncertainty Threshold", fontsize=12, fontweight='bold')
fig.tight_layout()
plt.savefig('assets/uncertainty_calibration.png', dpi=300)
plt.close()

print("-> Visual artifacts successfully generated in './assets/' directory.")