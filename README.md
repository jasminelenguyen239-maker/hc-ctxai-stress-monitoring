\# HC-CTXAI: Human-Centered Context-Aware Explainable AI Framework



An open-source prototype implementing the \*\*HC-CTXAI\*\* framework for collaborative early stress risk assessment in IoT-enabled remote mental healthcare, grounded in the \*\*Design Science Research (DSR)\*\* paradigm.



\---



\## 🏛️ Architectural Overview: The 5 Core Pillars



| Pillar | Focus | Implementation |

| :--- | :--- | :--- |

| \*\*Pillar 1\*\* | Multimodal Contextual Ingestion | Fusion of HRV, EDA, Skin Temp, Sleep metrics \& Physical Activity |

| \*\*Pillar 2\*\* | Predictive Engine \& Uncertainty | XGBoost Multi-Class Classification with Shannon Entropy calibration |

| \*\*Pillar 3\*\* | Algorithmic Explainability | Game-theoretic TreeSHAP feature attributions |

| \*\*Pillar 4\*\* | Dual-Channel Translation | Tailored Clinician Technical Audit vs. Patient Empathetic Guidance |

| \*\*Pillar 5\*\* | Human-AI Teaming \& Audit Trail | Clinician validation, override controls, and automation-bias logging |



\---



\## 📊 Visual Framework Artifacts



\### 1. Algorithmic Feature Explanations (Pillar 3)

Local feature attributions demonstrating how contextual factors (e.g., Sleep Duration) and autonomic biomarkers (EDA/HRV) drive predictions:



!\[SHAP Explanations](assets/shap\_attributions.png)



\### 2. Uncertainty Quantification \& Clinical Oversight Boundary (Pillars 2 \& 5)

When prediction entropy exceeds the predefined safety boundary, the framework escalates the case for clinician manual review:



!\[Uncertainty Calibration](assets/uncertainty\_calibration.png)



\---



\## 📂 Repository Structure



```text

├── assets/

│   ├── shap\_attributions.png          # Visual explanation artifacts

│   └── uncertainty\_calibration.png    # Uncertainty threshold visualizations

├── data/

│   └── patients\_data.csv              # Standardized clinical scenario vignettes

├── demo\_hc\_ctxai.py                   # Automated 5-pillar terminal script

├── generate\_visuals.py                # Visual generation module

├── requirements.txt                   # Computational dependencies

└── README.md                          # Technical report \& faculty overview

