# A-Causal-and-Interpretable-Machine-Learning-Framework-for-Post-Cranioplasty-Complications

This repository contains the source code for training and evaluating machine learning models for predicting postoperative complications after cranioplasty.

> ⚠️ Note: This repository contains **code only**. Patient data are not included due to privacy restrictions.

---
## ⚙️ Environment Setup

- Python version: `>=3.8`

Install dependencies:
```bash
pip install -r requirements.txt
```
# 🧠 Model Overview
•	Task: Binary classification — predict presence or absence of postoperative complications   
•	Algorithms supported: RandomForest, XGBoost, etc(15 total models)  
•	Input: Structured clinical and intraoperative variables(excel files)  
•	Output: Trained model files(.pkl)  
# 🚀 Usage Instructions
1. Model Training (training_validation/models/)  
Prepare your excel training data. The format should include:  
•	Clinical features (e.g., age, sex, BMI, operative time, material)  
•	Target label: complication (0 = no, 1 = yes)  
Run:
```bash
python train/xxx.py 
```
The script performs:   
•	Model training   
•	Model saving (model.pkl)  
2. Model Evaluation (置信区间/)
To apply the trained model to a separate external dataset, run:
python eval/evaluate_model.py --input your_external_data.csv --model model.pkl
Outputs:
•	Predicted probabilities
•	Performance metrics (AUC, Accuracy, Brier score)
•	Optional plots (ROC curve, calibration curve, etc.) saved to eval/output/
Expected external input format is described in eval/external_data_note.md.
________________________________________
🔒 Data Handling
•	This repository does not include any clinical or patient-level data.
•	All code is structured to ensure clear separation between training and external validation.
•	No test data are used in model training or hyperparameter tuning.
________________________________________
📄 License
This project is licensed under the MIT License. See LICENSE for more details.
________________________________________
📬 Contact
Your Name
Department of Neurosurgery, XYZ Hospital
📧 your.email@institution.edu
________________________________________
For academic and research purposes only. No patient data are stored in this repository.

