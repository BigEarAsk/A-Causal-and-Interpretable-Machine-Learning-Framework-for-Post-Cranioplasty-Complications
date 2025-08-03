# A-Causal-and-Interpretable-Machine-Learning-Framework-for-Post-Cranioplasty-Complications

This repository contains the source code for training and evaluating machine learning models for predicting postoperative complications after cranioplasty.

> Note: This repository contains **code only**. Patient data are not included due to privacy restrictions.

---
## Environment Setup

- Python version: `>=3.8`

Install dependencies:
```bash
pip install -r requirements.txt
```
# Model Overview
•	Task: Binary classification — predict presence or absence of postoperative complications   
•	Algorithms supported: RandomForest, XGBoost, etc(15 total models)  
•	Input: Structured clinical and intraoperative variables(excel files)  
•	Output: Trained model files(.pkl)  
# Usage Instructions
1. Model Training (training_validation/models/)  
Prepare your excel training data. The format should include:  
•	Clinical features (e.g., age, sex, surgery time, material)  
•	Target label: complication (0 = no, 1 = yes)  
Run:
```bash
python train/xxx.py 
```
The script performs:   
•	Model training   
•	Model saving (model.pkl)  
2. Model Evaluation (eval/)  
To apply the trained model to a separate external dataset, run:  
```bash
python eval/get_confidence.py
```
```bash
python eval/get_confidence_no_smote.py
```
Outputs:  
•	Performance metrics excel file(AUC, Accuracy, Brier score,etc)  
# Data Handling
•	All code is structured to ensure clear separation between training and external validation.  

For academic and research purposes only. No patient data are stored in this repository.  
Other codes for data preprocessing and visualization will be uploaded soon after the paper is accepted.
