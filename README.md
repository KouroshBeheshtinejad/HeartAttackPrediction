# Heart Attack Prediction using Machine Learning

This project is a **comprehensive machine learning pipeline** for predictiong the likelihood of heart attacks based on patient data. The goal is to develop, evaluate, and compare multiple models to identify the most accurate and reliable predictive algorithm.

## Dataset

The dataset(```heart.csv```) contains **303 patient records** with **13 feature** related to medical and demographic information. The target variable ```output``` is binary:
- No heart attack: ```0```
- Heart attack: ```1```

Features include metrics such as age, sex, cholesterol levels, blood pressure, and more.

## Project Structure

```bash
HeartAttackPrediction/
├── data/                 
│   └── heart.csv
├── models/                
├── plots/                 
├── src/
│   └── train_models.py    
├── .gitignore
├── requirements.txt
└── README.md
```

## Methodology

1. Data Loading & Preprocessing:
- Shuffle dataset for randomness
- Normalize features using ```MinMaxScaler```

2. Model Training:
Multiple models are trained and tuned via ```GridSearchCV``` with **5-fold Stratified Cross-Validation**:
- Logistic Regression
- Support Vector Machine(SVM)
- Decision Tree Classifier
- K-Nearest Neighbors(KNN)

3. Evaluation Metrics:
Each model is evaluated on the test set using:
- **Accuracy**
- **ROC-AUC**(Receiver Operating Charateristic - Area under Curve)
- **PR-AUC**(Precision-Recall - Area Under Curve)
- Confusion Matrix

4. Visualization:
- Correlation heatmap of features
- Confusion matrices
- ROC and Precision-Recall curves
- Comparative bar chart of ROC-AUC & PR-AUC for all models

5. Model Presistence:
- Each trained model is saved using ```joblib``` inside ```models/``` for later deployment or prediction.

## How to Run

1. Clone the repository:
```bash
git clone https://github.com/<your-username>/HeartAttackPrediction.git
cd HeartAttackPrediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the training script:
```bash
python src/train_models.py
```

4. Check generated plots in ```plots/``` and trained models in ```models/```.

## Result

- **SVM** achieved the highest performance with:
 - Test Accuracy ≈ 0.90
 - ROC-AUC ≈ 0.94
 - PR-AUC ≈ 0.95
- Logistic Regression and Decision Tree also performed competitively, while KNN was slightly lower.

This comprehensive workflow allows **easy extension** to new models, features, or datasets, and is structered for deployment or further research.