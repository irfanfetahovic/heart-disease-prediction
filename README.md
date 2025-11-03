# ❤️ Heart Disease Prediction Dashboard

![Python](https://img.shields.io/badge/Python-3.10-blue)
![License](https://img.shields.io/badge/License-MIT-green)


**ML project for predicting heart disease using patient health data.**  
This project demonstrates a complete machine learning workflow: data exploration, feature engineering, multiple model training and tuning, model evaluation, predictions and interactive deployment via Streamlit.



## Motivation

Early detection of heart disease can significantly reduce health risks and treatment costs.
This project demonstrates how data-driven machine learning techniques can support medical decision-making and risk prediction.



## Features

- **Exploratory Data Analysis (EDA)**:  
  - Interactive visualizations of distributions, correlations, and crosstabs.  
  - Plots for heart disease by sex, age vs cholesterol, chest pain types, and more.  

- **Data Preprocessing & Feature Engineering**:  
  - Handling missing values and categorical encoding.  
  - Standard scaling, one-hot encoding, and feature type conversions.  
  - Duplicate removal and optional outlier handling.  

- **Modeling & Evaluation**:  
  - Multiple algorithms: Logistic Regression, KNN, Random Forest, XGBoost, and Neural Network.  
  - Hyperparameter tuning using **Optuna** for Logistic Regression and XGBoost.  
  - Cross-validation for unbiased model performance estimates.  
  - Evaluation metrics: Accuracy, F1, Precision, Recall, Confusion Matrix, ROC & AUC.  
  - Feature importance visualization with SHAP and LIME for interpretable models.  

- **Interactive Prediction App**:  
  - Deploys the best-performing model (Logistic Regression) in **Streamlit**.  
  - Users can input patient data or upload CSVs for batch predictions.   
  - Displays model information and evaluation
  - Visualizes model metrics, ROC curve, and SHAP summary and force plots.



## Workflow Overview

### Data Cleaning & Preprocessing
- Simple encoding of categorical features using mapping
- Converting some numerical columns into categorical, based on their values
- Removing duplicate rows  
- Handling missing values and scaling numerical columns
- One-hot encoding for categorical variables  

### Exploratory Data Analysis (EDA)
- Feature distributions, correlations, and crosstabs  
- Visualizations for age, cholesterol, sex, and chest pain type  

### Model Training & Comparison
- Models used: Logistic Regression, KNN, Random Forest, XGBoost, Neural Network  
- Cross-validation to assess model performance  
- Hyperparameter tuning using **Optuna**  

### Best Model Selection & Evaluation
- Final model evaluation on unseen test set  
- Metrics: Accuracy, F1, Precision, Recall, Confusion Matrix, ROC & AUC  
- Model explainability and feature importance analysis using SHAP and LIME
- Model artifacts saved with **joblib** for reusability

### Deployment
- Streamlit dashboard for interactive predictions, EDA, model evaluation and explainability 



## Technologies Used

- **Python & Libraries**: pandas, NumPy, scikit-learn, XGBoost, TensorFlow/Keras, Optuna, seaborn, matplotlib, shap, lime  
- **Web App**: Streamlit  
- **Model Storage**: joblib for saving models, encoders, and scalers  
- **Deployment**: Streamlit Cloud or local execution



## Dataset

- **Source**: [UCI Heart Disease Classification Dataset (Kaggle)](https://www.kaggle.com/datasets/sumaiyatasmeem/heart-disease-classification-dataset)
- **Samples**: 303 
- **Features**: 13 (Age, Sex, Chest Pain Type, Resting Blood Pressure, Cholesterol, Resting ECG and more) 
- **Target**: Heart Disease Diagnosis (Yes/No)



## Demo
An interactive Streamlit app for presenting EDA, model evaluation and explainability, and live prediction.

[Streamlit App](https://irfanfetahovic-ecommerce-data-analysis-codedashboard-m1y05z.streamlit.app/)



## How to Run
1. Clone the repository:

   ```bash
   git clone https://github.com/irfanfetahovic/heart-disease-prediction.git
   cd heart-disease-prediction
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Download the dataset from Kaggle and place it in the data/ folder.
4. Open and run the Jupyter Notebook:

   ```bash
   jupyter notebook notebooks/heart_disease_prediction.ipynb
   ```
5. Run all cells to explore data, visualizations, and insights
6. Run the Streamlit dashboard:

   ```bash
   streamlit run app/app.py
   ```
 


## Project Structure

```bash
/heart-disease-prediction
├── data/                    # dataset
├── notebooks/               # Jupyter notebooks
├── app/                     # Streamlit app
├── models/                  # saved models, encoders and scalers
├── README.md
└── requirements.txt
```


## License
This project is licensed under the MIT License.



👤 **Author:** Irfan Fetahović  
📧 **Email:** [irfanfetahovic@gmail.com](mailto:irfanfetahovic@gmail.com)  
💼 **Portfolio:** [GitHub](https://github.com/irfanfetahovic) | [LinkedIn](https://www.linkedin.com/in/irfan-fetahovic-28473923/)