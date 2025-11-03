import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Heart Disease Prediction Dashboard", layout="wide")
st.title("❤️ Heart Disease Prediction Dashboard")


# Data paths
base_path = os.path.dirname(__file__)
data_path = os.path.join(base_path, "../data")
models_path = os.path.join(base_path, "../models")


st.markdown("""
This interactive dashboard lets you:
1. Explore the dataset (EDA) used for building prediction model
2. View evaluation metrics of the selected (best) model
3. Use selected model to predict heart disease for new data
""")

# --- LOAD MODEL
@st.cache_resource
def load_artifacts():  
    model = joblib.load(os.path.join(models_path, "model.joblib"))
    imputer = joblib.load(os.path.join(models_path, "imputer.joblib"))
    imputer_cat = joblib.load(os.path.join(models_path, "imputer_cat.joblib"))
    sc = joblib.load(os.path.join(models_path, "sc.joblib"))
    ct = joblib.load(os.path.join(models_path, "ct.joblib"))
    le = joblib.load(os.path.join(models_path, "le.joblib"))
    return model, imputer, imputer_cat, sc, ct, le

model, imputer, imputer_cat, sc, ct, le = load_artifacts()

# --- LOAD DEFAULT DATA ---
@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(data_path, "heart_disease_classification_dataset.csv"), encoding='ISO-8859-1')
    df = df.drop(columns=['Unnamed: 0'], errors='ignore')
    return df

df = load_data()

df['sex'] = df['sex'].map({'female':0, 'male':1})
df['target'] = df['target'].map({'no':0, 'yes':1})


# Separate features and target
X = df.drop("target", axis=1).to_numpy() 
y = df["target"].to_numpy()


# Identify numeric and categorical columns and their indices
dfc = df.drop(columns=['target'], errors='ignore').copy()
num_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
cat_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
# Define column indices
num_cols_idx = dfc.columns.get_indexer(num_cols)
cat_cols_idx = dfc.columns.get_indexer(cat_cols)





# --- SIDEBAR ---
#st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/3/3d/Heart_coronary_arteries-en.svg")
st.sidebar.title("Dashboard Info")
st.sidebar.markdown("Explore the dataset, evaluate the best trained model, and make heart disease predictions using this model.")
st.sidebar.metric("Samples", len(df))
st.sidebar.metric("Features", df.shape[1])
st.sidebar.metric("Positive cases", int(df['target'].sum()))
st.sidebar.info("Note: This model is trained on the UCI Heart Disease dataset and is for educational use only.")




# --- TAB SETUP ---
tab_overview, tab_eda, tab_model, tab_predict = st.tabs(["📊 Overview", "🔍 EDA", "🧠 Model", "🔮 Prediction"])


# =====================
# TAB 1: OVERVIEW
# =====================
with tab_overview:
    st.header("Dataset Overview")
    st.write("Showing first 5 rows of the dataset:")
    st.dataframe(df.head())
    st.write("**Shape:**", df.shape)
    # st.write("**Columns:**", list(df.columns))
    
    st.markdown("---")
    st.subheader("🧩 Feature Descriptions")
    st.markdown("""
1. **age**: The patient's age (in years).  
2. **sex**: The patient's sex (1 = male, 0 = female).  
3. **cp**: Type of chest pain experienced by the patient:
    - 0: Typical angina  
    - 1: Atypical angina  
    - 2: Non-anginal pain  
    - 3: Asymptomatic  
4. **trestbps**: Resting blood pressure (in mmHg). Anything above 130–140 is typically cause for concern.  
5. **chol**: Serum cholesterol level (in mg/dL).  
6. **fbs**: Fasting blood sugar.
    - 1: > 120 mg/dL (possible diabetes)  
    - 0: ≤ 120 mg/dL  
7. **restecg**: Resting electrocardiographic results
    - 0: Normal  
    - 1: ST–T wave abnormality  
    - 2: Left ventricular hypertrophy  
8. **thalach**: Maximum heart rate achieved.  
9. **exang**: Exercise-induced angina.
    - 1: Yes  
    - 0: No  
10. **oldpeak**: ST depression induced by exercise relative to rest.  
11. **slope**: Slope of the peak exercise ST segment.
    - 0: Upsloping — better heart rate with exercise (uncommon)  
    - 1: Flat — minimal change (typical healthy heart)  
    - 2: Downsloping — signs of an unhealthy heart  
12. **ca**: Number of major vessels (0–3) colored by fluoroscopy.  
13. **thal**: Thalassemia type:
    - 1: Normal  
    - 2: Fixed defect  
    - 3: Reversible defect  
14. **target**: Heart disease diagnosis (1 = yes, 0 = no).
""")


    # --- Target distribution plot ---
    st.write("**Target distribution:**")
    st.bar_chart(df["target"].value_counts())


# =====================
# TAB 2: EDA
# =====================
with tab_eda:
    st.header("Exploratory Data Analysis")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    with col2:
        st.subheader("Feature Distributions")
        feature = st.selectbox("Select a feature", df.columns)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(df, x=feature, hue="target", multiple="stack", ax=ax)
        st.pyplot(fig)


    crtab = pd.crosstab(df.target, df.sex)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Heart Disease by Sex")
        fig1, ax1 = plt.subplots(figsize=(5, 4))
        crtab.plot(kind='bar', ax=ax1, color=['#66b3ff', '#ff9999'])
        ax1.set_title('Heart Disease by Sex')
        ax1.set_xlabel('Sex (0=Female, 1=Male)')
        ax1.set_ylabel('Count')
        ax1.legend(['No Disease', 'Disease'])
        fig1.tight_layout()
        st.pyplot(fig1)

    with col4:
        st.subheader("Age vs Cholesterol")
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        sns.scatterplot(x='age', y='chol', hue='target', data=df, ax=ax2, palette=['#66b3ff', '#ff9999'])
        ax2.set_title('Age vs Cholesterol by Heart Disease Status')
        ax2.set_xlabel('Age')
        ax2.set_ylabel('Cholesterol Level')
        ax2.legend(title='Heart Disease', labels=['No Disease', 'Disease'])
        fig2.tight_layout()
        st.pyplot(fig2)




# =====================
# TAB 3: MODEL EVALUATION
# =====================
with tab_model:
    st.header("Model Information and Evaluation")

    # --- MODEL INFORMATION ---
    st.subheader("Model Information")
    st.markdown("""
    **Algorithm:** Logistic Regression  
    **Penalty:** L2  
    **Solver:** sag  
    **Regularization Strength (C):** 0.0457
    """)



    # --- METRICS ---
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, ConfusionMatrixDisplay, RocCurveDisplay, roc_auc_score
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_test[:,num_cols_idx]  = imputer.transform(X_test[:,num_cols_idx])
    X_train[:,num_cols_idx]  = imputer.transform(X_train[:,num_cols_idx])
    if len(cat_cols_idx) > 0:
        X_test[:,cat_cols_idx]  = imputer_cat.transform(X_test[:,cat_cols_idx])
        X_train[:,cat_cols_idx]  = imputer_cat.transform(X_train[:,cat_cols_idx])

    X_test[:,num_cols_idx] = sc.transform(X_test[:,num_cols_idx])
    X_test = ct.transform(X_test)
    X_train[:,num_cols_idx] = sc.transform(X_train[:,num_cols_idx])
    X_train = ct.transform(X_train)
    # Predict using the saved model
    y_pred = model.predict(X_test)
    #y_test  = le.transform(y_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)

    st.subheader("Model Evaluation Metrics")
    st.markdown(f"""
    **Accuracy:** {acc:.3f}  
    **Precision:** {prec:.3f}  
    **Recall:** {rec:.3f}  
    **F1 Score:** {f1:.3f}  
    **ROC AUC Score:** {auc:.3f}
    """)


    # Detailed classification report
    st.markdown("#### Detailed Classification Report")
    st.text(classification_report(y_test, y_pred))

    col1, col2 = st.columns(2)
    with col1:
        # --- CONFUSION MATRIX ---
        st.markdown("#### Confusion Matrix")
        #st.subheader("Confusion Matrix")
        fig, ax = plt.subplots(figsize=(4, 3))
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, cmap="Blues")
        st.pyplot(fig)

    with col2:
        # --- ROC CURVE ---
        st.markdown("#### ROC Curve")
        #st.subheader("ROC Curve")
        fig, ax = plt.subplots(figsize=(4, 3))
        RocCurveDisplay.from_predictions(y_test, model.predict_proba(X_test)[:, 1], ax=ax)
        st.pyplot(fig)
    
    st.subheader("Feature Importance (SHAP Summary Plot)")

        # Get feature names
    num_feature_names = list(num_cols)
    cat_feature_names = list(ct.named_transformers_['onehot'].get_feature_names_out(cat_cols))
    feature_names = cat_feature_names + num_feature_names

    # Convert to DataFrames
    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    X_test_df = pd.DataFrame(X_test, columns=feature_names)

    # Create explainer
    explainer = shap.LinearExplainer(model, X_train_df)
    shap_values = explainer.shap_values(X_test_df)

    # SHAP summary plot
    
    fig, ax = plt.subplots(figsize=(8, 5))
    shap.summary_plot(shap_values, X_test_df, feature_names=feature_names)
    st.pyplot(fig)

# =====================
# TAB 4: PREDICTION
# =====================
with tab_predict:
    st.header("Make Predictions")

    mode = st.radio("Select prediction mode:", ["Single Input", "Upload CSV"])

    if mode == "Upload CSV":
        st.subheader("Upload your dataset (same columns as training data)")
        uploaded_file = st.file_uploader("Choose CSV", type="csv")

        if uploaded_file is not None:
            user_df = pd.read_csv(uploaded_file)
            st.write("✅ Uploaded data preview:")
            st.dataframe(user_df.head())
            
            if not all(col in user_df.columns for col in df.drop("target", axis=1).columns):
                st.error("❌ Uploaded file does not match expected columns.")
                st.stop()
            else:
                # Transform and predict
                X = user_df.to_numpy()
                X[:,num_cols_idx]  = imputer.transform(X[:,num_cols_idx])
                if len(cat_cols_idx) > 0:
                    X[:,cat_cols_idx]  = imputer_cat.transform(X[:,cat_cols_idx])
                
                X[:,num_cols_idx] = sc.transform(X[:,num_cols_idx])
                X = ct.transform(X)
                preds = model.predict(X)
                #preds = le.inverse_transform(model.predict(X))
                user_df["prediction"] = preds
                st.subheader("Predictions:")
                st.dataframe(user_df)

                # Option to download results
                csv = user_df.to_csv(index=False).encode("utf-8")
                st.download_button("Download predictions", csv, "predictions.csv", "text/csv")

    elif mode == "Single Input":
        st.subheader("Enter feature values manually")

        # input_data = {}
        # for col in df.drop("target", axis=1).columns:
        #     col_type = df[col].dtype
        #     if col_type == "object" or df[col].nunique() < 10:
        #         input_data[col] = st.selectbox(col, sorted(df[col].unique()))
        #     else:
        #         input_data[col] = st.number_input(col, float(df[col].min()), float(df[col].max()), float(df[col].mean()))

        # Define feature descriptions
        feature_descriptions = {
            "age": "Age of the person (years)",
            "sex": "Sex (1 = male, 0 = female)",
            "cp": "Chest pain type (0–3)",
            "trestbps": "Resting blood pressure (mm Hg)",
            "chol": "Serum cholesterol (mg/dl)",
            "fbs": "Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)",
            "restecg": "Resting ECG results (0–2)",
            "thalach": "Maximum heart rate achieved",
            "exang": "Exercise induced angina (1 = yes, 0 = no)",
            "oldpeak": "ST depression induced by exercise relative to rest",
            "slope": "Slope of the peak exercise ST segment (0–2)",
            "ca": "Number of major vessels colored by flourosopy (0–4)",
            "thal": "Thalassemia (1 = normal, 2 = fixed defect, 3 = reversible defect)"
        }

        input_data = {}
        for col in df.drop("target", axis=1).columns:
            desc = feature_descriptions.get(col, "")
            col_label = f"{col} — {desc}" if desc else col
            col_type = df[col].dtype

            if col_type == "object" or df[col].nunique() < 10:
                input_data[col] = st.selectbox(col_label, sorted(df[col].unique()))
            else:
                input_data[col] = st.number_input(
                    col_label,
                    float(df[col].min()),
                    float(df[col].max()),
                    float(df[col].mean())
                )

        if st.button("Predict"):
            user_df = pd.DataFrame([input_data])
            X = user_df.to_numpy()
            
            X[:,num_cols_idx]  = imputer.transform(X[:,num_cols_idx])
            if len(cat_cols_idx) > 0:
                X[:,cat_cols_idx]  = imputer_cat.transform(X[:,cat_cols_idx])

            X[:,num_cols_idx] = sc.transform(X[:,num_cols_idx])
            X = ct.transform(X)
            pred = model.predict(X)
            #pred = le.inverse_transform(pred)[0]

            st.success(f"**Prediction:** {'🟥 Heart Disease (1)' if pred == 1 else '🟩 No Heart Disease (0)'}")
            
            # SHAP force plot for one instance (individual explanation)

            st.subheader("Prediction Explanation (SHAP Force Plot)")

            X_df = pd.DataFrame(X, columns=feature_names)     
            shap_values = explainer(X_df)

            fig, ax = plt.subplots(figsize=(10, 5))
            shap.plots.bar(shap_values[0], show=False)  # no feature_names argument
            st.pyplot(fig)

            #If you want explicit probability of 1 for this sample
            pred = model.predict_proba(X_df.iloc[[0]])[0,1]
            st.markdown(f"""**Predicted probability for this sample** {pred:.3f}""")

            # If the plot is not clear, you can list all features and their SHAP values for this sample via a table
            st.markdown("""**Table with SHAP values for this sample**""")
            sample_shap = pd.DataFrame({
                'Feature': X_df.columns,
                'Value': X_df.iloc[0].to_numpy(),
                'SHAP': shap_values[0].values
            })
            # Add a column for absolute SHAP values
            sample_shap['Abs_SHAP'] = sample_shap['SHAP'].abs()
            # Sort by absolute contribution in descending order
            sample_shap_sorted = sample_shap.sort_values(by='Abs_SHAP', ascending=False).drop(columns='Abs_SHAP')

            sample_shap_sorted




st.markdown("<hr><center>Developed by Irfan Fetahovic</center>", unsafe_allow_html=True)
