import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
import os
import io

# Load pre-trained model
@st.cache_data
def load_model():
    return joblib.load('xgb_model.joblib')

model = load_model()

# Custom CSS for styling and responsiveness (without dark mode)
st.markdown(
    """
    <style>
    .main {
        background-color: #f0f4f8;
        color: #333;
        font-family: 'Roboto', sans-serif;
        font-weight: 300;
        transition: all 0.3s ease;
    }
    .header {
        background: linear-gradient(90deg, #007BFF, #0056b3);
        color: white;
        padding: 20px;
        text-align: center;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .card {
        background: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .metric {
        background-color: #007BFF;
        color: white;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
    }
    .upload-area {
        border: 2px dashed #007BFF;
        padding: 20px;
        text-align: center;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    /* Responsive adjustments */
    @media (max-width: 600px) {
        .card, .upload-area {
            padding: 10px;
            margin: 10px 0;
        }
        .header {
            padding: 15px;
        }
        .metric {
            font-size: 14px;
            padding: 8px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar navigation
page = st.sidebar.selectbox("Navigate", ["Early Warning System Dashboard", "Individual Student Risk Assessment"])

# Page 1: Early Warning System Dashboard
if page == "Early Warning System Dashboard":
    st.markdown('<div class="header"><h1>Early Warning System Dashboard</h1></div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload a CSV file (e.g., data.csv)", type="csv", key="file_uploader", help="Upload a semicolon-separated CSV with a 'Target' column")

    if uploaded_file is not None:
        # Read and prepare data
        data = pd.read_csv(uploaded_file, sep=';')
        target_mapping = {'Dropout': 0, 'Graduate': 1, 'Enrolled': 2}
        y = data['Target'].map(target_mapping)
        X = data.drop('Target', axis=1)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Predict
        xgb_pred = model.predict(X_test)
        
        # Calculate F1-score
        f1_xgb = f1_score(y_test, xgb_pred, average='weighted')
        st.markdown(f'<div class="card"><div class="metric">XGBoost F1-Score: {f1_xgb:.2f}</div></div>', unsafe_allow_html=True)
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, xgb_pred)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=['Dropout', 'Graduate', 'Enrolled'], 
                    yticklabels=['Dropout', 'Graduate', 'Enrolled'], cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix')
        st.pyplot(fig)
        st.write("This confusion matrix shows the model's performance across 'Dropout', 'Graduate', and 'Enrolled' categories. The diagonal values represent correct predictions, while off-diagonal values indicate misclassifications, helping assess the model's accuracy and potential areas for improvement.")
        
        # Feature Importance
        st.markdown('<div class="card"><h3>Top 5 Risk Indicators:</h3><ul>', unsafe_allow_html=True)
        st.markdown('<li><strong>2nd Semester Units Approved</strong>: Low approval rates indicate significant disengagement.</li>', unsafe_allow_html=True)
        st.markdown('<li><strong>1st Semester Units Approved</strong>: Early academic struggles are a key predictor.</li>', unsafe_allow_html=True)
        st.markdown('<li><strong>Tuition Fees Paid</strong>: Financial issues correlate with higher dropout likelihood.</li>', unsafe_allow_html=True)
        st.markdown('<li><strong>2nd Semester Evaluations</strong>: Fewer evaluations suggest missed opportunities.</li>', unsafe_allow_html=True)
        st.markdown('<li><strong>Course</strong>: Specific course challenges may impact retention.</li>', unsafe_allow_html=True)
        st.markdown('</ul></div>', unsafe_allow_html=True)
        
        # SHAP Summary
        st.markdown('<div class="card"><h3>SHAP Summary</h3>', unsafe_allow_html=True)
        if os.path.exists('shap.png'):
            st.image("shap.png", caption="SHAP Summary Plot", use_container_width=True)
        else:
            st.write("SHAP plot not available.")
        st.write("This SHAP plot highlights the most influential features in predicting student outcomes, such as '2nd Semester Units Approved' and 'Age at Enrollment'. Higher SHAP values indicate stronger impact on the model's decision, helping identify key risk factors for dropout or success.")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # LIME Example
        st.markdown('<div class="card"><h3>LIME Summary</h3>', unsafe_allow_html=True)
        if os.path.exists('lime analysis.png'):
            st.image("lime analysis.png", caption="LIME Explanation", use_container_width=True)
        else:
            st.write("LIME plot not available.")
        st.write("This LIME analysis provides a local explanation for a single prediction, showing how features like '1st Semester Units Approved' and 'Tuition Fees Paid' contributed to classifying a student as 'Dropout' or 'Graduate'. It offers insight into individual case decisions.")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Download option with dropout indicators
        dropout_indicators = {
            "Major Indicators of Dropout Risk": [
                "2nd Semester Units Approved: Low approval rates indicate significant disengagement.",
                "1st Semester Units Approved: Early academic struggles are a key predictor.",
                "Tuition Fees Paid: Financial issues correlate with higher dropout likelihood.",
                "2nd Semester Evaluations: Fewer evaluations suggest missed opportunities.",
                "Course: Specific course challenges may impact retention."
            ]
        }
        df_indicators = pd.DataFrame(dropout_indicators)
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df_indicators.to_excel(writer, sheet_name='Dropout_Indicators', index=False)
        output.seek(0)
        st.download_button(
            label="Download Dropout Risk Indicators",
            data=output,
            file_name="dropout_indicators.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
        st.markdown('<div class="upload-area">Please upload a CSV file to analyze.</div>', unsafe_allow_html=True)

# Page 2: Individual Student Risk Assessment
elif page == "Individual Student Risk Assessment":
    st.markdown('<div class="header"><h2>Individual Student Risk Assessment</h2></div>', unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        manual_input = {}
        manual_input['Marital status'] = 1  # Default: Single
        manual_input['Application mode'] = 1  # Default: 1st phase
        manual_input['Application order'] = 1  # Default: 1st choice
        manual_input['Course'] = 1  # Default: Course 1
        manual_input['Daytime/evening attendance\t'] = 1  # Default: Daytime
        manual_input['Previous qualification'] = 1  # Default: Secondary education
        manual_input['Previous qualification (grade)'] = 120  # Default: Average grade
        manual_input['Nacionality'] = 1  # Default: Portuguese
        manual_input["Mother's qualification"] = 1  # Default: Secondary education
        manual_input["Father's qualification"] = 1  # Default: Secondary education
        manual_input["Mother's occupation"] = 1  # Default: Service
        manual_input["Father's occupation"] = 1  # Default: Service
        manual_input['Admission grade'] = st.number_input("Admission Grade", min_value=95, max_value=190, value=120)
        manual_input['Displaced'] = 0  # Default: Not displaced
        manual_input['Educational special needs'] = 0  # Default: No special needs
        manual_input['Debtor'] = 0  # Default: Not debtor
        manual_input['Tuition fees up to date'] = st.selectbox("Tuition Fees Paid", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0], index=0)[1]
        manual_input['Gender'] = 0  # Default: Male
        manual_input['Scholarship holder'] = 0  # Default: No scholarship
        manual_input['Age at enrollment'] = st.number_input("Age at Enrollment", min_value=17, max_value=70, value=20)
        manual_input['International'] = 0  # Default: Not international
        manual_input['Curricular units 1st sem (credited)'] = 0  # Default: None credited
        manual_input['Curricular units 1st sem (enrolled)'] = 6  # Default: Average enrollment
        manual_input['Curricular units 1st sem (evaluations)'] = 6  # Default: Average evaluations
        manual_input['Curricular units 1st sem (approved)'] = st.number_input("1st Semester Units Approved", min_value=0, max_value=26, value=5)
        manual_input['Curricular units 1st sem (grade)'] = 12  # Default: Average grade
        manual_input['Curricular units 1st sem (without evaluations)'] = 0  # Default: None
        manual_input['Curricular units 2nd sem (credited)'] = 0  # Default: None credited
        manual_input['Curricular units 2nd sem (enrolled)'] = 6  # Default: Average enrollment
        manual_input['Curricular units 2nd sem (evaluations)'] = st.number_input("2nd Semester Evaluations", min_value=0, max_value=26, value=5)
        manual_input['Curricular units 2nd sem (approved)'] = st.number_input("2nd Semester Units Approved", min_value=0, max_value=26, value=4)
        manual_input['Curricular units 2nd sem (grade)'] = 12  # Default: Average grade
        manual_input['Curricular units 2nd sem (without evaluations)'] = 0  # Default: None
        manual_input['Unemployment rate'] = 10.8  # Default: Average from dataset
        manual_input['Inflation rate'] = 2.3  # Default: Average from dataset
        manual_input['GDP'] = -0.4  # Default: Average from dataset

        feature_order = ['Marital status', 'Application mode', 'Application order', 'Course', 'Daytime/evening attendance\t', 'Previous qualification', 'Previous qualification (grade)', 'Nacionality', "Mother's qualification", "Father's qualification", "Mother's occupation", "Father's occupation", 'Admission grade', 'Displaced', 'Educational special needs', 'Debtor', 'Tuition fees up to date', 'Gender', 'Scholarship holder', 'Age at enrollment', 'International', 'Curricular units 1st sem (credited)', 'Curricular units 1st sem (enrolled)', 'Curricular units 1st sem (evaluations)', 'Curricular units 1st sem (approved)', 'Curricular units 1st sem (grade)', 'Curricular units 1st sem (without evaluations)', 'Curricular units 2nd sem (credited)', 'Curricular units 2nd sem (enrolled)', 'Curricular units 2nd sem (evaluations)', 'Curricular units 2nd sem (approved)', 'Curricular units 2nd sem (grade)', 'Curricular units 2nd sem (without evaluations)', 'Unemployment rate', 'Inflation rate', 'GDP']

        if st.button("Assess Risk"):
            manual_df = pd.DataFrame([manual_input], columns=feature_order)
            prediction = model.predict(manual_df)
            prediction_proba = model.predict_proba(manual_df)
            if prediction[0] == 0:  # Dropout
                risk_label = "At Risk"
                risk_prob = prediction_proba[0][0] * 100
            else:  # Graduate or Enrolled
                risk_label = "Not At Risk"
                risk_prob = max(prediction_proba[0][1], prediction_proba[0][2]) * 100  # Highest non-Dropout probability
            st.write(f"**Prediction:** The student is predicted to be {risk_label} with {risk_prob:.1f}% confidence.")
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div style="text-align:center; padding:10px; color:#0056b3;">© 2025 Boyo Oritsedere</div>', unsafe_allow_html=True)