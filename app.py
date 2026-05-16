import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import joblib

# Import modular utilities
from utils.preprocessing import full_preprocessing_pipeline
from utils.visualization import (
    plot_survival_distribution, plot_age_distribution, plot_fare_distribution,
    plot_gender_vs_survival, plot_pclass_vs_survival, plot_embarked_vs_survival,
    plot_correlation_heatmap
)
from utils.prediction import train_and_evaluate_models, load_model_and_predict

# Config
st.set_page_config(page_title="AI Titanic Predictor", page_icon="🚢", layout="wide", initial_sidebar_state="expanded")

# Load CSS
def load_css():
    if os.path.exists("assets/styles.css"):
        with open("assets/styles.css") as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css()

# Sidebar Navigation
st.sidebar.title("🚢 AI Titanic Predictor")
st.sidebar.markdown("Advanced Machine Learning Dashboard")
menu = st.sidebar.radio("Navigation", ["Dataset & EDA", "Model Training & Comparison", "Real-time Prediction", "AI Insights"])

# Theme switcher
theme = st.sidebar.selectbox("Theme", ["Dark Mode", "Light Mode"])
if theme == "Light Mode":
    st.markdown("""<style>[data-testid="stAppViewContainer"] {background-color: #FFFFFF; color: #000000;}</style>""", unsafe_allow_html=True)

# Helper function to load data
@st.cache_data
def load_data(path_or_file):
    return pd.read_csv(path_or_file)

# Dataset State
if 'data' not in st.session_state:
    if os.path.exists('dataset/train.csv'):
        st.session_state['data'] = load_data('dataset/train.csv')
    else:
        st.session_state['data'] = None

if 'processed_data' not in st.session_state:
    st.session_state['processed_data'] = None

# ----------------- DATASET & EDA -----------------
if menu == "Dataset & EDA":
    st.title("📊 Dataset & Exploratory Data Analysis")
    
    # File Uploader
    st.sidebar.header("Upload Data")
    uploaded_file = st.sidebar.file_uploader("Upload Titanic CSV", type=['csv'])
    if uploaded_file:
        st.session_state['data'] = load_data(uploaded_file)
        # Process automatically on upload
        df, enc = full_preprocessing_pipeline(st.session_state['data'])
        st.session_state['processed_data'] = df
        st.success("Dataset loaded and preprocessed automatically!")

    if st.session_state['data'] is not None:
        raw_df = st.session_state['data']
        
        # Tabs
        tab1, tab2, tab3 = st.tabs(["Data Overview", "Visualizations", "Preprocessing"])
        
        with tab1:
            st.subheader("Raw Dataset Preview")
            
            # Data Filtering / Search
            search_col = st.selectbox("Search by Column", raw_df.columns)
            search_term = st.text_input("Search Term")
            if search_term:
                filtered_df = raw_df[raw_df[search_col].astype(str).str.contains(search_term, case=False, na=False)]
                st.dataframe(filtered_df)
            else:
                st.dataframe(raw_df.head(15))
                
            st.markdown(f"**Total Rows:** {raw_df.shape[0]} | **Total Columns:** {raw_df.shape[1]}")
            
        with tab2:
            st.subheader("Titanic Survival Prediction Analysis Dashboard")
            
            # Top metrics
            total = len(raw_df)
            survived = raw_df['Survived'].sum() if 'Survived' in raw_df.columns else 0
            deaths = total - survived
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Passengers", total)
            m2.metric("No. of Deaths", deaths)
            m3.metric("No. of Survival", survived)
            
            # Middle row
            st.markdown("---")
            c1, c2, c3 = st.columns(3)
            fig_pclass = plot_pclass_vs_survival(raw_df)
            fig_pie = plot_survival_distribution(raw_df) 
            fig_gender = plot_gender_vs_survival(raw_df)
            
            if fig_pclass: c1.plotly_chart(fig_pclass, use_container_width=True)
            if fig_pie: c2.plotly_chart(fig_pie, use_container_width=True)
            if fig_gender: c3.plotly_chart(fig_gender, use_container_width=True)
            
            # Bottom row
            st.markdown("---")
            c4, c5 = st.columns(2)
            fig_embarked = plot_embarked_vs_survival(raw_df)
            fig_age = plot_age_distribution(raw_df)
            
            if fig_embarked: c4.plotly_chart(fig_embarked, use_container_width=True)
            if fig_age: c5.plotly_chart(fig_age, use_container_width=True)
                
            st.subheader("Feature Correlation")
            # For correlation we need numeric data
            num_df, _ = full_preprocessing_pipeline(raw_df, save_encoders=False)
            st.plotly_chart(plot_correlation_heatmap(num_df), use_container_width=True)
            
        with tab3:
            st.subheader("Data Preprocessing Pipeline")
            st.write("Applying Modular Preprocessing: Imputation -> Outlier Handling -> Encoding -> Scaling")
            if st.button("Run Preprocessing Pipeline"):
                with st.spinner("Processing data..."):
                    df, enc = full_preprocessing_pipeline(raw_df)
                    st.session_state['processed_data'] = df
                    st.success("Pipeline executed successfully!")
                    
            if st.session_state['processed_data'] is not None:
                st.write("Processed Data Preview:")
                st.dataframe(st.session_state['processed_data'].head())
    else:
        st.info("Please upload a dataset or ensure 'dataset/train.csv' exists.")

# ----------------- MODEL TRAINING -----------------
elif menu == "Model Training & Comparison":
    st.title("🤖 ML Model Training & Comparison")
    
    if st.session_state['processed_data'] is not None:
        df = st.session_state['processed_data']
        
        if 'Survived' not in df.columns:
            st.error("Target column 'Survived' missing from dataset.")
        else:
            if st.button("Train Multiple Models"):
                with st.spinner("Training Logistic Regression, Random Forest, Decision Tree, KNN, and SVM..."):
                    X = df.drop('Survived', axis=1)
                    y = df['Survived']
                    
                    results_df, best_name, best_model, cm, cr = train_and_evaluate_models(X, y)
                    
                    st.session_state['model_results'] = results_df
                    st.session_state['best_model'] = best_model
                    st.session_state['best_name'] = best_name
                    st.session_state['cm'] = cm
                    st.session_state['cr'] = cr
                    
                    st.success(f"Training Complete! The best model is **{best_name}**")
                    
            if 'model_results' in st.session_state:
                st.subheader("Model Comparison Dashboard")
                st.dataframe(st.session_state['model_results'].style.background_gradient(subset=['F1 Score', 'Accuracy'], cmap='viridis'))
                
                tab1, tab2, tab3 = st.tabs(["Best Model Evaluation", "Confusion Matrix", "Download Model"])
                
                with tab1:
                    st.write(f"### Detailed Report: {st.session_state['best_name']}")
                    cr_df = pd.DataFrame(st.session_state['cr']).transpose()
                    st.dataframe(cr_df)
                    
                with tab2:
                    import plotly.figure_factory as ff
                    z = st.session_state['cm']
                    x = ['Predicted Dead', 'Predicted Survived']
                    y = ['Actual Dead', 'Actual Survived']
                    fig = ff.create_annotated_heatmap(z, x=x, y=y, colorscale='Blues')
                    fig.update_layout(title="Confusion Matrix")
                    st.plotly_chart(fig)
                    
                with tab3:
                    if os.path.exists('model/trained_model.pkl'):
                        with open('model/trained_model.pkl', 'rb') as f:
                            st.download_button("📥 Download Trained Model (.pkl)", data=f, file_name=f"{st.session_state['best_name'].replace(' ', '_')}.pkl")
    else:
        st.warning("Please run the Preprocessing Pipeline in the 'Dataset & EDA' tab first.")

# ----------------- REAL-TIME PREDICTION -----------------
elif menu == "Real-time Prediction":
    st.title("🔮 Real-time Prediction System")
    
    if os.path.exists('model/trained_model.pkl'):
        st.markdown("Enter passenger details below to predict survival probability using the trained AI model.")
        
        with st.form("predict_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                age = st.number_input("Age", 0.0, 100.0, 30.0)
                pclass = st.selectbox("Passenger Class", [1, 2, 3])
                sibsp = st.number_input("Siblings/Spouses", 0, 10, 0)
            with col2:
                fare = st.number_input("Fare", 0.0, 600.0, 32.0)
                sex = st.selectbox("Gender", ["male", "female"])
                parch = st.number_input("Parents/Children", 0, 10, 0)
            with col3:
                embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"])
                
            submit = st.form_submit_button("Launch Prediction Analysis")
            
        if submit:
            input_df = pd.DataFrame([[pclass, sex, age, sibsp, parch, fare, embarked]],
                                    columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'])
            
            with st.spinner("Analyzing neural pathways..."):
                pred, prob = load_model_and_predict(input_df)
                
            if pred is not None:
                st.markdown("---")
                st.subheader("Prediction Results")
                
                # Animated Result Card
                if pred == 1:
                    st.markdown(f"""
                    <div class="prediction-card survived">
                        <h1 style="color:white !important;">🎉 SURVIVED</h1>
                        <h3 style="color:white !important;">Confidence: {prob[1]*100:.2f}%</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    st.balloons()
                else:
                    st.markdown(f"""
                    <div class="prediction-card not-survived">
                        <h1 style="color:white !important;">💀 NOT SURVIVED</h1>
                        <h3 style="color:white !important;">Confidence: {prob[0]*100:.2f}%</h3>
                        <p style="color:white !important;">High Risk Indicator</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                # Download PDF Report
                from utils.report_generator import generate_prediction_pdf
                from utils.visualization import plot_survival_distribution, plot_gender_vs_survival
                
                passenger_details = {
                    'Age': age, 'Sex': sex, 'Pclass': pclass, 'Fare': fare,
                    'SibSp': sibsp, 'Parch': parch, 'Embarked': embarked
                }
                
                figures = []
                if st.session_state['data'] is not None:
                    raw_df = st.session_state['data']
                    fig1 = plot_survival_distribution(raw_df)
                    fig2 = plot_gender_vs_survival(raw_df)
                    if fig1: figures.append(fig1)
                    if fig2: figures.append(fig2)
                
                with st.spinner("Generating PDF Report..."):
                    try:
                        pdf_bytes = generate_prediction_pdf(passenger_details, pred, max(prob)*100, figures)
                        if pdf_bytes:
                            st.download_button("📥 Download Advanced PDF Report", data=pdf_bytes, file_name="Titanic_Prediction_Report.pdf", mime="application/pdf")
                        else:
                            st.error("Failed to generate PDF content.")
                    except Exception as e:
                        st.error(f"Could not generate PDF. Ensure kaleido is installed: {e}")
            else:
                st.error("Prediction failed. Ensure the model and encoders are properly trained.")
    else:
        st.warning("No trained model found. Please go to the 'Model Training' tab and train the models first.")

# ----------------- AI INSIGHTS -----------------
elif menu == "AI Insights":
    st.title("🧠 AI Insights & Feature Importance")
    
    if os.path.exists('model/trained_model.pkl') and os.path.exists('model/features.pkl'):
        model = joblib.load('model/trained_model.pkl')
        features = joblib.load('model/features.pkl')
        
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                
                fig = px.bar(x=importances, y=features, orientation='h', 
                             title="Feature Importance generated by AI Model",
                             color=importances, color_continuous_scale="Viridis")
                fig.update_layout(template='plotly_dark')
                st.plotly_chart(fig, use_container_width=True)
                
                st.info("Insight: Features with higher values had the greatest impact on whether the AI predicted survival or death.")
            else:
                st.info(f"The best model selected ({type(model).__name__}) does not support extracting feature importances natively.")
        except Exception as e:
            st.error("Could not load feature importances.")
    else:
        st.warning("Train a model first to view AI Insights.")
