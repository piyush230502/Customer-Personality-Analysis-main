import streamlit as st
import pickle
import pandas as pd
import numpy as np
from customer_personality.config.configuration import AppConfiguration
from customer_personality.pipeline.training_pipeline import TrainingPipeline
import plotly.express as px

# Set page config
st.set_page_config(
    page_title="Customer Personality Analysis",
    page_icon="👥",
    layout="wide"
)

def load_model():
    config = AppConfiguration()
    model_path = config.get_prediction_config().trained_model_path
    try:
        model = pickle.load(open(model_path, 'rb'))
        return model
    except:
        return None

def main():
    st.title("Customer Personality Analysis 👥")
    
    menu = ["Home", "Train Model", "Predict", "About"]
    choice = st.sidebar.selectbox("Menu", menu)
    
    if choice == "Home":
        st.subheader("Home")
        st.write("""
        This application helps analyze customer personalities based on various features.
        You can:
        - Train a new model
        - Make predictions for customer segments
        - Visualize the results
        """)
        
    elif choice == "Train Model":
        st.subheader("Train Model")
        if st.button("Start Training"):
            with st.spinner("Training in progress..."):
                try:
                    pipeline = TrainingPipeline()
                    pipeline.start_training_pipeline()
                    st.success("Training completed successfully!")
                except Exception as e:
                    st.error(f"Error during training: {str(e)}")
                    
    elif choice == "Predict":
        st.subheader("Predict Customer Segment")
        
        col1, col2 = st.columns(2)
        
        with col1:
            income = st.number_input("Income", min_value=0.0, max_value=200000.0, value=50000.0)
            age = st.number_input("Age", min_value=18, max_value=100, value=30)
            children = st.number_input("Number of Children", min_value=0, max_value=5, value=0)
            
        with col2:
            recency = st.number_input("Recency (days since last purchase)", min_value=0, max_value=365, value=30)
            total_spending = st.number_input("Total Spending", min_value=0, max_value=10000, value=1000)
            month_enrollment = st.number_input("Months of Enrollment", min_value=1, max_value=60, value=12)
            
        if st.button("Predict"):
            model = load_model()
            if model is not None:
                try:
                    data = [income, recency, age, total_spending, children, month_enrollment]
                    prediction = model.predict([data])[0]
                    
                    segment_descriptions = {
                        0: "Budget-conscious Families",
                        1: "Affluent Professionals",
                        2: "Average Spenders",
                        3: "High-Value Regular Customers"
                    }
                    
                    st.success(f"Customer Segment: {segment_descriptions[prediction]}")
                    
                    # Visualization
                    fig = px.scatter(x=[income], y=[total_spending], 
                                   title="Customer Position",
                                   labels={'x': 'Income', 'y': 'Total Spending'})
                    st.plotly_chart(fig)
                    
                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")
            else:
                st.warning("Please train the model first!")
                
    elif choice == "About":
        st.subheader("About")
        st.write("""
        ### Customer Personality Analysis Project
        
        This project helps businesses understand their customers better through segmentation.
        The model uses various customer attributes to group them into meaningful segments.
        
        Features used:
        - Income
        - Age
        - Total Spending
        - Children
        - Recency of Purchase
        - Enrollment Duration
        
        Built with Streamlit, Scikit-learn, and Python.
        """)

if __name__ == '__main__':
    main()