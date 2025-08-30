# Full GenAI Developer Assistant (Steps 1 to 16 Integrated)
import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from git import Repo, GitCommandError
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from openai import OpenAI
import numpy as np
import time
import random
import pandas
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from scipy.stats import weibull_min
import json

# --- NEW IMPORTS for Weibull Plotting ---
import matplotlib.pyplot as plt
import statsmodels.api as sm
from reliability.Fitters import Fit_Weibull_2P # Using reliability library for easier plotting

# Load .env for OpenAI key
load_dotenv()
# Set OpenAI API key from environment variables
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# --- Initial App Setup ---
st.set_page_config(page_title="AI Assistant", layout="wide")
st.title("🧠 **AI - Assistant**")
st.markdown(
    "A multi-agent AI assistant for various tasks, from ticket analysis to remote diagnostics."
)

# Initialize the OpenAI LLM and Client
try:
    llm = ChatOpenAI(model="gpt-4o-mini")
    openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    openai_initialized = True
except Exception as e:
    st.error(f"❌ Error initializing OpenAI: {e}. Please check your API key.")
    openai_initialized = False
    llm = None
    openai_client = None

# --- UPDATED: Weibull Reliability Agent ---
def run_weibull_analysis_agent(failure_times, censored_times, current_op_hours, machine_id):
    """
    Agent 4: Performs Weibull analysis and generates a probability plot.
    """
    try:
        # Ensure data is numpy array of floats
        failure_times = np.asarray(failure_times, dtype=float)
        censored_times = np.asarray(censored_times, dtype=float)

        if len(failure_times) < 2:
             return {
                "analysis_success": False,
                "error_message": "Weibull analysis requires at least 2 failure data points."
            }


        # Fit the Weibull distribution using the reliability library
        fitter = Fit_Weibull_2P(failures=failure_times, right_censored=censored_times)
        
        beta_shape = fitter.beta
        eta_scale = fitter.alpha

        # --- Generate Weibull Probability Plot ---
        fig, ax = plt.subplots(figsize=(8, 6))
        fitter.probability_plot(ax=ax)
        ax.set_title(f'Weibull Probability Plot for {machine_id}')
        ax.grid(True)

        # Save plot to an in-memory buffer
        buf = BytesIO()
        plt.savefig(buf, format="png", bbox_inches='tight')
        buf.seek(0)
        plot_image = buf.getvalue()
        plt.close(fig) # Close the figure to free memory

        # --- Interpretation of the failure mode based on beta ---
        if beta_shape < 1.0:
            failure_mode = 'infant_mortality'
            interpretation_text = f"Infant Mortality (β = {beta_shape:.2f} < 1): The component is failing early, suggesting issues with manufacturing, installation, or burn-in."
        elif np.isclose(beta_shape, 1.0, atol=0.15):
            failure_mode = 'useful_life'
            interpretation_text = f"Useful Life (β = {beta_shape:.2f} ≈ 1): Failures are random and constant, as expected during the component's normal operating life."
        else: # beta_shape > 1.0
            failure_mode = 'wear_out'
            interpretation_text = f"Wear-Out (β = {beta_shape:.2f} > 1): The failure rate is increasing, indicating the component is aging and nearing the end of its life."

        # --- Prediction of failure probability in the next 30 days (720 hours) ---
        t = current_op_hours
        delta_t = 30 * 24 # 30 days in hours
        
        # We calculate the conditional probability: P(T < t+Δt | T > t) = (F(t+Δt) - F(t)) / (1 - F(t))
        cdf_t = fitter.CDF(t)
        cdf_t_plus_delta = fitter.CDF(t + delta_t)
        survival_t = 1 - cdf_t
        
        if survival_t > 1e-9: # Avoid division by zero
            prob_failure_next_30_days = (cdf_t_plus_delta - cdf_t) / survival_t
        else:
            prob_failure_next_30_days = 1.0 # If survival probability is zero, failure is certain

        # --- Recommendation ---
        if prob_failure_next_30_days > 0.05:
            recommended_action = "Generate maintenance alert for component replacement. The predicted failure probability exceeds the 5% threshold."
        else:
            recommended_action = "No immediate action required. Continue monitoring the component."

        # --- Construct the final output ---
        output = {
            "analysis_success": True,
            "machine_id": machine_id,
            "weibull_parameters": {
                "beta_shape_parameter": beta_shape,
                "eta_scale_parameter": eta_scale
            },
            "failure_mode": failure_mode,
            "interpretation": interpretation_text,
            "predicted_30_day_failure_probability": prob_failure_next_30_days,
            "recommended_action": recommended_action,
            "weibull_plot": plot_image # <-- ADDED THE PLOT IMAGE
        }

    except Exception as e:
        output = {
            "analysis_success": False,
            "error_message": str(e)
        }
    
    return output

# --- NEW: UI Function for Weibull Analysis ---
def run_weibull_agent_ui():
    st.header("⚙️ Weibull Reliability Agent")
    st.info("Upload your equipment failure data to perform a Weibull analysis and predict future reliability.")

    if "weibull_analysis_results" not in st.session_state:
        st.session_state.weibull_analysis_results = None

    uploaded_file = st.file_uploader(
        "Upload Excel or CSV file with failure/censored times", type=["xlsx", "xls", "csv"], key="weibull_uploader"
    )

    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.dataframe(df.head())
            
            col1, col2 = st.columns(2)
            with col1:
                failure_cols = st.multiselect("Select column(s) with **Failure Times** (hours to failure)", options=df.columns)
            with col2:
                censored_cols = st.multiselect("Select column(s) with **Censored Times** (hours run without failure)", options=df.columns)

            st.markdown("---")
            col3, col4 = st.columns(2)
            with col3:
                 machine_id = st.text_input("Enter Machine/Component ID", "Pump-12A")
            with col4:
                 current_op_hours = st.number_input("Enter Current Operating Hours", min_value=0, value=5000)

            if st.button("🔬 Run Weibull Analysis"):
                failure_times = pd.concat([df[col] for col in failure_cols], ignore_index=True).dropna().tolist()
                censored_times = pd.concat([df[col] for col in censored_cols], ignore_index=True).dropna().tolist()
                
                if not failure_times:
                    st.error("Please select at least one failure time column with data.")
                else:
                    with st.spinner("Performing Weibull analysis and generating plot..."):
                        results = run_weibull_analysis_agent(failure_times, censored_times, current_op_hours, machine_id)
                        st.session_state.weibull_analysis_results = results
        
        except Exception as e:
            st.error(f"Error processing file: {e}")

    if st.session_state.weibull_analysis_results:
        results = st.session_state.weibull_analysis_results
        st.markdown("---")
        st.subheader("📊 Weibull Analysis Results")
        
        if results["analysis_success"]:
            col1, col2 = st.columns([1, 1])
            with col1:
                st.image(results['weibull_plot'], caption=f"Weibull Plot for {results['machine_id']}")
            
            with col2:
                # Display key metrics
                st.metric(
                    label="Failure Probability (Next 30 Days)",
                    value=f"{results['predicted_30_day_failure_probability']:.2%}"
                )
                st.metric(label="β (Shape Parameter)", value=f"{results['weibull_parameters']['beta_shape_parameter']:.2f}")
                st.metric(label="η (Scale Parameter / Characteristic Life)", value=f"{results['weibull_parameters']['eta_scale_parameter']:.0f} hours")
                
                # Display interpretation and recommendation
                st.write(f"**Failure Mode:** {results['failure_mode'].replace('_', ' ').title()}")
                st.info(f"**Interpretation:** {results['interpretation']}")
                st.warning(f"**Recommendation:** {results['recommended_action']}")
            
        else:
            st.error(f"Analysis Failed: {results['error_message']}")

# --- CNC AI Agent Functions ---
def run_cnc_expert_agent(llm, cnc_data_string):
    """Agent 2: Analyzes CNC data for anomalies, provides solutions, and predicts failures."""
    prompt = PromptTemplate.from_template(
        """
You are a world-class CNC Machine diagnostician and maintenance expert with 30 years of experience in high-volume automotive manufacturing.
Your task is to analyze the following dataset from a CNC machine. The data contains various sensor readings over time.
Your analysis must be thorough and actionable for a shop floor manager to prevent downtime.

**Dataset to Analyze:**
---
{cnc_data}
---

**Instructions:**
1.  **Identify Anomalies:** Carefully examine the data for any unusual patterns, spikes, drops, or correlations between different metrics (e.g., temperature vs. vibration, spindle speed vs. tool wear). List each anomaly you find.
2.  **Determine Root Cause:** For each anomaly, provide a detailed explanation of the most likely root cause. Be
