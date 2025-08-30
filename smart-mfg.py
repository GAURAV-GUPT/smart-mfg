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
from scipy.stats import weibull_min # <-- New import for Weibull analysis
import json # <-- New import for pretty printing dicts

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

# --- NEW: Weibull Reliability Agent ---
def run_weibull_analysis_agent(failure_times, censored_times, current_op_hours, machine_id):
    """
    Agent 4: Performs Weibull analysis on equipment failure data.
    This agent is a programmatic agent using scipy, not an LLM.
    """
    try:
        # Combine failure and censored data for fitting
        all_data = np.concatenate([failure_times, censored_times])
        
        # Create a boolean array indicating uncensored (failures) and censored data
        # True for a failure, False for censored data
        is_uncensored = np.concatenate([
            np.ones_like(failure_times, dtype=bool),
            np.zeros_like(censored_times, dtype=bool)
        ])

        # Fit the Weibull distribution using right-censored data
        # floc=0 fixes the location parameter at 0, which is standard for lifetime analysis
        shape, loc, scale = weibull_min.fit(all_data, is_uncensored, floc=0)
        
        beta_shape = shape
        eta_scale = scale

        # Step 2: Interpretation of the failure mode based on beta
        if beta_shape < 1.0:
            failure_mode = 'infant_mortality'
            interpretation_text = f"Infant Mortality (β = {beta_shape:.2f} < 1): The component is failing early, suggesting issues with manufacturing, installation, or burn-in."
        elif np.isclose(beta_shape, 1.0, atol=0.1):
            failure_mode = 'useful_life'
            interpretation_text = f"Useful Life (β = {beta_shape:.2f} ≈ 1): Failures are random and constant, as expected during the component's normal operating life."
        else: # beta_shape > 1.0
            failure_mode = 'wear_out'
            interpretation_text = f"Wear-Out (β = {beta_shape:.2f} > 1): The failure rate is increasing, indicating the component is aging and nearing the end of its life."

        # Step 3: Prediction of failure probability in the next 30 days (720 hours)
        t = current_op_hours
        delta_t = 30 * 24 # 30 days in hours
        
        # Weibull CDF: F(t) = 1 - exp(-(t/η)^β)
        # We calculate the conditional probability: P(T < t+Δt | T > t) = (F(t+Δt) - F(t)) / (1 - F(t))
        cdf_t = weibull_min.cdf(t, c=beta_shape, scale=eta_scale)
        cdf_t_plus_delta = weibull_min.cdf(t + delta_t, c=beta_shape, scale=eta_scale)
        
        # The survival function S(t) is 1 - F(t)
        survival_t = 1 - cdf_t
        
        if survival_t > 0:
            prob_failure_next_30_days = (cdf_t_plus_delta - cdf_t) / survival_t
        else:
            prob_failure_next_30_days = 1.0 # If survival probability is zero, failure is certain

        # Recommendation based on a 5% threshold
        if prob_failure_next_30_days > 0.05:
            recommended_action = "Generate maintenance alert for component replacement. The predicted failure probability exceeds the 5% threshold."
        else:
            recommended_action = "No immediate action required. Continue monitoring the component."

        # Construct the final output
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
            "recommended_action": recommended_action
        }

    except Exception as e:
        output = {
            "analysis_success": False,
            "error_message": str(e)
        }
    
    return output

def run_weibull_ui():
    """UI for the Weibull Failure Prediction Agent."""
    st.header("⚙️ Weibull Failure Prediction Agent")
    st.info("This agent analyzes historical data to predict equipment failures using the Weibull distribution.")

    with st.expander("ℹ️ How to use this agent", expanded=False):
        st.markdown("""
        1.  **Machine ID**: Enter a unique identifier for the machine you are analyzing.
        2.  **Failure Data**: Provide a comma-separated list of the operating hours at which the component failed (e.g., `1200, 1550, 2100`). These are **uncensored** data points.
        3.  **Censored Data**: Provide a comma-separated list of operating hours for components that are *still in service* and have not yet failed (e.g., `1800, 2500`). These are **right-censored** data points.
        4.  **Current Operating Hours**: Enter the total hours the machine has been running to calculate the future failure probability from this point.
        5.  Click **Analyze Failure Data** to run the analysis.
        """)

    machine_id = st.text_input("Enter Machine ID", "CNC-001")
    failure_data_str = st.text_area("Failure Data (Operating Hours, comma-separated)", "850, 1020, 1400, 1850, 2200, 3000")
    censored_data_str = st.text_area("Censored Data (Operating Hours of units still running, comma-separated)", "2500, 2800")
    current_hours = st.number_input("Current Operating Hours for Prediction", min_value=0, value=3000)

    if st.button("📈 Analyze Failure Data"):
        try:
            # Parse input strings into lists of floats
            failure_times = [float(x.strip()) for x in failure_data_str.split(',') if x.strip()]
            censored_times = [float(x.strip()) for x in censored_data_str.split(',') if x.strip()]

            if not failure_times:
                st.error("❌ Failure Data cannot be empty.")
            else:
                with st.spinner("Reliability Engineer Agent is performing Weibull analysis..."):
                    result = run_weibull_analysis_agent(failure_times, censored_times, current_hours, machine_id)

                    if result["analysis_success"]:
                        st.success("✅ Weibull Analysis Complete!")
                        
                        st.subheader("Results")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(
                                label="β (Shape Parameter)", 
                                value=f"{result['weibull_parameters']['beta_shape_parameter']:.2f}"
                            )
                        with col2:
                            st.metric(
                                label="η (Scale Parameter / Characteristic Life)", 
                                value=f"{result['weibull_parameters']['eta_scale_parameter']:.0f} hours"
                            )
                        
                        st.subheader("Interpretation")
                        st.markdown(f"**Failure Mode:** `{result['failure_mode']}`")
                        st.write(result['interpretation'])

                        st.subheader("Prediction & Recommendation")
                        prob = result['predicted_30_day_failure_probability']
                        st.metric(
                            label="Predicted Failure Probability (Next 30 Days)",
                            value=f"{prob:.2%}"
                        )
                        st.warning(f"**Recommended Action:** {result['recommended_action']}")

                        st.subheader("Full Analysis Output")
                        st.json(result)

                    else:
                        st.error(f"Analysis Failed: {result['error_message']}")

        except ValueError:
            st.error("❌ Invalid input. Please ensure all data points are numbers separated by commas.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

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
2.  **Determine Root Cause:** For each anomaly, provide a detailed explanation of the most likely root cause. Be specific. For example, instead of "machine is vibrating," say "A sudden spike in Y-axis vibration concurrent with a rise in spindle temperature suggests a worn spindle bearing or an unbalanced tool."
3.  **Provide Detailed Fixes:** For each identified issue, provide a clear, step-by-step Standard Operating Procedure (SOP) that a maintenance technician can follow to fix the problem. Include required tools and estimated time for the fix.
4.  **Perform Failure Prediction (Weibull Analysis):** Based on the pattern of anomalies you identified (which can be treated as potential failure events), perform a conceptual Weibull analysis to predict future failures.
    * **Estimate Weibull Parameters:** Based on whether the anomalies are occurring more frequently over time, randomly, or less frequently, provide a hypothesized shape parameter (β) and an estimated scale parameter (η, characteristic life in hours).
    * **Interpret Failure Mode:** State whether the machine is likely in **infant mortality (β < 1)**, **useful life (β ≈ 1)**, or **wear-out (β > 1)**, and justify your choice based on the data.
    * **Predict 30-Day Failure Probability:** Estimate the probability of a critical failure within the next 30 days of operation.
    * **Recommend Predictive Action:** Based on the predicted probability, give a clear recommendation (e.g., 'Generate maintenance alert' or 'No immediate action required').

**Structure Your Report:**
Format your response using markdown with the following sections:
    - **Overall Machine Health Summary**
    - **Anomaly 1: [Name of Anomaly]**
        - **Description:**
        - **Potential Root Cause:**
        - **Recommended Action Plan (SOP):**
    - **Anomaly 2: [Name of Anomaly]** (If applicable)
        - ... and so on.
    - **Failure Prediction Analysis**
        - **Weibull Parameter Estimates:** (Your estimated β and η)
        - **Failure Mode Interpretation:** (Your analysis of the failure mode)
        - **30-Day Failure Probability:** (Your calculated probability)
        - **Recommendation:** (Your final predictive action)
"""
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(cnc_data=cnc_data_string)

def run_business_impact_agent(llm, analysis_report):
    """Agent 3: Calculates the business benefits of the proactive fix."""
    prompt = PromptTemplate.from_template(
        """
You are a business operations analyst for a major automotive plant.
Your plant's productivity is **40 cars per hour**. The average hourly labor cost for a reactive maintenance team (2 people) is **£100 per hour**.

You have been given a proactive maintenance report from a CNC expert AI. Your task is to quantify the business benefits of implementing the recommended fixes *before* a machine failure occurs.

**CNC Expert's Analysis Report:**
---
{analysis_report}
---

**Instructions:**
Based on the severity and nature of the issues described in the report, calculate the following:

1.  **Downtime Avoidance (in hours):** Estimate the total potential plant floor downtime (in hours) that would have occurred if these issues led to a full machine breakdown. Justify your estimation based on the report.
2.  **Productivity Savings:** Calculate the number of cars that will not be lost due to this avoided downtime. Use the formula: `Avoided Downtime Hours * 40 Cars/Hour`.
3.  **Labor Cost Savings:** Estimate the reactive labor hours that would have been needed to diagnose and fix a a full breakdown (this is typically much higher than proactive maintenance). Calculate the cost savings using the formula: `(Reactive Repair Hours - Proactive Repair Hours from report) * £100/Hour`. Be realistic in your estimation of reactive hours.
4.  **Weibull Failure Prediction** Perform Weibull analysis on equipment failure data on uploaded data

**Output Format:**
Provide a concise summary using markdown.

### Proactive Maintenance - Business Value Report

- **Estimated Downtime Avoided:**
- **Productivity Impact:**
- **Labor Cost Savings:**
- **Overall Summary:** (A brief concluding sentence on the value of this proactive analysis).
- **% of accuracy of the findings**
"""
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(analysis_report=analysis_report)


def run_cnc_ai_agent(llm):
    """Main orchestrator for the CNC AI Agent workflow."""
    st.header("⚙️ CNC AI Agent")

    # Agent 1: File Upload and Display
    st.markdown("### Agent 1: Data Ingestion")
    st.info("Please upload the CNC machine data as an Excel file.")

    uploaded_file = st.file_uploader(
        "Upload Excel file", type=["xlsx", "xls"], key="cnc_uploader"
    )

    # Initialize session state
    if "cnc_df" not in st.session_state:
        st.session_state.cnc_df = None
    if "cnc_analysis" not in st.session_state:
        st.session_state.cnc_analysis = None
    if "cnc_benefits" not in st.session_state:
        st.session_state.cnc_benefits = None
    if "last_file_name" not in st.session_state:
        st.session_state.last_file_name = ""

    if uploaded_file:
        # If a new file is uploaded, reset the analysis
        if uploaded_file.name != st.session_state.last_file_name:
            st.session_state.cnc_df = None
            st.session_state.cnc_analysis = None
            st.session_state.cnc_benefits = None
            st.session_state.last_file_name = uploaded_file.name

        try:
            df = pd.read_excel(uploaded_file)
            st.session_state.cnc_df = df
            st.dataframe(df)
        except Exception as e:
            st.error(f"Error reading the Excel file: {e}")
            st.session_state.cnc_df = None

    if st.session_state.cnc_df is not None:
        if st.button("🔍 Analyze CNC Data"):
            # Agent 2: Expert Analysis
            with st.spinner("Agent 2 (CNC Expert) is analyzing the data..."):

                # data_string = st.session_state.cnc_df.to_string() # Old line
                # New line - sends a statistical summary
                # data_string = st.session_state.cnc_df.describe().to_string()
                data_string = st.session_state.cnc_df.sample(
                    n=100, random_state=1
                ).to_string()
                analysis = run_cnc_expert_agent(llm, data_string)
                st.session_state.cnc_analysis = analysis

            # Agent 3: Business Impact
            with st.spinner("Agent 3 (Business Impact) is calculating the benefits..."):
                benefits = run_business_impact_agent(llm, st.session_state.cnc_analysis)
                st.session_state.cnc_benefits = benefits

    if st.session_state.cnc_analysis:
        st.markdown("---")
        st.markdown("### Agent 2: CNC Expert Analysis")
        st.markdown(st.session_state.cnc_analysis)

    if st.session_state.cnc_benefits:
        st.markdown("---")
        st.markdown("### Agent 3: Business Impact Report")
        st.markdown(st.session_state.cnc_benefits)
        st.success("✅ CNC Analysis Workflow Complete.")

# ===============================================================================

# --- Streamlit UI and Logic ---
step = st.sidebar.radio(
    "**Available Agents:**",
    ["1. CNC AI Agent")]

# Initialize session state variables
if "vectordb" not in st.session_state:
    st.session_state.vectordb = None
if "few_shot" not in st.session_state:
    st.session_state.few_shot = ""
if "ticket" not in st.session_state:
    st.session_state.ticket = ""
if "generated_code" not in st.session_state:
    st.session_state.generated_code = ""

if not openai_initialized:
    st.error("❌ OpenAI is not initialized. Please check your API key and refresh the page.")
else:
    if step == "1. CNC AI Agent":
        st.subheader("🛠️ Litmus EDGE Agent: CNC Machine Diagnostics")
        st.markdown(
            "This agent looks at your CNC Machine data captured from Litmus EDGE and predicts failure."
        )
        run_cnc_ai_agent(llm)
    
    # --- NEW: Logic to run the Weibull agent UI ---
    elif step == "2. Weibull Failure Prediction Agent":
        st.subheader("🔧 Reliability Engineer Agent: Failure Prediction")
        st.markdown(
            "This agent uses Weibull analysis to determine failure modes and predict future failures."
        )
        run_weibull_ui()
