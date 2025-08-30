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

# --- NEW: CNC AI Agent Functions ---
def run_cnc_expert_agent(llm, cnc_data_string):
    """Agent 2: Analyzes CNC data for anomalies and provides solutions."""
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
1.  **Identify Anomalies:** Carefully examine the data for any unusual patterns, spikes, drops, or correlations between different metrics for devicename, locations, tagname, the position or velocity error on the axes, the reference velocity or another reference value for the axes and others like e.g temperature vs. vibration, spindle speed vs. tool wear). List each anomaly you find.
2.  **Determine Root Cause:** For each anomaly, provide a detailed explanation of the most likely root cause. Be specific. For example, instead of "machine is vibrating," say "A sudden spike in Y-axis vibration concurrent with a rise in spindle temperature suggests a worn spindle bearing or an unbalanced tool."
3.  **Provide Detailed Fixes:** For each identified issue, provide a clear, step-by-step Standard Operating Procedure (SOP) that a maintenance technician can follow to fix the problem. Include required tools and estimated time for the fix.
4.  **Structure Your Report:** Format your response using markdown with the following sections:
    - **Overall Machine Health Summary**
    - **Anomaly 1: [Name of Anomaly]**
        - **Description:**
        - **Potential Root Cause:**
        - **Recommended Action Plan (SOP):**
    - **Anomaly 2: [Name of Anomaly]** (If applicable)
        - ... and so on.
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
3.  **Labor Cost Savings:** Estimate the reactive labor hours that would have been needed to diagnose and fix a full breakdown (this is typically much higher than proactive maintenance). Calculate the cost savings using the formula: `(Reactive Repair Hours - Proactive Repair Hours from report) * £100/Hour`. Be realistic in your estimation of reactive hours.

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

    # # New section for PDF export
    #    st.markdown("---")
    #    st.markdown("### Export Report")

    #    # Generate the PDF file in memory
    #    pdf_buffer = BytesIO()
    #    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
    #    styles = getSampleStyleSheet()

    #    # Add custom styles if needed
    #    styles.add(ParagraphStyle(name='Justify', alignment=TA_CENTER))

    #    story = []
    #    story.append(Paragraph("CNC AI Agent - Final Report", styles['Title']))
    #    story.append(Spacer(1, 12))

    #    # Add analysis content
    #    story.append(Paragraph("Agent 2: CNC Expert Analysis", styles['Heading2']))
    #    story.append(Spacer(1, 12))
    #    # Replace markdown bold with HTML for reportlab
    #    analysis_html = st.session_state.cnc_analysis.replace("**", "<b>")
    #    story.append(Paragraph(analysis_html, styles['Normal']))
    #    story.append(Spacer(1, 12))

    #    # Add benefits content
    #    story.append(Paragraph("Agent 3: Business Impact Report", styles['Heading2']))
    #    story.append(Spacer(1, 12))
    #    benefits_html = st.session_state.cnc_benefits.replace("**", "<b>")
    #    story.append(Paragraph(benefits_html, styles['Normal']))

    #    doc.build(story)
    #    pdf_buffer.seek(0)

    #    # Create a download button for the generated PDF
    #    st.download_button(
    #        label="⬇️ Download Report as PDF",
    #        data=pdf_buffer,
    #        file_name="cnc_analysis_report.pdf",
    #        mime="application/pdf"
    #    )


# ===============================================================================

# --- Streamlit UI and Logic ---
step = st.sidebar.radio(
    "**Available Agents:**",
    ["1. CNC AI Agent"],
)

# Initialize session state variables
if "vectordb" not in st.session_state:
    st.session_state.vectordb = None
if "few_shot" not in st.session_state:
    st.session_state.few_shot = ""
if "ticket" not in st.session_state:
    st.session_state.ticket = ""
if "generated_code" not in st.session_state:
    st.session_state.generated_code = ""

if step == "1. CNC AI Agent":
    st.subheader("🛠️ Litmus EDGE Agent: CNC Machine Diagnostics")
    st.markdown(
        "This agent looks at your CNC Machine data captured from Litmus EDGE and predicts failure."
    )
    if llm:
        run_cnc_ai_agent(llm)
    else:
        st.error("❌ OpenAI is not initialized. Please check your API key.")
