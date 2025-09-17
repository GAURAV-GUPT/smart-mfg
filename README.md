### Full GenAI Developer Assistant

**Full GenAI Developer Assistant** is a multi-agent AI application designed to streamline various developer and operational tasks.  Built with Python, Streamlit, and the LangChain framework, this tool integrates several specialized agents to handle tasks ranging from automated diagnostics to business impact analysis.

-----

### Key Features

  * **Multi-Agent Architecture:** The application uses a modular design with specialized agents, including a CNC AI Agent, a Weibull Reliability Agent, and a Business Impact Agent, to perform specific tasks.
  * **CNC Machine Diagnostics:** It analyzes CNC machine data from a user-uploaded Excel file, identifies anomalies, determines root causes, and provides step-by-step Standard Operating Procedures (SOPs) for maintenance.
  * **Predictive Maintenance:** The CNC Agent and the dedicated **Weibull Reliability Agent** use statistical analysis to predict future machine failures. This helps users transition from reactive to proactive maintenance.
  * **Business Value Analysis:** The Business Impact Agent quantifies the financial benefits of proactive maintenance by estimating avoided downtime, calculating productivity savings, and measuring labor cost reductions.
  * **Intuitive UI:** The web-based interface, powered by Streamlit, makes it easy for users to upload data and view detailed reports.

-----

### Agents

This application comprises a series of interconnected agents that work together to provide a comprehensive solution.

| Agent Name | Description |
| :--- | :--- |
| **Agent 1: Data Ingestion** | Handles the uploading and display of Excel files containing CNC machine data. |
| **Agent 2: CNC Expert Agent** | An LLM-powered agent that analyzes CNC data for anomalies and provides expert-level diagnostic reports. |
| **Agent 3: Business Impact Agent** | Quantifies the financial value of the expert agent's analysis by calculating cost savings and productivity gains. |
| **Agent 4: Weibull Reliability Agent** | A specialized programmatic agent that performs precise Weibull statistical analysis to predict equipment failure probabilities. |

-----

### Technologies

  * **Python:** The core programming language.
  * **Streamlit:** Used to create the interactive and user-friendly web application interface.
  * **LangChain:** Provides the framework for building and orchestrating the multi-agent system.
  * **OpenAI GPT-4o-mini:** The large language model (LLM) that powers the analytical and conversational agents.
  * **`scipy.stats.weibull_min`:** A scientific Python library used for the precise statistical calculations in the Weibull Reliability Agent.
  * **`pandas`:** For efficient data manipulation and analysis of the uploaded Excel files.
  * **`python-dotenv`:** Manages environment variables, ensuring secure API key handling.

-----

### How to Run Locally

#### **Prerequisites**

Ensure you have Python installed on your system.

#### **1. Clone the Repository**

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

#### **2. Set Up Your Environment**

Create a virtual environment to manage dependencies.

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

#### **3. Install Dependencies**

Install all the required packages from the `requirements.txt` file (you will need to create this file yourself if it's not present).

```bash
pip install -r requirements.txt
```

#### **4. Configure Your OpenAI API Key**

Create a `.env` file in the root directory of the project and add your OpenAI API key.

```ini
OPENAI_API_KEY="your_openai_api_key_here"
```

#### **5. Run the Application**

Launch the Streamlit application from your terminal.

```bash
streamlit run your_app_file_name.py
```

*(Note: Replace `your_app_file_name.py` with the actual name of your Python script.)*

-----

### Contribution

Feel free to open issues or submit pull requests to improve the functionality of this application.

### License

This project is licensed under the MIT License - see the `LICENSE` file for details.
