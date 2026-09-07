# 🤖 Blue Orbit Office Assistant

An AI-powered workplace assistant that helps employees quickly access company information, employee details, leave information, expense records, IT assets, office information, and company policies.

The application combines **LangGraph**, **Qwen3 via Ollama**, deterministic tool routing, and a **RAG-based policy retrieval system**, with a simple **Streamlit** interface.

---

## 🚀 Features

The Blue Orbit Office Assistant can answer questions related to:

### 👤 Employee Information
- Employee details
- Department
- Designation
- Manager information
- Joining information

### 🏖️ Leave Management
- Casual leave balance
- Earned leave balance
- Sick leave balance
- Total leave balance
- Leave policies
- Carry-forward rules
- Questions about additional/insufficient leave

### 💰 Expenses
- Employee expense records
- Total expenses
- Expense summaries
- Approved expenses
- Pending expenses
- Rejected expenses

### 💻 IT Assets
- Assigned laptop
- IT assets assigned to an employee

### 🏢 Office Information
- Office locations
- Office details by city

### 📚 Company Policies
The RAG pipeline retrieves relevant information from company policy documents and uses it to answer policy-related questions.

---

## 🏗️ Architecture

```text
                    ┌─────────────────────┐
                    │    Streamlit UI     │
                    │       app.py        │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Employee Agent    │
                    │  employee_agent.py  │
                    └──────────┬──────────┘
                               │
                    ┌──────────┴──────────┐
                    │ Deterministic Intent│
                    │      Routing        │
                    └──────────┬──────────┘
                               │
             ┌─────────────────┼─────────────────┐
             ▼                 ▼                 ▼
      Employee Tools      Policy RAG        Qwen3 / Ollama
             │                 │                 │
             ▼                 ▼                 ▼
          CSV Data       FAISS Vector DB   Final Response
                               │
                               ▼
                         Policy Documents

🛠️ Tech Stack
Technology	Purpose
Python	Application development
Streamlit	User interface
LangGraph	Agent workflow
Qwen3	Local LLM
Ollama	Local model runtime
FAISS	Vector similarity search
Google Gemini Embeddings	Document embeddings
PyPDF	PDF processing
Pandas	CSV data processing
NumPy	Numerical operations
📁 Project Structure
BlueOrbit-Office-Assistant/
│
├── agent/
│   └── employee_agent.py
│
├── data/
│   ├── csvs/
│   │   ├── employees_with_null.csv
│   │   ├── leave_balance.csv
│   │   ├── expense_records.csv
│   │   ├── IT_assets.csv
│   │   └── office_locations.csv
│   │
│   └── faiss_policy/
│
├── rag/
│   ├── embeddings.py
│   ├── ingestion.py
│   ├── retrieval.py
│   └── vectorstore.py
│
├── tools/
│   └── employee_tools_clean.py
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── .env.example
⚙️ Setup
1. Clone the repository
git clone <your-github-repository-url>
cd BlueOrbit-Office-Assistant
2. Create a virtual environment
python3 -m venv venv

Activate it:

source venv/bin/activate
3. Install dependencies
pip install -r requirements.txt
🧠 Install Ollama

Install Ollama on your system and make sure the Ollama service is running.

Pull the Qwen3 model:

ollama pull qwen3:0.6b

Verify it:

ollama list

You should see:

qwen3:0.6b
🔐 Environment Variables

Create a .env file in the project root.

GOOGLE_API_KEY=your_google_api_key

Do not commit the actual .env file to GitHub.

Use .env.example as a template.

📚 RAG Pipeline

The policy RAG pipeline follows these steps:

Policy PDF
    ↓
PDF Text Extraction
    ↓
Text Cleaning
    ↓
Document Chunking
    ↓
Gemini Embeddings
    ↓
FAISS Vector Store
    ↓
Similarity Search
    ↓
Relevant Policy Chunks
    ↓
Agent Response

The vector store is used to retrieve relevant policy information instead of relying only on the LLM's knowledge.

🔧 Employee Tools

The assistant uses deterministic Python tools to retrieve structured employee information.

Available tools:

get_employee_details()
get_leave_balance()
get_expense_records()
get_total_expenses()
get_expense_summary()
get_it_assets()
get_office_details()

These tools read information from CSV files stored under:

data/csvs/
🤖 Agent Workflow

The application uses LangGraph to manage the agent workflow.

Instead of allowing the small local LLM to freely decide when to call tools, the system first identifies the user's intent using deterministic routing.

For example:

"What is my leave balance?"
        ↓
Leave intent detected
        ↓
get_leave_balance()
        ↓
Verified employee data
        ↓
Qwen3 formats the response

This approach improves reliability and prevents incorrect tool calls or repeated agent loops.

💬 Example Questions

Try questions such as:

What is my leave balance?

How many earned leaves do I have?

Can I carry forward my unused leave?

Can I take more leave than my current balance?

Show me my expense records.

What is my total expense?

What laptop is assigned to me?

Which office is available in Bangalore?

What is the work from home policy?

What is the leave policy?
▶️ Run the Application

From the project root:

streamlit run app.py

The Streamlit application will open in your browser.

🎯 Project Objective

The objective of the Blue Orbit Office Assistant is to provide employees with a single conversational interface for accessing workplace information.

Instead of manually searching through spreadsheets, documents, or internal policies, employees can ask questions in natural language and receive relevant answers from structured company data and policy documents.

🔒 Data & Security
Employee information is retrieved from local CSV files.
Policy documents are processed through the RAG pipeline.
Qwen3 runs locally through Ollama.
API keys should be stored in .env.
Sensitive credentials should never be committed to Git.
👥 Project

Blue Orbit Office Assistant

Built using:

Python
Streamlit
LangGraph
Qwen3
Ollama
FAISS
RAG
Pandas

