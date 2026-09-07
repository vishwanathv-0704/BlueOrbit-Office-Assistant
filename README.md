# 🤖 Blue Orbit Office Assistant

An AI-powered workplace assistant that provides employees with quick and conversational access to company information, employee records, leave details, expense information, IT assets, office locations, and company policies.

The application combines **Streamlit**, **LangGraph**, **Qwen3 running locally through Ollama**, deterministic intent routing, structured employee tools, and a **FAISS-based Retrieval-Augmented Generation (RAG) pipeline** for policy-related questions.

---

## 📌 Project Overview

Employees often need to search through multiple sources to find basic workplace information such as:

* Leave balances
* Leave policies
* Expense records
* IT asset assignments
* Employee information
* Office details
* Work-from-home policies
* Attendance policies
* Company guidelines

The **Blue Orbit Office Assistant** provides a single conversational interface where employees can ask these questions in natural language.

Instead of manually searching CSV files or policy documents, the assistant identifies the user's intent, retrieves the required information from the appropriate source, and generates a concise response.

---

# 🎯 Objectives

The main objectives of this project are:

1. Build a conversational AI workplace assistant.
2. Retrieve structured employee information from CSV files.
3. Answer policy-related questions using RAG.
4. Use a local LLM through Ollama.
5. Use LangGraph to structure the agent workflow.
6. Reduce incorrect tool selection through deterministic routing.
7. Provide an easy-to-use Streamlit interface.
8. Keep employee data retrieval separate from policy document retrieval.
9. Provide reliable answers based on verified company data.

---

# ✨ Key Features

## 👤 Employee Information

The assistant can retrieve employee-related information such as:

* Employee name
* Employee ID
* Department
* Designation
* Manager
* Joining information
* Other available employee attributes

Example:

```text
What are my employee details?
```

---

## 🏖️ Leave Management

The assistant can answer questions related to employee leave information.

Examples:

```text
What is my leave balance?

How many earned leaves do I have?

How many sick leaves do I have?

How many casual leaves do I have?

Can I carry forward my unused leave?

Can I take more leave than my current balance?
```

The leave balance is retrieved directly from the employee leave data.

Policy-related leave questions are answered through the RAG pipeline.

---

## 💰 Expense Management

The assistant can retrieve and summarize employee expense information.

Supported queries include:

```text
Show my expense records.

What is my total expense?

How much of my expense is approved?

How much is pending?

Give me an expense summary.
```

The system can distinguish between:

* Total expenses
* Approved expenses
* Pending expenses
* Rejected expenses

---

## 💻 IT Asset Management

Employees can ask about the IT assets assigned to them.

Example:

```text
What laptop is assigned to me?
```

The assistant retrieves the asset information from the IT asset dataset.

---

## 🏢 Office Information

The assistant can provide information about Blue Orbit office locations.

Example:

```text
Which Blue Orbit office is in Bangalore?
```

Office information is retrieved based on the requested city.

---

## 📚 Company Policy Assistant

Policy-related questions are handled using a **Retrieval-Augmented Generation (RAG)** pipeline.

Examples:

```text
What is the work from home policy?

What is the leave policy?

What is the attendance policy?

What is the expense policy?

What are the IT security guidelines?
```

The system retrieves relevant information from the available company policy documents before generating an answer.

---

# 🧠 System Architecture

```text
                         ┌──────────────────────┐
                         │      User            │
                         │ Natural Language     │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │    Streamlit UI      │
                         │       app.py         │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   Employee Agent     │
                         │ employee_agent.py    │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Intent Detection &   │
                         │ Deterministic Route  │
                         └──────────┬───────────┘
                                    │
                     ┌──────────────┴──────────────┐
                     │                             │
                     ▼                             ▼
          ┌─────────────────────┐       ┌─────────────────────┐
          │  Employee Tools     │       │     Policy RAG      │
          │                     │       │                     │
          │ Employee CSVs       │       │ Policy Documents    │
          │ Leave CSV           │       │       ↓             │
          │ Expense CSV         │       │ Text Extraction     │
          │ IT Assets CSV       │       │       ↓             │
          │ Office CSV          │       │ Chunking            │
          └──────────┬──────────┘       │       ↓             │
                     │                  │ Embeddings          │
                     │                  │       ↓             │
                     │                  │ FAISS Vector Store  │
                     │                  │       ↓             │
                     │                  │ Similarity Search   │
                     │                  └──────────┬──────────┘
                     │                             │
                     └──────────────┬──────────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   Verified Context   │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   Qwen3 via Ollama   │
                         │   Response Formatting│
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │    Final Answer      │
                         │   → Streamlit UI     │
                         └──────────────────────┘
```

---

# 🛠️ Technology Stack

| Technology                   | Purpose                              |
| ---------------------------- | ------------------------------------ |
| **Python**                   | Core programming language            |
| **Streamlit**                | User interface                       |
| **LangGraph**                | Agent workflow orchestration         |
| **Qwen3**                    | Local language model                 |
| **Ollama**                   | Local LLM runtime                    |
| **FAISS**                    | Vector similarity search             |
| **Google Gemini Embeddings** | Text embeddings for policy documents |
| **PyPDF**                    | PDF text extraction                  |
| **Pandas**                   | CSV data processing                  |
| **NumPy**                    | Numerical operations                 |
| **python-dotenv**            | Environment variable management      |

---

# 📁 Project Structure

```text
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
│       └── FAISS vector store files
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
├── .env.example
└── .gitignore
```

---

# 🔄 How the System Works

The system separates **structured employee data retrieval** from **unstructured policy retrieval**.

## Step 1 — User asks a question

For example:

```text
What is my leave balance?
```

---

## Step 2 — Intent detection

The agent determines what type of information is required.

For example:

```text
Leave Balance Intent
```

---

## Step 3 — Appropriate tool is selected

The system calls:

```python
get_leave_balance(employee_id)
```

---

## Step 4 — Employee data is retrieved

The tool searches the leave balance CSV for the employee ID.

Example:

```text
EMP001
```

The result is returned as structured data.

---

## Step 5 — Verified data is passed to the response layer

The retrieved information is treated as the source of truth.

The language model is not responsible for calculating or inventing employee values.

---

## Step 6 — Qwen3 formats the response

Qwen3 converts the verified information into a natural-language response.

Example:

```text
You currently have:

Casual Leave: 1 day
Earned Leave: 14 days
Sick Leave: 8 days
Total: 23 days
```

---

# 🧩 Employee Tools

The project contains dedicated tools for structured employee information.

### `get_employee_details()`

Retrieves employee information.

```python
get_employee_details(employee_id)
```

---

### `get_leave_balance()`

Retrieves leave balances.

```python
get_leave_balance(employee_id)
```

---

### `get_expense_records()`

Retrieves individual expense records.

```python
get_expense_records(employee_id)
```

---

### `get_total_expenses()`

Calculates the total employee expenses.

```python
get_total_expenses(employee_id)
```

---

### `get_expense_summary()`

Provides an expense breakdown.

```python
get_expense_summary(employee_id)
```

The summary includes:

* Total
* Approved
* Pending
* Rejected

---

### `get_it_assets()`

Retrieves IT assets assigned to an employee.

```python
get_it_assets(employee_id)
```

---

### `get_office_details()`

Retrieves office information based on city.

```python
get_office_details(city)
```

---

# 📚 RAG Pipeline

The project uses Retrieval-Augmented Generation for company policy questions.

The pipeline follows:

```text
                Policy PDF
                    │
                    ▼
             PDF Text Extraction
                    │
                    ▼
              Text Cleaning
                    │
                    ▼
                 Chunking
                    │
                    ▼
             Text Embeddings
                    │
                    ▼
             FAISS Vector Store
                    │
                    ▼
             Similarity Search
                    │
                    ▼
          Relevant Policy Chunks
                    │
                    ▼
             Response Generation
```

---

# 🔎 RAG Retrieval Process

When a user asks a policy question:

```text
What is the work from home policy?
```

The system:

1. Identifies the question as a policy-related query.
2. Searches the policy vector store.
3. Calculates semantic similarity.
4. Retrieves the most relevant policy chunks.
5. Passes the retrieved context to the response generation layer.
6. Generates an answer based on the retrieved policy information.

This allows the assistant to answer questions using the organization's documents rather than relying only on the model's pretrained knowledge.

---

# 🤖 Why Qwen3 + Ollama?

The project uses **Qwen3** as the language model and runs it locally through **Ollama**.

Benefits include:

* Local model execution
* No need to send every conversational query to an external LLM
* Simple model management through Ollama
* Suitable for lightweight local agent applications
* Easy model replacement if required

The current model used by the project is:

```text
qwen3:0.6b
```

---

# 🔗 Why LangGraph?

LangGraph is used to structure the agent workflow.

It provides a graph-based approach for managing:

* Agent state
* Workflow execution
* Message handling
* Extensible agent architecture

The current implementation intentionally uses **deterministic intent routing** before calling tools.

This was chosen because smaller local models can sometimes make unreliable tool-selection decisions.

Instead of allowing the LLM to repeatedly decide whether to call tools, the application identifies the user's intent and invokes the appropriate function directly.

This makes the system:

* More predictable
* Easier to debug
* Less prone to tool-call loops
* More reliable for structured employee queries

---

# 🖥️ Streamlit Interface

The application provides a simple conversational UI built with Streamlit.

The user can:

1. Enter an employee ID.
2. Enter a natural-language question.
3. Submit the question.
4. Receive the assistant's response.

Example:

```text
Employee ID:
EMP001

Question:
What laptop is assigned to me?
```

The assistant then retrieves the relevant employee asset information and displays the answer.

---

# ⚙️ Installation

## 1. Clone the repository

```bash
git clone <your-repository-url>
```

Move into the project:

```bash
cd BlueOrbit-Office-Assistant
```

---

## 2. Create a virtual environment

```bash
python3 -m venv venv
```

Activate it:

### Linux / macOS

```bash
source venv/bin/activate
```

### Windows

```bash
venv\Scripts\activate
```

---

## 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

---

# 🦙 Ollama Setup

Install Ollama on your machine.

After installation, download the Qwen3 model:

```bash
ollama pull qwen3:0.6b
```

Verify that the model is available:

```bash
ollama list
```

You should see:

```text
qwen3:0.6b
```

Make sure Ollama is running before starting the Streamlit application.

---

# 🔐 Environment Configuration

Create a `.env` file in the project root.

Example:

```env
GOOGLE_API_KEY=your_google_api_key
```

The `.env` file should **never be committed to GitHub**.

The repository contains `.env.example` as a template.

---

# ▶️ Running the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

Streamlit will start the application locally.

Open the displayed local URL in your browser.

---

# 🧪 Example Queries

## Employee Information

```text
What are my employee details?
```

```text
Who is my manager?
```

---

## Leave

```text
What is my leave balance?
```

```text
How many earned leaves do I have?
```

```text
How many sick leaves do I have?
```

```text
Can I carry forward my unused leave?
```

```text
Can I take more leave than my current balance?
```

---

## Expenses

```text
Show my expense records.
```

```text
What is my total expense?
```

```text
How much of my expense is approved?
```

```text
How much is pending?
```

---

## IT Assets

```text
What laptop is assigned to me?
```

```text
Show my IT assets.
```

---

## Office

```text
Which office is in Bangalore?
```

```text
Tell me about the Bangalore office.
```

---

## Policies

```text
What is the leave policy?
```

```text
What is the work from home policy?
```

```text
What is the attendance policy?
```

```text
What is the expense policy?
```

```text
What are the IT security guidelines?
```

---

# 🧪 Testing

The agent can also be tested directly from Python.

Example:

```python
from agent.employee_agent import ask_employee

response = ask_employee(
    "What is my leave balance?",
    employee_id="EMP001"
)

print(response)
```

Another example:

```python
response = ask_employee(
    "What laptop is assigned to me?",
    employee_id="EMP001"
)

print(response)
```

---

# 🛡️ Reliability Approach

A key design decision in this project is that **structured employee information is retrieved deterministically**.

For example, if the user asks:

```text
What is my total expense?
```

The application directly uses:

```python
get_total_expenses(employee_id)
```

The LLM is not responsible for calculating the total.

Similarly, leave balances are retrieved directly from the CSV data.

This prevents the language model from:

* Guessing employee information
* Changing numerical values
* Inventing records
* Selecting incorrect tools
* Getting stuck in repeated tool-calling loops

The LLM is primarily used to turn verified information into a natural conversational response.

---

# 🔐 Data Handling

The project separates data into two major categories.

### Structured Data

Stored in CSV files:

```text
data/csvs/
```

Examples:

* Employee information
* Leave balances
* Expenses
* IT assets
* Office locations

### Unstructured Data

Company policies are processed through the RAG pipeline and stored in the FAISS vector store.

This separation allows each type of data to be retrieved using the most appropriate method.

---

# 📈 Future Improvements

Potential future enhancements include:

* Authentication and employee login
* Role-based access control
* More company tools
* More policy documents
* Conversation memory
* Persistent long-term memory
* Better source citations in responses
* Multi-document RAG
* Admin dashboard
* Analytics on frequently asked questions
* Support for additional local LLMs
* Integration with internal HR systems
* Integration with Slack or Microsoft Teams

---

# 🎓 Key Concepts Demonstrated

This project demonstrates practical implementation of:

* Generative AI
* AI Agents
* LangGraph
* Local LLMs
* Ollama
* Qwen3
* Retrieval-Augmented Generation
* Vector databases
* FAISS
* Semantic search
* Text embeddings
* PDF processing
* Tool-based data retrieval
* Deterministic intent routing
* Streamlit application development
* Structured and unstructured data integration

---

# 👥 Project Use Case

The Blue Orbit Office Assistant is designed as an internal employee-support assistant.

A typical interaction looks like:

```text
Employee
   │
   │ "What is my leave balance?"
   ▼
Blue Orbit Office Assistant
   │
   ▼
Intent Detection
   │
   ▼
Leave Tool
   │
   ▼
Employee Leave Data
   │
   ▼
Verified Result
   │
   ▼
Qwen3
   │
   ▼
Natural Language Response
```

For policy questions:

```text
Employee
   │
   │ "What is the work from home policy?"
   ▼
Intent Detection
   │
   ▼
Policy RAG
   │
   ▼
FAISS Similarity Search
   │
   ▼
Relevant Policy Context
   │
   ▼
Qwen3
   │
   ▼
Policy Answer
```

---

# 🚨 Important Notes

### Do not commit the virtual environment

The following should remain ignored:

```text
venv/
```

### Do not commit secrets

Never commit:

```text
.env
```

API keys and other credentials should always be stored securely.

### Keep `.env.example`

`.env.example` should contain only placeholder configuration values.

---

# 📄 License

This project is intended for educational, demonstration, and internal development purposes.

---

# 🙌 Acknowledgements

Built as part of an AI-powered workplace assistant project using modern LLM, agent, RAG, and data-retrieval technologies.

---

## ⭐ Blue Orbit Office Assistant

**Ask. Retrieve. Understand.**

A conversational AI assistant for smarter workplace information access.
