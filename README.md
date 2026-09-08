# 🤖 Blue Orbit Office Assistant

An AI-powered workplace assistant that provides employees with conversational access to company information, employee records, leave details, expenses, IT assets, office information, and company policies.

The project combines **Streamlit, LangGraph, Qwen3, Ollama, deterministic intent routing, Python/Pandas tools, and a FAISS-based RAG pipeline**.

---

## 📌 Overview

The Blue Orbit Office Assistant provides a single conversational interface for common employee-related queries.

The system handles two types of information:

* **Structured employee data** — retrieved from CSV files using Python/Pandas tools.
* **Unstructured company policies** — retrieved using a Retrieval-Augmented Generation (RAG) pipeline with embeddings and FAISS.

The retrieved information is then passed to **Qwen3**, running locally through **Ollama**, to generate a natural-language response.

---

## ✨ Features

### 👤 Employee Information

* Employee details
* Department and designation
* Manager information
* Joining information
* Other available employee attributes

### 🏖️ Leave Management

* Leave balance
* Casual leave
* Earned leave
* Sick leave
* Leave carry-forward policy
* Additional leave-related queries

### 💰 Expense Management

* Individual expense records
* Total expenses
* Approved expenses
* Pending expenses
* Rejected expenses
* Expense summary

### 💻 IT Assets

* Assigned laptop and other IT assets

### 🏢 Office Information

* Office details based on city

### 📚 Company Policies

RAG-based answers for policies such as:

* Leave policy
* Work-from-home policy
* Attendance policy
* Expense policy
* IT security guidelines

---

## 🏗️ Architecture

```text
                         User
                           │
                           ▼
                    Streamlit UI
                           │
                           ▼
                  Employee Agent
                    (LangGraph)
                           │
                           ▼
              Deterministic Intent Routing
                     │           │
             ┌───────┘           └────────┐
             ▼                            ▼
      Employee Tools                  Policy RAG
        (Pandas)                         │
             │                     ┌──────┴──────┐
             ▼                     ▼             ▼
          CSV Data              Embeddings     FAISS
             │                     │             │
             └──────────┬──────────┴─────────────┘
                        ▼
                 Verified Context
                        │
                        ▼
                 Qwen3 via Ollama
                        │
                        ▼
                   Final Answer
                        │
                        ▼
                   Streamlit UI
```

---

## 🛠️ Technology Stack

| Technology        | Purpose                         |
| ----------------- | ------------------------------- |
| Python            | Core development                |
| Streamlit         | Web-based user interface        |
| LangGraph         | Agent workflow orchestration    |
| Qwen3             | Local language model            |
| Ollama            | Local LLM runtime               |
| Pandas            | Structured CSV data processing  |
| FAISS             | Vector similarity search        |
| Gemini Embeddings | Text embeddings                 |
| PyPDF             | PDF text extraction             |
| NumPy             | Numerical operations            |
| python-dotenv     | Environment variable management |

---

## 📁 Project Structure

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
└── .gitignore
```

---

## 🔄 How It Works

1. The employee enters their **Employee ID** and question through the Streamlit interface.
2. The question is passed to the LangGraph-based employee agent.
3. The agent determines the query intent using deterministic routing.
4. Structured queries are routed to the appropriate employee tool.
5. Policy-related queries are routed to the RAG pipeline.
6. The relevant information is retrieved from CSV data or policy documents.
7. The verified context is passed to Qwen3.
8. Qwen3 generates the final conversational response.
9. The response is displayed in the Streamlit interface.

---

## 🧩 Employee Tools

The project includes the following structured-data tools:

```python
get_employee_details(employee_id)
get_leave_balance(employee_id)
get_expense_records(employee_id)
get_total_expenses(employee_id)
get_expense_summary(employee_id)
get_it_assets(employee_id)
get_office_details(city)
```

These tools retrieve or calculate information directly from the project's structured datasets.

---

## 📚 RAG Pipeline

Policy documents are processed through the following pipeline:

```text
Policy PDF
    ↓
Text Extraction
    ↓
Text Cleaning
    ↓
Chunking
    ↓
Embeddings
    ↓
FAISS Vector Store
    ↓
Similarity Search
    ↓
Relevant Policy Chunks
    ↓
Qwen3
    ↓
Final Response
```

RAG allows the assistant to answer policy questions using the organization's documents instead of relying solely on the LLM's pretrained knowledge.

---

## 🤖 Qwen3 + Ollama

The project uses:

```text
Qwen3: qwen3:0.6b
```

The model runs locally using **Ollama**.

Qwen3 is primarily used for natural-language response generation, while critical employee information is retrieved deterministically from the underlying data sources.

---

## 🔗 Deterministic Intent Routing

The system initially explored LLM-based tool selection. However, unrestricted tool calling with a lightweight local model could result in incorrect tool selection and repeated tool calls.

To improve reliability, the final implementation uses **deterministic intent routing**.

```text
User Question
      ↓
Intent Detection
      ↓
┌───────────────┬───────────────┐
│               │               │
▼               ▼               ▼
Employee       Expense        Policy
Data           Data            RAG
│               │               │
└───────────────┴───────────────┘
                ↓
          Verified Context
                ↓
             Qwen3
```

This makes the system more predictable and reduces tool-calling loops and incorrect tool selection.

---

## 🧪 Example Queries

### Employee Data

```text
What are my employee details?
Who is my manager?
```

### Leave

```text
What is my leave balance?
How many earned leaves do I have?
How many sick leaves do I have?
Can I carry forward my unused leave?
```

### Expenses

```text
Show my expense records.
What is my total expense?
what is my approved expense ?
what is my pending expense ?
Give me an expense summary.
```

### IT Assets

```text
What laptop is assigned to me?
Show my IT assets.
```

### Office

```text
Which Blue Orbit office is in Bangalore?
```

### Policies

```text
What is the work-from-home policy?
What is the leave policy?
What is the attendance policy?
What is the expense policy?
What are the IT security guidelines?
```

---

## ⚙️ Setup

### 1. Clone the repository

```bash
git clone <repository-url>
cd BlueOrbit-Office-Assistant
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

For Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🦙 Ollama Setup

Install Ollama and pull the required model:

```bash
ollama pull qwen3:0.6b
```

Verify:

```bash
ollama list
```

Make sure Ollama is running before starting the application.

---

## 🔐 Environment Variables

Create a `.env` file in the project root using `.env.example` as a reference.

Example:

```env
GOOGLE_API_KEY=your_google_api_key
```

**Do not commit the ****`.env`**** file or API keys to GitHub.**

---

## ▶️ Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will be available through the local Streamlit URL displayed in the terminal.

---

## 🧪 Direct Agent Testing

The agent can also be tested directly from Python:

```python
from agent.employee_agent import ask_employee

response = ask_employee(
    "What is my leave balance?",
    employee_id="EMP001"
)

print(response)
```

---

## 🛡️ Reliability

The project follows a hybrid approach:

* **Structured employee information** is retrieved directly from CSV data.
* **Policy information** is retrieved through RAG.
* **Python handles calculations** such as expense totals.
* **Qwen3 generates natural-language responses** from verified context.
* Deterministic routing reduces unreliable LLM tool selection.

This separation helps reduce incorrect or fabricated employee information.

---

## 🚀 Future Enhancements

Potential improvements include:

* Employee authentication
* Role-based access control
* Integration with HR systems
* Persistent conversation memory
* Additional employee tools
* More policy documents
* Source citations in responses
* Deployment to a cloud environment
* Integration with Slack or Microsoft Teams
* Admin analytics dashboard
