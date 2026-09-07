# ============================================================
# BLUE ORBIT OFFICE ASSISTANT
# employee_agent.py
#
# Architecture:
#
# Streamlit
#     ↓
# ask_employee()
#     ↓
# Deterministic intent detection
#     ↓
# Existing employee tools / RAG
#     ↓
# Qwen3 via Ollama
#     ↓
# Final answer
#
# IMPORTANT:
# Qwen3 does NOT decide which tool to call.
# This avoids the previous recursive tool-calling problem.
# ============================================================

from typing import TypedDict, List, Dict, Any

from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# ============================================================
# EXISTING EMPLOYEE TOOLS
# ============================================================

from tools.employee_tools_clean import (
    get_employee_details,
    get_leave_balance,
    get_expense_records,
    get_it_assets,
    get_office_details,
    get_total_expenses,
    get_expense_summary,
)

# ============================================================
# EXISTING RAG
# ============================================================

from rag.retrieval import search_policy as retrieve_policy


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "qwen3:0.6b"

llm = ChatOllama(
    model=MODEL_NAME,
    temperature=0,
)


# ============================================================
# STATE
# ============================================================

class AgentState(TypedDict, total=False):
    messages: List[Any]
    employee_id: str
    final_answer: str


# ============================================================
# BASIC HELPERS
# ============================================================

def day_word(value: int) -> str:
    """Return singular/plural day wording."""
    return "day" if value == 1 else "days"


def safe_int(value: Any, default: int = 0) -> int:
    """Safely convert a value to integer."""

    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


# ============================================================
# INTENT DETECTION
# ============================================================

def detect_intents(question: str) -> Dict[str, bool]:

    q = question.lower().strip()

    intents = {
        "employee_details": False,
        "leave_balance": False,
        "leave_policy": False,
        "expenses": False,
        "total_expenses": False,
        "expense_summary": False,
        "it_assets": False,
        "office_info": False,
        "unrelated": False,
    }

    # ========================================================
    # LEAVE BALANCE
    # ========================================================

    leave_balance_keywords = [
        "leave balance",
        "how much leave",
        "how many leave",
        "how many leaves",
        "remaining leave",
        "remaining leaves",
        "available leave",
        "available leaves",
        "my leave",
        "my leaves",
        "casual leave",
        "earned leave",
        "sick leave",
    ]

    if any(keyword in q for keyword in leave_balance_keywords):
        intents["leave_balance"] = True

    # ========================================================
    # LEAVE POLICY
    # ========================================================

    leave_policy_keywords = [
        "leave policy",
        "leave policies",
        "carry forward",
        "carry-forward",
        "carryforward",
        "how many can i carry",
        "how much can i carry",
        "earned leave carry",
        "leave entitlement",
        "leave rules",
        "leave approval",
        "leave application",
        "leave expires",
        "leave expiry",
        "leave expiration",
        "what happens if i need more leave",
        "more leave than",
        "not enough leave",
        "insufficient leave",
        "exceed available",
        "exceed my leave",
        "leave beyond my balance",
        "additional leave",
        "extra leave",
    ]

    if any(keyword in q for keyword in leave_policy_keywords):
        intents["leave_policy"] = True

    # ========================================================
    # EMPLOYEE DETAILS
    # ========================================================

    employee_keywords = [
        "employee details",
        "my details",
        "my information",
        "employee information",
        "my department",
        "my designation",
        "my role",
        "my manager",
        "my email",
        "my employee id",
        "my employee number",
        "joining date",
        "when did i join",
    ]

    if any(keyword in q for keyword in employee_keywords):
        intents["employee_details"] = True

    # ========================================================
    # IT ASSETS
    # ========================================================

    asset_keywords = [
        "laptop",
        "computer",
        "desktop",
        "monitor",
        "keyboard",
        "mouse",
        "it asset",
        "it assets",
        "asset",
        "assets",
        "device",
        "devices",
        "assigned to me",
        "what was assigned",
    ]

    if any(keyword in q for keyword in asset_keywords):
        intents["it_assets"] = True

    # ========================================================
    # TOTAL EXPENSES
    # ========================================================

    total_expense_keywords = [
        "total expenses",
        "total expense",
        "how much have i spent",
        "how much did i spend",
        "how much i spent",
        "total amount spent",
        "total spending",
        "my total spending",
        "sum of my expenses",
    ]

    if any(keyword in q for keyword in total_expense_keywords):
        intents["total_expenses"] = True

    # ========================================================
    # EXPENSE SUMMARY
    # ========================================================

    expense_summary_keywords = [
        "expense summary",
        "summarize my expenses",
        "summary of my expenses",
        "expense breakdown",
        "spending breakdown",
        "breakdown of my expenses",
    ]

    if any(keyword in q for keyword in expense_summary_keywords):
        intents["expense_summary"] = True

    # ========================================================
    # EXPENSE RECORDS
    # ========================================================

    expense_keywords = [
        "expense",
        "expenses",
        "reimbursement",
        "reimbursements",
        "spending",
        "spent",
        "claims",
        "expense records",
    ]

    if any(keyword in q for keyword in expense_keywords):
        intents["expenses"] = True

    # ========================================================
    # OFFICE
    # ========================================================

    office_keywords = [
        "office",
        "office location",
        "office address",
        "office details",
        "where is the office",
        "office timings",
        "office hours",
        "working hours",
        "building",
        "location",
    ]

    if any(keyword in q for keyword in office_keywords):
        intents["office_info"] = True

    # ========================================================
    # WORKPLACE QUESTION CHECK
    # ========================================================

    workplace_words = [
        "employee",
        "leave",
        "office",
        "expense",
        "expenses",
        "reimbursement",
        "laptop",
        "asset",
        "assets",
        "it",
        "policy",
        "department",
        "manager",
        "designation",
        "joining",
        "working",
        "work",
        "company",
        "attendance",
        "wfh",
        "work from home",
        "holiday",
        "holidays",
        "security",
        "password",
        "access",
        "benefit",
        "benefits",
    ]

    is_workplace = any(
        word in q for word in workplace_words
    )

    if not is_workplace:
        intents["unrelated"] = True

    return intents


# ============================================================
# POLICY SEARCH
# ============================================================

def search_company_policy(question: str) -> str:

    q = question.lower().strip()

    # ========================================================
    # INSUFFICIENT LEAVE / ADDITIONAL LEAVE
    # ========================================================

    insufficient_leave_keywords = [
        "what happens if i need more leave",
        "more leave than",
        "not enough leave",
        "insufficient leave",
        "exceed available",
        "exceed my leave",
        "leave beyond my balance",
        "additional leave",
        "extra leave",
    ]

    if any(
        keyword in q
        for keyword in insufficient_leave_keywords
    ):

        policy_query = (
            "company leave policy for taking leave when "
            "available leave balance is insufficient, "
            "leave exceeding available balance, "
            "leave beyond entitlement, additional leave, "
            "extra leave, approval when employee does not "
            "have enough leave balance"
        )

        try:
            result = retrieve_policy(policy_query)
            return str(result)

        except Exception as e:
            return f"POLICY_SEARCH_ERROR: {e}"

    # ========================================================
    # CARRY FORWARD
    # ========================================================

    carry_forward_keywords = [
        "carry forward",
        "carry-forward",
        "carryforward",
        "how many can i carry",
        "how much can i carry",
        "earned leave carry",
    ]

    if any(
        keyword in q
        for keyword in carry_forward_keywords
    ):

        policy_query = (
            "company leave carry forward policy, "
            "earned leave carry forward limit, "
            "number of earned leave days that can be "
            "carried into the next calendar year"
        )

        try:
            result = retrieve_policy(policy_query)
            return str(result)

        except Exception as e:
            return f"POLICY_SEARCH_ERROR: {e}"

    # ========================================================
    # GENERAL LEAVE POLICY
    # ========================================================

    leave_policy_keywords = [
        "leave policy",
        "leave policies",
        "leave entitlement",
        "leave rules",
        "leave approval",
        "leave application",
        "leave expires",
        "leave expiry",
        "casual leave",
        "earned leave",
        "sick leave",
    ]

    if any(
        keyword in q
        for keyword in leave_policy_keywords
    ):

        policy_query = (
            "company leave policy including casual leave, "
            "earned leave, sick leave, entitlement, "
            "leave application, approval, expiry and "
            "carry forward"
        )

        try:
            result = retrieve_policy(policy_query)
            return str(result)

        except Exception as e:
            return f"POLICY_SEARCH_ERROR: {e}"

    # ========================================================
    # WFH
    # ========================================================

    if (
        "wfh" in q
        or "work from home" in q
        or "remote work" in q
        or "remote working" in q
    ):

        policy_query = (
            "company work from home policy, remote work "
            "policy, WFH eligibility, approval and rules"
        )

        try:
            result = retrieve_policy(policy_query)
            return str(result)

        except Exception as e:
            return f"POLICY_SEARCH_ERROR: {e}"

    # ========================================================
    # ATTENDANCE
    # ========================================================

    if "attendance" in q:

        policy_query = (
            "company attendance policy, attendance rules, "
            "working hours and attendance requirements"
        )

        try:
            result = retrieve_policy(policy_query)
            return str(result)

        except Exception as e:
            return f"POLICY_SEARCH_ERROR: {e}"

    # ========================================================
    # EXPENSE POLICY
    # ========================================================

    if (
        "expense policy" in q
        or "reimbursement policy" in q
        or "expense reimbursement" in q
    ):

        policy_query = (
            "company expense reimbursement policy, "
            "expense rules, reimbursement limits and process"
        )

        try:
            result = retrieve_policy(policy_query)
            return str(result)

        except Exception as e:
            return f"POLICY_SEARCH_ERROR: {e}"

    # ========================================================
    # IT / SECURITY POLICY
    # ========================================================

    if (
        "security" in q
        or "password" in q
        or "it policy" in q
        or "access policy" in q
    ):

        policy_query = (
            "company IT security policy, password policy, "
            "access rules and security requirements"
        )

        try:
            result = retrieve_policy(policy_query)
            return str(result)

        except Exception as e:
            return f"POLICY_SEARCH_ERROR: {e}"

    # ========================================================
    # FALLBACK POLICY SEARCH
    # ========================================================

    try:
        result = retrieve_policy(question)
        return str(result)

    except Exception as e:
        return f"POLICY_SEARCH_ERROR: {e}"


# ============================================================
# EMPLOYEE TOOL WRAPPERS
# ============================================================

def get_my_employee_details(
    employee_id: str
) -> Dict[str, Any]:

    try:
        return get_employee_details(employee_id)

    except Exception as e:
        return {
            "success": False,
            "data": None,
            "message": str(e),
        }


# ============================================================
# LEAVE BALANCE
# ============================================================

def get_my_leave_balance(
    employee_id: str
) -> Dict[str, Any]:

    try:
        result = get_leave_balance(employee_id)

        # IMPORTANT:
        # employee_tools_clean.py returns:
        #
        # {
        #   "success": True,
        #   "data": {...},
        #   "message": "Leave balance found."
        # }
        #
        # We preserve that exact structure.

        return result

    except Exception as e:
        return {
            "success": False,
            "data": None,
            "message": str(e),
        }


# ============================================================
# EXPENSE RECORDS
# ============================================================

def get_my_expense_records(
    employee_id: str
) -> Dict[str, Any]:

    try:
        return get_expense_records(employee_id)

    except Exception as e:
        return {
            "success": False,
            "data": [],
            "message": str(e),
        }


# ============================================================
# TOTAL EXPENSES
# ============================================================

def get_my_total_expenses(
    employee_id: str
) -> Dict[str, Any]:

    try:
        return get_total_expenses(employee_id)

    except Exception as e:
        return {
            "success": False,
            "data": None,
            "message": str(e),
        }


# ============================================================
# EXPENSE SUMMARY
# ============================================================

def get_my_expense_summary(
    employee_id: str
) -> Dict[str, Any]:

    try:
        return get_expense_summary(employee_id)

    except Exception as e:
        return {
            "success": False,
            "data": None,
            "message": str(e),
        }


# ============================================================
# IT ASSETS
# ============================================================

def get_my_it_assets(
    employee_id: str
) -> Dict[str, Any]:

    try:
        return get_it_assets(employee_id)

    except Exception as e:
        return {
            "success": False,
            "data": [],
            "message": str(e),
        }


# ============================================================
# OFFICE INFORMATION
# ============================================================

def get_my_office_information(
    question: str
) -> Dict[str, Any]:

    """
    get_office_details() requires a city.

    Try to extract a city from the user's question.
    If no city is specified, return a clear message rather
    than calling the tool incorrectly.
    """

    q = question.lower().strip()

    # Common cities that may exist in the office CSV.
    known_cities = [
        "bangalore",
        "bengaluru",
        "chennai",
        "hyderabad",
        "pune",
        "mumbai",
        "delhi",
        "noida",
        "gurgaon",
        "gurugram",
        "kolkata",
        "kochi",
        "mysore",
        "mysuru",
    ]

    city = None

    for candidate in known_cities:

        if candidate in q:
            city = candidate
            break

    if city is None:

        return {
            "success": False,
            "data": None,
            "message": (
                "The user did not specify a city for the "
                "office lookup."
            ),
        }

    # Normalize Bengaluru → Bangalore
    if city == "bengaluru":
        city = "bangalore"

    try:
        return get_office_details(city)

    except Exception as e:
        return {
            "success": False,
            "data": None,
            "message": str(e),
        }


# ============================================================
# EXECUTE INTENTS
# ============================================================

def execute_intents(
    question: str,
    employee_id: str,
) -> Dict[str, Any]:

    intents = detect_intents(question)

    # ========================================================
    # UNRELATED QUESTION
    #
    # VERY IMPORTANT:
    # This happens BEFORE Qwen.
    # ========================================================

    if intents["unrelated"]:

        return {
            "type": "unrelated",
            "data": None,
        }

    results = {}

    # ========================================================
    # EMPLOYEE DETAILS
    # ========================================================

    if intents["employee_details"]:

        results["employee_details"] = (
            get_my_employee_details(employee_id)
        )

    # ========================================================
    # LEAVE BALANCE
    # ========================================================

    if intents["leave_balance"]:

        results["leave_balance"] = (
            get_my_leave_balance(employee_id)
        )

    # ========================================================
    # LEAVE POLICY
    # ========================================================

    if intents["leave_policy"]:

        results["leave_policy"] = (
            search_company_policy(question)
        )

    # ========================================================
    # TOTAL EXPENSES
    # ========================================================

    if intents["total_expenses"]:

        results["total_expenses"] = (
            get_my_total_expenses(employee_id)
        )

    # ========================================================
    # EXPENSE SUMMARY
    # ========================================================

    elif intents["expense_summary"]:

        results["expense_summary"] = (
            get_my_expense_summary(employee_id)
        )

    # ========================================================
    # EXPENSE RECORDS
    # ========================================================

    elif intents["expenses"]:

        results["expenses"] = (
            get_my_expense_records(employee_id)
        )

    # ========================================================
    # IT ASSETS
    # ========================================================

    if intents["it_assets"]:

        results["it_assets"] = (
            get_my_it_assets(employee_id)
        )

    # ========================================================
    # OFFICE
    # ========================================================

    if intents["office_info"]:

        results["office_info"] = (
            get_my_office_information(question)
        )

    # ========================================================
    # NOTHING FOUND
    # ========================================================

    if not results:

        return {
            "type": "unrelated",
            "data": None,
        }

    return {
        "type": "workplace",
        "data": results,
    }


# ============================================================
# FORMAT LEAVE BALANCE
#
# THIS IS NOW FULLY DETERMINISTIC.
#
# QWEN DOES NOT TOUCH THE NUMBERS.
# ============================================================

def format_leave_balance(
    tool_result: Dict[str, Any]
) -> str:

    # --------------------------------------------------------
    # Check tool success
    # --------------------------------------------------------

    if not tool_result.get("success", False):

        return tool_result.get(
            "message",
            "Leave balance could not be found.",
        )

    # --------------------------------------------------------
    # Correct structure:
    #
    # tool_result["data"]
    # --------------------------------------------------------

    data = tool_result.get("data")

    if not isinstance(data, dict):

        return (
            "I couldn't read the leave balance correctly."
        )

    # --------------------------------------------------------
    # Extract exact CSV columns
    # --------------------------------------------------------

    casual = safe_int(
        data.get("casual_leave", 0)
    )

    earned = safe_int(
        data.get("earned_leave", 0)
    )

    sick = safe_int(
        data.get("sick_leave", 0)
    )

    # --------------------------------------------------------
    # Use provided total if available.
    # Otherwise calculate it.
    # --------------------------------------------------------

    if "total_leave" in data:

        total = safe_int(
            data.get("total_leave")
        )

    elif "total" in data:

        total = safe_int(
            data.get("total")
        )

    else:

        total = casual + earned + sick

    # --------------------------------------------------------
    # FINAL DETERMINISTIC ANSWER
    # --------------------------------------------------------

    return (
        "Your leave balance is:\n"
        f"- Casual Leave: {casual} {day_word(casual)}\n"
        f"- Earned Leave: {earned} {day_word(earned)}\n"
        f"- Sick Leave: {sick} {day_word(sick)}\n"
        f"- Total: {total} {day_word(total)}"
    )


# ============================================================
# FORMAT TOTAL EXPENSES
#
# ALSO DETERMINISTIC.
# ============================================================

def format_total_expenses(
    tool_result: Dict[str, Any]
) -> str:

    if not tool_result.get("success", False):

        return tool_result.get(
            "message",
            "No expense information was found.",
        )

    data = tool_result.get("data")

    if not isinstance(data, dict):

        return (
            "I couldn't read the total expense information."
        )

    total = data.get(
        "total_expenses",
        0,
    )

    try:
        total = float(total)

        return (
            f"Your total expenses are "
            f"{total:,.2f}."
        )

    except (TypeError, ValueError):

        return (
            f"Your total expenses are {total}."
        )


# ============================================================
# FORMAT EMPLOYEE DETAILS
# ============================================================

def format_employee_details(
    tool_result: Dict[str, Any]
) -> str:

    if not tool_result.get("success", False):

        return tool_result.get(
            "message",
            "Employee details could not be found.",
        )

    data = tool_result.get("data")

    if not isinstance(data, dict):

        return "I couldn't read the employee details."

    # We intentionally let Qwen handle the natural-language
    # formatting for employee details because there may be
    # several columns.
    return ""


# ============================================================
# FINAL SYSTEM PROMPT
# ============================================================

FINAL_SYSTEM_PROMPT = """
You are the Blue Orbit Office Assistant.

Your job is ONLY to turn verified company information into a
short, clear and friendly response.

STRICT RULES:

1. Use ONLY information provided in CONTEXT.

2. NEVER invent employee information.

3. NEVER invent company policies.

4. NEVER change, calculate differently, or reinterpret numeric
   employee data that has already been supplied.

5. If the context does not contain the answer, say:
   "The available company information does not specify that."

6. For policy questions, answer ONLY from the retrieved policy
   information.

7. Never infer one policy from another.

8. IMPORTANT DISTINCTION:

   "How many earned leave days can I carry forward?"
   means CARRY-FORWARD.

   "What happens if I need more leave than my available balance?"
   means INSUFFICIENT BALANCE / ADDITIONAL LEAVE.

   These are different questions.

9. NEVER claim that carry-forward means the employee can
   automatically take additional leave beyond their balance.

10. If the policy does not explicitly explain what happens when
    someone needs more leave than their available balance, say:
    "The available company policy does not specify what happens
    when the available leave balance is insufficient."

11. Do not invent HR approval rules, exceptions, penalties,
    forfeiture rules, or additional leave rules.

12. Do not mention tools, RAG, LangGraph, Ollama, Qwen,
    retrieval, prompts, context, or internal systems.

13. Do not say "there are no records" for an unrelated question.

14. Keep answers concise.

15. Use bullet points when useful.

16. If the user asks multiple questions, answer every part.

17. Preserve exact numbers from the supplied employee data.

18. For leave balances, do not modify the numbers.

19. For currency amounts, preserve the supplied total.

20. If the context contains a successful tool result, do not say
    that you do not have access to that information.
"""


# ============================================================
# FINAL ANSWER GENERATOR
# ============================================================

def generate_final_answer(
    question: str,
    result: Dict[str, Any],
) -> str:

    # ========================================================
    # UNRELATED
    # ========================================================

    if result.get("type") == "unrelated":

        return (
            "I can help with Blue Orbit workplace questions such as "
            "employee details, leave balance and policies, expenses, "
            "IT assets, and office information."
        )

    data = result.get("data", {})

    # ========================================================
    # LEAVE BALANCE
    #
    # Completely deterministic.
    # Qwen cannot mess up the numbers.
    # ========================================================

    if (
        isinstance(data, dict)
        and "leave_balance" in data
        and len(data) == 1
    ):

        return format_leave_balance(
            data["leave_balance"]
        )

    # ========================================================
    # TOTAL EXPENSES
    #
    # Completely deterministic.
    # ========================================================

    if (
        isinstance(data, dict)
        and "total_expenses" in data
        and len(data) == 1
    ):

        return format_total_expenses(
            data["total_expenses"]
        )

    # ========================================================
    # BUILD CONTEXT
    # ========================================================

    context = str(data)

    # ========================================================
    # QWEN ONLY FORMATS VERIFIED INFORMATION
    # ========================================================

    prompt = f"""
{FINAL_SYSTEM_PROMPT}

USER QUESTION:
{question}

VERIFIED COMPANY INFORMATION:
{context}

Now answer the user's question.

Remember:
- Use only the verified information.
- Never guess.
- Never invent policy.
- Never change numbers.
- Do not confuse leave carry-forward with taking additional leave.
"""

    try:

        response = llm.invoke(
            [
                HumanMessage(
                    content=prompt
                )
            ]
        )

        answer = str(
            response.content
        ).strip()

        if not answer:

            return (
                "I couldn't generate an answer from "
                "the available company information."
            )

        return answer

    except Exception as e:

        print(
            f"[Qwen formatting error] {e}"
        )

        # ----------------------------------------------------
        # Fallback:
        # Return raw verified information rather than hallucinate.
        # ----------------------------------------------------

        return (
            "Here is the available company information:\n"
            f"{context}"
        )


# ============================================================
# LANGGRAPH NODE
# ============================================================

def agent_node(
    state: AgentState
) -> AgentState:

    messages = state.get(
        "messages",
        []
    )

    employee_id = state.get(
        "employee_id",
        "EMP001"
    )

    # ========================================================
    # GET CURRENT USER QUESTION
    # ========================================================

    question = ""

    for message in reversed(messages):

        if isinstance(message, HumanMessage):

            question = str(
                message.content
            ).strip()

            break

    if not question:

        return {
            "final_answer": (
                "Please enter a workplace-related question."
            )
        }

    # ========================================================
    # DETERMINISTIC ROUTING
    # ========================================================

    result = execute_intents(
        question=question,
        employee_id=employee_id,
    )

    # ========================================================
    # GENERATE ANSWER
    # ========================================================

    answer = generate_final_answer(
        question=question,
        result=result,
    )

    return {
        "final_answer": answer
    }


# ============================================================
# BUILD LANGGRAPH
# ============================================================

workflow = StateGraph(
    AgentState
)

workflow.add_node(
    "agent",
    agent_node
)

workflow.add_edge(
    START,
    "agent"
)

workflow.add_edge(
    "agent",
    END
)


# ============================================================
# MEMORY
# ============================================================

memory = MemorySaver()


# ============================================================
# COMPILE GRAPH
# ============================================================

app = workflow.compile(
    checkpointer=memory
)


# ============================================================
# PUBLIC FUNCTION
#
# Streamlit should continue calling:
#
# ask_employee(question, employee_id, thread_id)
# ============================================================

def ask_employee(
    question: str,
    employee_id: str = "EMP001",
    thread_id: str = "default",
) -> str:

    if not question or not question.strip():

        return "Please enter a question."

    # ========================================================
    # INITIAL STATE
    # ========================================================

    initial_state: AgentState = {
        "messages": [
            HumanMessage(
                content=question.strip()
            )
        ],
        "employee_id": employee_id,
    }

    # ========================================================
    # THREAD CONFIG
    # ========================================================

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    # ========================================================
    # INVOKE LANGGRAPH
    # ========================================================

    try:

        result = app.invoke(
            initial_state,
            config=config,
        )

        answer = result.get(
            "final_answer",
            ""
        )

        if not answer:

            return (
                "I couldn't generate an answer. "
                "Please try asking the question another way."
            )

        return str(answer).strip()

    except Exception as e:

        print(
            f"[Blue Orbit Agent Error] {e}"
        )

        return (
            "Sorry, I ran into a problem while processing "
            "your question. Please try again."
        )


# ============================================================
# LOCAL TESTING
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("BLUE ORBIT OFFICE ASSISTANT")
    print("=" * 60)

    tests = [
        "What laptop was assigned to me?",
        "What is my leave balance?",
        "I have 14 earned leave days. How many can I carry forward?",
        "What happens if I need more leave than my available balance?",
        "How much have I spent in total?",
        "What is my leave balance and what does the leave policy say?",
        "What is the capital of France?",
    ]

    for question in tests:

        print("\nUSER:")
        print(question)

        answer = ask_employee(
            question=question,
            employee_id="EMP001",
            thread_id="local_test",
        )

        print("\nASSISTANT:")
        print(answer)

        print("-" * 60)