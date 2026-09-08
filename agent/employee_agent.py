# ============================================================
# BLUE ORBIT OFFICE ASSISTANT
# employee_agent.py
# ============================================================

from typing import TypedDict, List, Dict, Any
import re

from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from tools.employee_tools_clean import (
    get_employee_details,
    get_leave_balance,
    get_expense_records,
    get_it_assets,
    get_office_details,
    get_total_expenses,
    get_expense_summary,
)

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
# HELPERS
# ============================================================

def safe_int(value, default=0):

    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def day_word(value):

    return "day" if value == 1 else "days"


def is_empty_value(value):

    if value is None:
        return True

    text = str(value).strip().lower()

    return text in [
        "",
        "null",
        "none",
        "nan",
        "nat",
    ]


def normalize_question(question: str) -> str:

    """
    Handles a few common typing mistakes so that
    deterministic intent detection still works.
    """

    q = question.lower().strip()

    replacements = {
        "offfice": "office",
        "loaction": "location",
        "assest": "asset",
        "expensee": "expense",
    }

    for wrong, correct in replacements.items():

        q = q.replace(
            wrong,
            correct,
        )

    return q


# ============================================================
# INTENT DETECTION
# ============================================================

def detect_intents(question: str) -> Dict[str, bool]:

    q = normalize_question(question)

    intents = {
        "employee_details": False,
        "manager_info": False,

        "leave_balance": False,
        "specific_leave_balance": False,
        "leave_policy": False,
        "pending_leave": False,

        "policy": False,

        "expenses": False,
        "pending_expenses": False,
        "approved_expenses": False,
        "rejected_expenses": False,

        "total_expenses": False,
        "expense_summary": False,

        "it_assets": False,
        "specific_asset": False,

        "office_info": False,

        "unrelated": False,
    }

    # ========================================================
    # MANAGER
    # ========================================================

    manager_keywords = [
        "who is my manager",
        "who's my manager",
        "my manager",
        "manager name",
        "manager details",
        "reporting manager",
        "who do i report to",
        "who do i report",
        "my reporting manager",
    ]

    if any(
        keyword in q
        for keyword in manager_keywords
    ):
        intents["manager_info"] = True

    # ========================================================
    # PENDING LEAVE
    # ========================================================

    pending_leave_keywords = [
        "pending leave",
        "pending leaves",
        "pending sick leave",
        "pending casual leave",
        "pending earned leave",
        "pending annual leave",
        "leave pending",
        "leave request pending",
        "pending leave request",
        "pending leave requests",
        "leave approval pending",
    ]

    if any(
        keyword in q
        for keyword in pending_leave_keywords
    ):
        intents["pending_leave"] = True

    # ========================================================
    # PENDING EXPENSES
    # ========================================================

    pending_expense_keywords = [
        "pending expense",
        "pending expenses",
        "pending reimbursement",
        "pending reimbursements",
        "pending claim",
        "pending claims",
        "expenses pending",
        "reimbursement pending",
        "which expenses are pending",
        "what expenses are pending",
        "show pending expenses",
        "my pending expenses",
    ]

    if any(
        keyword in q
        for keyword in pending_expense_keywords
    ):
        intents["pending_expenses"] = True

    # ========================================================
    # APPROVED EXPENSES
    # ========================================================

    approved_expense_keywords = [
        "approved expense",
        "approved expenses",
        "approved reimbursement",
        "approved reimbursements",
        "approved claims",
        "expenses approved",
        "which expenses are approved",
        "what expenses are approved",
        "show approved expenses",
        "my approved expenses",
    ]

    if any(
        keyword in q
        for keyword in approved_expense_keywords
    ):
        intents["approved_expenses"] = True

    # ========================================================
    # REJECTED EXPENSES
    # ========================================================

    rejected_expense_keywords = [
        "rejected expense",
        "rejected expenses",
        "rejected reimbursement",
        "rejected reimbursements",
        "rejected claims",
        "expenses rejected",
        "which expenses are rejected",
        "what expenses are rejected",
        "show rejected expenses",
        "my rejected expenses",
    ]

    if any(
        keyword in q
        for keyword in rejected_expense_keywords
    ):
        intents["rejected_expenses"] = True

    # ========================================================
    # SPECIFIC LEAVE BALANCE
    # ========================================================

    specific_leave_keywords = [
        "sick leave",
        "sick leaves",

        "casual leave",
        "casual leaves",

        "earned leave",
        "earned leaves",

        "annual leave",
        "annual leaves",
    ]

    has_leave_balance_context = any(
        word in q
        for word in [
            "balance",
            "remaining",
            "available",
            "how many",
            "how much",
        ]
    )

    if (
        not intents["pending_leave"]
        and any(
            keyword in q
            for keyword in specific_leave_keywords
        )
        and has_leave_balance_context
    ):

        intents["specific_leave_balance"] = True
        intents["leave_balance"] = True

    # ========================================================
    # GENERAL LEAVE BALANCE
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
        "leave details",
        "my leave details",
        "leave information",
        "my leave information",
    ]

    if (
        not intents["pending_leave"]
        and not intents["specific_leave_balance"]
        and any(
            keyword in q
            for keyword in leave_balance_keywords
        )
    ):

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
        "carry unused leave",
        "carry my leave",
        "earned leave carry",
        "leave entitlement",
        "leave rules",
        "leave approval",
        "leave application",
        "leave expires",
        "leave expiry",
        "leave expiration",
        "leave encashment",
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
        for keyword in leave_policy_keywords
    ):

        intents["leave_policy"] = True

    # ========================================================
    # GENERAL POLICY
    # ========================================================

    policy_keywords = [
        "policy",
        "policies",
        "guideline",
        "guidelines",
        "rule",
        "rules",
        "eligibility",
        "procedure",
        "procedures",

        "wfh",
        "work from home",
        "work-from-home",
        "remote work",
        "remote working",
        "work remotely",
        "working remotely",

        "attendance",
        "attendance requirement",
        "attendance requirements",

        "security",
        "security guideline",
        "security guidelines",
        "security requirement",
        "security requirements",
        "password",
        "passwords",
        "cybersecurity",
        "information security",
        "vpn",
        "multi-factor authentication",
        "mfa",
    ]

    if (
        not intents["leave_policy"]
        and any(
            keyword in q
            for keyword in policy_keywords
        )
    ):

        intents["policy"] = True

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
        "my email",
        "my employee id",
        "my employee number",
        "joining date",
        "when did i join",
        "employment status",
    ]

    if (
        not intents["manager_info"]
        and any(
            keyword in q
            for keyword in employee_keywords
        )
    ):

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
        "asset name",
        "asset id",
        "asset type",
        "device",
        "devices",
        "device name",
        "assigned to me",
        "what was assigned",
        "equipment",
        "equipment name",
    ]

    if any(
        keyword in q
        for keyword in asset_keywords
    ):

        intents["it_assets"] = True

    specific_asset_keywords = [
        "asset name",
        "device name",
        "what laptop",
        "laptop name",
        "computer name",
        "what device",
        "which laptop",
    ]

    if any(
        keyword in q
        for keyword in specific_asset_keywords
    ):

        intents["specific_asset"] = True

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
        "total reimbursement",
        "total reimbursements",
    ]

    if any(
        keyword in q
        for keyword in total_expense_keywords
    ):

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
        "expense status summary",
    ]

    if any(
        keyword in q
        for keyword in expense_summary_keywords
    ):

        intents["expense_summary"] = True

    # ========================================================
    # GENERAL EXPENSES
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

    if (
        not intents["pending_expenses"]
        and not intents["approved_expenses"]
        and not intents["rejected_expenses"]
        and not intents["total_expenses"]
        and not intents["expense_summary"]
        and any(
            keyword in q
            for keyword in expense_keywords
        )
    ):

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
        "where is my office",
        "my office",
        "office timings",
        "office timing",
        "office hours",
        "working hours",
        "working hour",
        "work hours",
        "building",
        "location",
        "location id",
    ]

    if any(
        keyword in q
        for keyword in office_keywords
    ):

        intents["office_info"] = True

    # ========================================================
    # WORKPLACE CHECK
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
        "policies",
        "department",
        "manager",
        "designation",
        "joining",
        "company",
        "attendance",
        "wfh",
        "work from home",
        "work-from-home",
        "remote",
        "holiday",
        "holidays",
        "security",
        "password",
        "access",
        "benefit",
        "benefits",
        "vpn",
        "claim",
        "claims",
        "device",
        "working hours",
        "office hours",
    ]

    is_workplace = any(
        word in q
        for word in workplace_words
    )

    if not is_workplace:

        intents["unrelated"] = True

    return intents


# ============================================================
# POLICY ANSWER GENERATION
# ============================================================

def clean_policy_answer(answer: str) -> str:

    if not answer:
        return ""

    unwanted_sentence_patterns = [
        "no additional information is provided",
        "no additional policy details are provided",
        "no additional leave or policy details are provided",
        "no additional details beyond this policy information are provided",
        "no additional details are provided",
        "no additional information beyond the verified data",
        "no specific status",
        "no additional details",
        "no additional information",
    ]

    # --------------------------------------------------------
    # Remove unwanted sentences even when they are attached
    # to a paragraph instead of appearing on their own line.
    # --------------------------------------------------------

    sentences = re.split(
        r"(?<=[.!?])\s+",
        answer.strip(),
    )

    cleaned_sentences = []

    for sentence in sentences:

        lower_sentence = sentence.lower().strip()

        if any(
            phrase in lower_sentence
            for phrase in unwanted_sentence_patterns
        ):
            continue

        cleaned_sentences.append(
            sentence.strip()
        )

    cleaned = " ".join(
        cleaned_sentences
    ).strip()

    return cleaned


def generate_policy_answer(
    question: str,
    context: str,
) -> str:

    if not context:
        return (
            "I couldn't find relevant information "
            "in the company policy documents."
        )

    prompt = f"""
You are the Blue Orbit Office Assistant.

Answer the user's question using ONLY the verified
company policy context below.

IMPORTANT RULES:

1. Do not invent information.
2. Do not mention the retrieval process.
3. Do not mention FAISS, embeddings, chunks, sources,
   context, or documents unless the user specifically asks.
4. Do not output Python dictionaries.
5. Do not output source metadata.
6. Do not repeat the question.
7. Give a concise and direct answer.
8. If the context contains several relevant rules,
   summarize the important ones as bullet points.
9. Never add filler such as:
   "No additional information is provided."
10. Never say that information is unavailable when it
    is clearly present in the context.

USER QUESTION:
{question}

VERIFIED POLICY INFORMATION:
{context}

Now provide only the clean answer for the employee.
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

        return clean_policy_answer(
            answer
        )

    except Exception as e:

        print(
            f"[Qwen policy error] {e}"
        )

        # Safe fallback:
        # return retrieved context rather than the raw dict.
        return context.strip()


# ============================================================
# POLICY SEARCH
# ============================================================

def search_company_policy(question: str) -> str:

    q = normalize_question(question)

    # ========================================================
    # WFH
    # ========================================================

    wfh_keywords = [
        "wfh",
        "work from home",
        "work-from-home",
        "remote work",
        "remote working",
        "work remotely",
        "working remotely",
        "can i work remotely",
        "can i work from home",
        "can i take wfh",
    ]

    if any(
        keyword in q
        for keyword in wfh_keywords
    ):

        policy_query = """
        Retrieve the complete Work From Home policy.

        Include:
        - eligibility
        - approval requirements
        - maximum WFH days per calendar month
        - whether unused WFH days carry forward
        - manager approval
        - role suitability
        - team requirements
        - restrictions
        - emergency WFH rules if available
        - equipment requirements if available
        """

    # ========================================================
    # ATTENDANCE
    # ========================================================

    elif "attendance" in q:

        policy_query = """
        Retrieve the complete company attendance policy.

        Include:
        - attendance requirements
        - working hours
        - late arrival
        - absence
        - manager notification
        - unapproved absence
        - remote employee attendance requirements
        """

    # ========================================================
    # SECURITY
    # ========================================================

    elif any(
        keyword in q
        for keyword in [
            "security",
            "cybersecurity",
            "information security",
            "it security",
            "access policy",
            "vpn",
            "mfa",
            "multi-factor authentication",
        ]
    ):

        policy_query = """
        Retrieve the complete company IT security policy.

        Include:
        - password requirements
        - password sharing
        - password reuse
        - MFA
        - remote access
        - VPN
        - company data protection
        - approved software
        - unauthorized software
        """

    # ========================================================
    # PASSWORD
    # ========================================================

    elif "password" in q:

        policy_query = """
        Retrieve the complete company password policy.

        Include:
        - password requirements
        - password uniqueness
        - password reuse
        - password sharing
        - password storage
        - password manager
        - compromised credentials
        """

    # ========================================================
    # INSUFFICIENT LEAVE
    # ========================================================

    elif any(
        keyword in q
        for keyword in [
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
    ):

        policy_query = """
        Retrieve the company policy about requesting more leave
        than the employee's available leave balance.

        Explain:
        - what happens
        - whether additional leave can be requested
        - approval requirements
        - exceptions
        """

    # ========================================================
    # CARRY FORWARD
    # ========================================================

    elif any(
        keyword in q
        for keyword in [
            "carry forward",
            "carry-forward",
            "carryforward",
            "how many can i carry",
            "how much can i carry",
            "carry unused leave",
            "carry my leave",
            "earned leave carry",
        ]
    ):

        policy_query = """
        Retrieve the company leave carry-forward policy.

        Explain:
        - eligible leave type
        - maximum carry-forward days
        - year-end rules
        - forfeiture rules
        """

    # ========================================================
    # GENERAL LEAVE POLICY
    # ========================================================

    elif any(
        keyword in q
        for keyword in [
            "leave policy",
            "leave policies",
            "leave entitlement",
            "leave rules",
            "leave approval",
            "leave application",
            "leave expires",
            "leave expiry",
            "leave expiration",
            "leave encashment",
        ]
    ):

        policy_query = """
        Retrieve the complete company leave policy.

        Include:
        - leave types
        - leave entitlement
        - leave application
        - approval
        - expiry
        - carry-forward
        - forfeiture
        - additional leave rules
        """

    # ========================================================
    # EXPENSE POLICY
    # ========================================================

    elif (
        "expense policy" in q
        or "reimbursement policy" in q
        or "expense reimbursement" in q
    ):

        policy_query = """
        Retrieve the company expense reimbursement policy.

        Include:
        - eligible expenses
        - reimbursement rules
        - approval
        - limits
        - required documentation
        - submission process
        """

    # ========================================================
    # GENERAL POLICY
    # ========================================================

    else:

        policy_query = f"""
        Retrieve the most relevant company policy for:

        {question}

        Return only information relevant to the question.
        """

    try:

        # ----------------------------------------------------
        # IMPORTANT:
        # retrieve_policy returns a dictionary such as:
        #
        # {
        #     "context": "...",
        #     "sources": [...]
        # }
        #
        # We extract ONLY the context.
        # ----------------------------------------------------

        result = retrieve_policy(
            policy_query
        )

        if not isinstance(
            result,
            dict,
        ):

            return clean_policy_answer(
                str(result)
            )

        context = result.get(
            "context",
            "",
        )

        if not context:

            return (
                "I couldn't find relevant information "
                "in the company policy documents."
            )

        # ----------------------------------------------------
        # Send only the retrieved policy context to Qwen.
        # ----------------------------------------------------

        return generate_policy_answer(
            question=question,
            context=str(context),
        )

    except Exception as e:

        print(
            f"[Policy search error] {e}"
        )

        return (
            "I couldn't retrieve the company policy "
            "information right now."
        )


# ============================================================
# EMPLOYEE WRAPPERS
# ============================================================

def get_my_employee_details(employee_id):

    try:

        return get_employee_details(
            employee_id
        )

    except Exception as e:

        return {
            "success": False,
            "data": None,
            "message": str(e),
        }


def get_my_leave_balance(employee_id):

    try:

        return get_leave_balance(
            employee_id
        )

    except Exception as e:

        return {
            "success": False,
            "data": None,
            "message": str(e),
        }


def get_my_expense_records(employee_id):

    try:

        return get_expense_records(
            employee_id
        )

    except Exception as e:

        return {
            "success": False,
            "data": [],
            "message": str(e),
        }


def get_my_total_expenses(employee_id):

    try:

        return get_total_expenses(
            employee_id
        )

    except Exception as e:

        return {
            "success": False,
            "data": None,
            "message": str(e),
        }


def get_my_expense_summary(employee_id):

    try:

        return get_expense_summary(
            employee_id
        )

    except Exception as e:

        return {
            "success": False,
            "data": None,
            "message": str(e),
        }


def get_my_it_assets(employee_id):

    try:

        return get_it_assets(
            employee_id
        )

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
    question: str,
    employee_id: str,
) -> Dict[str, Any]:

    q = normalize_question(question)

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

    # Normalize city names

    if city == "bengaluru":
        city = "bangalore"

    if city == "mysuru":
        city = "mysore"

    if city == "gurugram":
        city = "gurgaon"

    # --------------------------------------------------------
    # If no city was supplied, use employee location.
    # --------------------------------------------------------

    if city is None:

        employee_result = get_my_employee_details(
            employee_id
        )

        if employee_result.get(
            "success",
            False,
        ):

            employee_data = employee_result.get(
                "data"
            )

            if isinstance(
                employee_data,
                dict,
            ):

                location_id = employee_data.get(
                    "location_id"
                )

                if not is_empty_value(
                    location_id
                ):

                    location_map = {
                        "L001": "bangalore",
                        "L002": "chennai",
                        "L003": "bangalore",
                        "L004": "hyderabad",
                        "L005": "pune",
                    }

                    city = location_map.get(
                        str(location_id).upper()
                    )

    if city is None:

        return {
            "success": False,
            "data": None,
            "message": (
                "I couldn't determine your office location. "
                "Please specify a city."
            ),
        }

    try:

        return get_office_details(
            city
        )

    except Exception as e:

        return {
            "success": False,
            "data": None,
            "message": str(e),
        }


# ============================================================
# EXPENSE STATUS FILTER
# ============================================================

def filter_expenses_by_status(
    tool_result,
    status,
):

    if not tool_result.get(
        "success",
        False,
    ):

        return {
            "success": False,
            "data": [],
            "message": tool_result.get(
                "message",
                "Expense information could not be found.",
            ),
        }

    data = tool_result.get(
        "data"
    )

    if not isinstance(
        data,
        list,
    ):

        return {
            "success": False,
            "data": [],
            "message": (
                "Expense records could not be read."
            ),
        }

    target_status = status.lower()

    filtered = []

    for record in data:

        if not isinstance(
            record,
            dict,
        ):
            continue

        record_status = str(
            record.get(
                "status",
                "",
            )
        ).strip().lower()

        if record_status == target_status:

            filtered.append(
                record
            )

    return {
        "success": True,
        "data": filtered,
        "message": (
            f"{len(filtered)} {status} expense record(s) found."
        ),
    }


def get_my_expenses_by_status(
    employee_id,
    status,
):

    all_expenses = get_my_expense_records(
        employee_id
    )

    return filter_expenses_by_status(
        all_expenses,
        status,
    )


# ============================================================
# PENDING LEAVE
# ============================================================

def get_my_pending_leave(
    employee_id,
    question,
):

    leave_result = get_my_leave_balance(
        employee_id
    )

    if not leave_result.get(
        "success",
        False,
    ):

        return {
            "success": False,
            "data": None,
            "message": leave_result.get(
                "message",
                "Leave information could not be found.",
            ),
        }

    data = leave_result.get(
        "data"
    )

    # --------------------------------------------------------
    # If leave data is a list of requests
    # --------------------------------------------------------

    if isinstance(
        data,
        list,
    ):

        pending = []

        q = normalize_question(question)

        requested_type = None

        if "sick" in q:
            requested_type = "sick"

        elif "casual" in q:
            requested_type = "casual"

        elif "earned" in q:
            requested_type = "earned"

        for record in data:

            if not isinstance(
                record,
                dict,
            ):
                continue

            status = str(
                record.get(
                    "status",
                    "",
                )
            ).lower()

            if status != "pending":
                continue

            if requested_type:

                record_text = " ".join(
                    str(v).lower()
                    for v in record.values()
                )

                if requested_type not in record_text:
                    continue

            pending.append(
                record
            )

        return {
            "success": True,
            "data": pending,
            "message": (
                f"{len(pending)} pending leave request(s) found."
            ),
        }

    # --------------------------------------------------------
    # If dictionary contains pending fields
    # --------------------------------------------------------

    if isinstance(
        data,
        dict,
    ):

        pending_fields = [
            "pending",
            "pending_leave",
            "pending_leaves",
            "pending_requests",
            "leave_requests",
            "requests",
        ]

        for field in pending_fields:

            if field in data:

                return {
                    "success": True,
                    "data": data[field],
                    "message": (
                        "Pending leave information found."
                    ),
                }

    # --------------------------------------------------------
    # Current data contains only balances
    # --------------------------------------------------------

    return {
        "success": False,
        "data": None,
        "message": (
            "Pending leave request information is not "
            "available in the current employee data."
        ),
    }


# ============================================================
# EXECUTE INTENTS
# ============================================================

def execute_intents(
    question,
    employee_id,
):

    intents = detect_intents(
        question
    )

    if intents["unrelated"]:

        return {
            "type": "unrelated",
            "data": None,
        }

    results = {}

    # ========================================================
    # MANAGER
    # ========================================================

    if intents["manager_info"]:

        results["manager_info"] = (
            get_my_employee_details(
                employee_id
            )
        )

    # ========================================================
    # EMPLOYEE DETAILS
    # ========================================================

    elif intents["employee_details"]:

        results["employee_details"] = (
            get_my_employee_details(
                employee_id
            )
        )

    # ========================================================
    # PENDING LEAVE
    # ========================================================

    if intents["pending_leave"]:

        results["pending_leave"] = (
            get_my_pending_leave(
                employee_id,
                question,
            )
        )

    # ========================================================
    # SPECIFIC LEAVE BALANCE
    # ========================================================

    elif intents["specific_leave_balance"]:

        results["leave_balance"] = (
            get_my_leave_balance(
                employee_id
            )
        )

        results["specific_leave_type"] = True

    # ========================================================
    # GENERAL LEAVE BALANCE
    # ========================================================

    elif intents["leave_balance"]:

        results["leave_balance"] = (
            get_my_leave_balance(
                employee_id
            )
        )

    # ========================================================
    # LEAVE POLICY
    # ========================================================

    if intents["leave_policy"]:

        results["leave_policy"] = (
            search_company_policy(
                question
            )
        )

    # ========================================================
    # GENERAL POLICY
    #
    # leave_policy already handles the policy,
    # so do NOT retrieve twice.
    # ========================================================

    elif intents["policy"]:

        results["policy"] = (
            search_company_policy(
                question
            )
        )

    # ========================================================
    # EXPENSES
    # ========================================================

    if intents["pending_expenses"]:

        results["pending_expenses"] = (
            get_my_expenses_by_status(
                employee_id,
                "Pending",
            )
        )

    elif intents["approved_expenses"]:

        results["approved_expenses"] = (
            get_my_expenses_by_status(
                employee_id,
                "Approved",
            )
        )

    elif intents["rejected_expenses"]:

        results["rejected_expenses"] = (
            get_my_expenses_by_status(
                employee_id,
                "Rejected",
            )
        )

    elif intents["total_expenses"]:

        results["total_expenses"] = (
            get_my_total_expenses(
                employee_id
            )
        )

    elif intents["expense_summary"]:

        results["expense_summary"] = (
            get_my_expense_summary(
                employee_id
            )
        )

    elif intents["expenses"]:

        results["expenses"] = (
            get_my_expense_records(
                employee_id
            )
        )

    # ========================================================
    # IT ASSETS
    # ========================================================

    if intents["it_assets"]:

        results["it_assets"] = (
            get_my_it_assets(
                employee_id
            )
        )

    # ========================================================
    # OFFICE
    # ========================================================

    if intents["office_info"]:

        results["office_info"] = (
            get_my_office_information(
                question,
                employee_id,
            )
        )

    # ========================================================
    # NOTHING FOUND
    # ========================================================

    if not results:

        return {
            "type": "unrelated",
            "data": None,
        }

    # Store intent information separately.
    # This is important for specific asset/leave formatting.

    return {
        "type": "workplace",
        "data": results,
        "intents": intents,
    }


# ============================================================
# FORMAT LEAVE BALANCE
# ============================================================

def extract_leave_type(question):

    q = normalize_question(
        question
    )

    if "sick" in q:

        return "sick_leave"

    if "casual" in q:

        return "casual_leave"

    if "earned" in q:

        return "earned_leave"

    if "annual" in q:

        return "earned_leave"

    return None


def format_leave_balance(
    tool_result,
    question=None,
):

    if not tool_result.get(
        "success",
        False,
    ):

        return tool_result.get(
            "message",
            "Leave balance could not be found.",
        )

    data = tool_result.get(
        "data"
    )

    if not isinstance(
        data,
        dict,
    ):

        return (
            "I couldn't read the leave balance correctly."
        )

    # ========================================================
    # SPECIFIC LEAVE
    # ========================================================

    if question:

        requested_type = extract_leave_type(
            question
        )

        if requested_type:

            value = data.get(
                requested_type
            )

            if value is not None:

                days = safe_int(
                    value
                )

                label_map = {
                    "sick_leave": "Sick Leave",
                    "casual_leave": "Casual Leave",
                    "earned_leave": "Earned Leave",
                }

                label = label_map.get(
                    requested_type,
                    requested_type.replace(
                        "_",
                        " ",
                    ).title(),
                )

                return (
                    f"Your {label.lower()} balance is "
                    f"{days} {day_word(days)}."
                )

    # ========================================================
    # FULL BALANCE
    # ========================================================

    casual = safe_int(
        data.get(
            "casual_leave",
            0,
        )
    )

    earned = safe_int(
        data.get(
            "earned_leave",
            0,
        )
    )

    sick = safe_int(
        data.get(
            "sick_leave",
            0,
        )
    )

    if "total_leave" in data:

        total = safe_int(
            data.get(
                "total_leave"
            )
        )

    elif "total" in data:

        total = safe_int(
            data.get(
                "total"
            )
        )

    else:

        total = casual + earned + sick

    return (
        "Your leave balance is:\n"
        f"- Casual Leave: {casual} {day_word(casual)}\n"
        f"- Earned Leave: {earned} {day_word(earned)}\n"
        f"- Sick Leave: {sick} {day_word(sick)}\n"
        f"- Total: {total} {day_word(total)}"
    )


# ============================================================
# FORMAT PENDING LEAVE
# ============================================================

def format_pending_leave(
    tool_result,
    question,
):

    if not tool_result.get(
        "success",
        False,
    ):

        return tool_result.get(
            "message",
            "Pending leave information is not available.",
        )

    data = tool_result.get(
        "data"
    )

    if not data:

        if "sick" in normalize_question(
            question
        ):

            return "You have no pending sick leave requests."

        return "You have no pending leave requests."

    if isinstance(
        data,
        list,
    ):

        lines = [
            "Your pending leave requests:"
        ]

        for record in data:

            if not isinstance(
                record,
                dict,
            ):
                continue

            parts = []

            for key, value in record.items():

                if is_empty_value(value):
                    continue

                label = key.replace(
                    "_",
                    " ",
                ).title()

                parts.append(
                    f"{label}: {value}"
                )

            if parts:

                lines.append(
                    "- " + ", ".join(parts)
                )

        return "\n".join(
            lines
        )

    if isinstance(
        data,
        dict,
    ):

        lines = [
            "Your pending leave information:"
        ]

        for key, value in data.items():

            if is_empty_value(value):
                continue

            label = key.replace(
                "_",
                " ",
            ).title()

            lines.append(
                f"- {label}: {value}"
            )

        return "\n".join(lines)

    return str(data)


# ============================================================
# FORMAT EMPLOYEE DETAILS
# ============================================================

def format_employee_details(
    tool_result,
):

    if not tool_result.get(
        "success",
        False,
    ):

        return tool_result.get(
            "message",
            "Employee details could not be found.",
        )

    data = tool_result.get(
        "data"
    )

    if not isinstance(
        data,
        dict,
    ):

        return (
            "I couldn't read the employee details."
        )

    lines = [
        "Here are your employee details:"
    ]

    preferred_order = [
        "employee_id",
        "name",
        "email",
        "department",
        "designation",
        "role",
        "manager",
        "manager_name",
        "manager_id",
        "joining_date",
        "location_id",
        "employment_status",
    ]

    used = set()

    for key in preferred_order:

        if key not in data:
            continue

        value = data.get(
            key
        )

        if is_empty_value(value):
            continue

        label = key.replace(
            "_",
            " ",
        ).title()

        lines.append(
            f"- {label}: {value}"
        )

        used.add(key)

    for key, value in data.items():

        if key in used:
            continue

        if is_empty_value(value):
            continue

        label = key.replace(
            "_",
            " ",
        ).title()

        lines.append(
            f"- {label}: {value}"
        )

    return "\n".join(lines)


# ============================================================
# FORMAT MANAGER
# ============================================================

def format_manager_information(
    tool_result,
):

    if not tool_result.get(
        "success",
        False,
    ):

        return tool_result.get(
            "message",
            "Manager information could not be found.",
        )

    data = tool_result.get(
        "data"
    )

    if not isinstance(
        data,
        dict,
    ):

        return (
            "Manager information could not be found."
        )

    manager_name = data.get(
        "manager_name"
    )

    if not is_empty_value(
        manager_name
    ):

        return (
            f"Your manager is {manager_name}."
        )

    manager = data.get(
        "manager"
    )

    if not is_empty_value(
        manager
    ):

        return (
            f"Your manager is {manager}."
        )

    manager_id = data.get(
        "manager_id"
    )

    if not is_empty_value(
        manager_id
    ):

        return (
            f"Your manager's employee ID is "
            f"{manager_id}. The manager name is "
            f"not available in the employee data."
        )

    return (
        "No manager is listed in your employee record."
    )


# ============================================================
# FORMAT EXPENSES
# ============================================================

def format_expense_records(
    tool_result,
    heading="Here are your expense records:",
):

    if not tool_result.get(
        "success",
        False,
    ):

        return tool_result.get(
            "message",
            "No expense information was found.",
        )

    data = tool_result.get(
        "data"
    )

    if not isinstance(
        data,
        list,
    ):

        return (
            "I couldn't read the expense records correctly."
        )

    if not data:

        return (
            "No matching expense records were found."
        )

    lines = [
        heading
    ]

    for record in data:

        if not isinstance(
            record,
            dict,
        ):
            continue

        expense_id = record.get(
            "expense_id",
            record.get(
                "id",
                "N/A",
            ),
        )

        category = record.get(
            "category",
            record.get(
                "expense_type",
                "Expense",
            ),
        )

        status = record.get(
            "status",
            "Unknown",
        )

        amount = record.get(
            "amount",
            0,
        )

        date = record.get(
            "date",
            record.get(
                "expense_date",
                "N/A",
            ),
        )

        try:

            amount_text = (
                f"{float(amount):,.2f}"
            )

        except (
            TypeError,
            ValueError,
        ):

            amount_text = str(
                amount
            )

        lines.append(
            f"- {category} "
            f"({expense_id}): "
            f"{status}, "
            f"{amount_text} "
            f"({date})."
        )

    return "\n".join(lines)


# ============================================================
# FORMAT TOTAL EXPENSES
# ============================================================

def format_total_expenses(
    tool_result,
):

    if not tool_result.get(
        "success",
        False,
    ):

        return tool_result.get(
            "message",
            "No expense information was found.",
        )

    data = tool_result.get(
        "data"
    )

    if isinstance(
        data,
        dict,
    ):

        total = data.get(
            "total_expenses",
            data.get(
                "total",
                0,
            ),
        )

    else:

        total = data

    try:

        return (
            f"Your total expenses are "
            f"{float(total):,.2f}."
        )

    except (
        TypeError,
        ValueError,
    ):

        return (
            f"Your total expenses are {total}."
        )


# ============================================================
# FORMAT EXPENSE SUMMARY
# ============================================================

def format_expense_summary(
    tool_result,
):

    if not tool_result.get(
        "success",
        False,
    ):

        return tool_result.get(
            "message",
            "Expense summary could not be found.",
        )

    data = tool_result.get(
        "data"
    )

    if not isinstance(
        data,
        dict,
    ):

        return (
            "I couldn't read the expense summary correctly."
        )

    total = data.get(
        "total_expenses",
        data.get(
            "total",
            0,
        ),
    )

    approved = data.get(
        "approved",
        0,
    )

    pending = data.get(
        "pending",
        0,
    )

    rejected = data.get(
        "rejected",
        0,
    )

    def money(value):

        try:
            return f"{float(value):,.2f}"
        except (
            TypeError,
            ValueError,
        ):
            return str(value)

    return (
        "Your expense summary is:\n"
        f"- Total expenses: {money(total)}\n"
        f"- Approved: {money(approved)}\n"
        f"- Pending: {money(pending)}\n"
        f"- Rejected: {money(rejected)}"
    )


# ============================================================
# FORMAT IT ASSETS
# ============================================================

def format_it_assets(
    tool_result,
    specific=False,
):

    if not tool_result.get(
        "success",
        False,
    ):

        return tool_result.get(
            "message",
            "IT asset information could not be found.",
        )

    data = tool_result.get(
        "data"
    )

    if not isinstance(
        data,
        list,
    ):

        return (
            "I couldn't read the IT asset information."
        )

    if not data:

        return (
            "No IT assets are assigned to you."
        )

    # ========================================================
    # SPECIFIC ASSET NAME
    # ========================================================

    if specific:

        first_asset = data[0]

        if isinstance(
            first_asset,
            dict,
        ):

            asset_name = first_asset.get(
                "asset_name"
            )

            if not is_empty_value(
                asset_name
            ):

                return (
                    f"Your assigned asset is "
                    f"{asset_name}."
                )

    # ========================================================
    # FULL ASSET DETAILS
    # ========================================================

    lines = [
        "Here are your assigned IT assets:"
    ]

    for asset in data:

        if not isinstance(
            asset,
            dict,
        ):
            continue

        parts = []

        for key, value in asset.items():

            if is_empty_value(value):
                continue

            label = key.replace(
                "_",
                " ",
            ).title()

            parts.append(
                f"{label}: {value}"
            )

        if parts:

            lines.append(
                "- " + ", ".join(parts)
            )

    return "\n".join(lines)


# ============================================================
# FORMAT OFFICE
# ============================================================

def format_office_information(
    tool_result,
):

    if not tool_result.get(
        "success",
        False,
    ):

        return tool_result.get(
            "message",
            "Office information could not be found.",
        )

    data = tool_result.get(
        "data"
    )

    if not isinstance(
        data,
        dict,
    ):

        return (
            "I couldn't read the office information."
        )

    lines = [
        "Here is the office information:"
    ]

    for key, value in data.items():

        if is_empty_value(value):
            continue

        label = key.replace(
            "_",
            " ",
        ).title()

        lines.append(
            f"- {label}: {value}"
        )

    return "\n".join(lines)


# ============================================================
# GENERATE FINAL ANSWER
# ============================================================

def generate_final_answer(
    question,
    result,
):

    if result.get(
        "type"
    ) == "unrelated":

        return (
            "I can help with Blue Orbit workplace questions "
            "such as employee details, leave balance and "
            "policies, expenses, IT assets, and office "
            "information."
        )

    data = result.get(
        "data",
        {},
    )

    intents = result.get(
        "intents",
        detect_intents(question),
    )

    if not isinstance(
        data,
        dict,
    ):

        return (
            "I couldn't find usable company information "
            "for that question."
        )

    # ========================================================
    # MANAGER ONLY
    # ========================================================

    if (
        "manager_info" in data
        and len(data) == 1
    ):

        return format_manager_information(
            data["manager_info"]
        )

    # ========================================================
    # EMPLOYEE DETAILS ONLY
    # ========================================================

    if (
        "employee_details" in data
        and len(data) == 1
    ):

        return format_employee_details(
            data["employee_details"]
        )

    # ========================================================
    # PENDING LEAVE ONLY
    # ========================================================

    if (
        "pending_leave" in data
        and len(data) == 1
    ):

        return format_pending_leave(
            data["pending_leave"],
            question,
        )

    # ========================================================
    # EXPENSE STATUS
    # ========================================================

    if (
        "pending_expenses" in data
        and len(data) == 1
    ):

        return format_expense_records(
            data["pending_expenses"],
            "Here are your pending expenses:",
        )

    if (
        "approved_expenses" in data
        and len(data) == 1
    ):

        return format_expense_records(
            data["approved_expenses"],
            "Here are your approved expenses:",
        )

    if (
        "rejected_expenses" in data
        and len(data) == 1
    ):

        return format_expense_records(
            data["rejected_expenses"],
            "Here are your rejected expenses:",
        )

    # ========================================================
    # TOTAL EXPENSES
    # ========================================================

    if (
        "total_expenses" in data
        and len(data) == 1
    ):

        return format_total_expenses(
            data["total_expenses"]
        )

    # ========================================================
    # EXPENSE SUMMARY
    # ========================================================

    if (
        "expense_summary" in data
        and len(data) == 1
    ):

        return format_expense_summary(
            data["expense_summary"]
        )

    # ========================================================
    # GENERAL EXPENSES
    # ========================================================

    if (
        "expenses" in data
        and len(data) == 1
    ):

        return format_expense_records(
            data["expenses"]
        )

    # ========================================================
    # IT ASSETS
    # ========================================================

    if (
        "it_assets" in data
        and len(data) == 1
    ):

        return format_it_assets(
            data["it_assets"],
            specific=intents.get(
                "specific_asset",
                False,
            ),
        )

    # ========================================================
    # OFFICE
    # ========================================================

    if (
        "office_info" in data
        and len(data) == 1
    ):

        return format_office_information(
            data["office_info"]
        )

    # ========================================================
    # MULTI-INTENT ANSWER
    # ========================================================

    answer_parts = []

    # --------------------------------------------------------
    # LEAVE BALANCE
    # --------------------------------------------------------

    if "leave_balance" in data:

        leave_answer = format_leave_balance(
            data["leave_balance"],
            question,
        )

        answer_parts.append(
            leave_answer
        )

    # --------------------------------------------------------
    # LEAVE POLICY
    # --------------------------------------------------------

    if "leave_policy" in data:

        policy_answer = str(
            data["leave_policy"]
        ).strip()

        if policy_answer:

            answer_parts.append(
                "Leave policy:\n"
                + policy_answer
            )

    # --------------------------------------------------------
    # GENERAL POLICY
    # --------------------------------------------------------

    if "policy" in data:

        policy_answer = str(
            data["policy"]
        ).strip()

        if policy_answer:

            answer_parts.append(
                policy_answer
            )

    # --------------------------------------------------------
    # If we have deterministic answers, return them.
    # --------------------------------------------------------

    if answer_parts:

        return "\n\n".join(
            answer_parts
        )

    # ========================================================
    # FINAL LLM FALLBACK
    # ========================================================

    context = str(
        data
    )

    prompt = f"""
You are the Blue Orbit Office Assistant.

Answer the employee's question using ONLY the verified
company information below.

Rules:

- Do not invent information.
- Do not mention internal implementation details.
- Do not output Python dictionaries.
- Do not mention retrieval, FAISS, embeddings, or chunks.
- Do not add unrelated information.
- Do not add filler such as:
  "No additional information is provided."
- Give a concise direct answer.

USER QUESTION:
{question}

VERIFIED INFORMATION:
{context}

Give only the final answer.
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

        return clean_policy_answer(
            answer
        )

    except Exception as e:

        print(
            f"[Qwen error] {e}"
        )

        return (
            "Here is the available company information:\n"
            f"{context}"
        )


# ============================================================
# LANGGRAPH NODE
# ============================================================

def agent_node(
    state: AgentState,
) -> AgentState:

    messages = state.get(
        "messages",
        []
    )

    employee_id = state.get(
        "employee_id",
        "EMP001",
    )

    question = ""

    for message in reversed(
        messages
    ):

        if isinstance(
            message,
            HumanMessage,
        ):

            question = str(
                message.content
            ).strip()

            break

    if not question:

        return {
            "final_answer":
                "Please enter a workplace-related question."
        }

    result = execute_intents(
        question=question,
        employee_id=employee_id,
    )

    answer = generate_final_answer(
        question=question,
        result=result,
    )

    return {
        "final_answer": answer
    }


# ============================================================
# LANGGRAPH
# ============================================================

workflow = StateGraph(
    AgentState
)

workflow.add_node(
    "agent",
    agent_node,
)

workflow.add_edge(
    START,
    "agent",
)

workflow.add_edge(
    "agent",
    END,
)


# ============================================================
# MEMORY
# ============================================================

memory = MemorySaver()


# ============================================================
# COMPILE
# ============================================================

app = workflow.compile(
    checkpointer=memory
)


# ============================================================
# PUBLIC API
# ============================================================

def ask_employee(
    question: str,
    employee_id: str = "EMP001",
    thread_id: str = "default",
) -> str:

    if (
        not question
        or not question.strip()
    ):

        return "Please enter a question."

    initial_state: AgentState = {
        "messages": [
            HumanMessage(
                content=question.strip()
            )
        ],
        "employee_id": employee_id,
    }

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    try:

        result = app.invoke(
            initial_state,
            config=config,
        )

        answer = result.get(
            "final_answer",
            "",
        )

        if not answer:

            return (
                "I couldn't generate an answer. "
                "Please try asking the question another way."
            )

        return str(
            answer
        ).strip()

    except Exception as e:

        print(
            f"[Blue Orbit Agent Error] {e}"
        )

        return (
            "Sorry, I ran into a problem while processing "
            "your question. Please try again."
        )


# ============================================================
# LOCAL TEST
# ============================================================

if __name__ == "__main__":

    tests = [

        # -----------------------------------------------
        # EXPENSES
        # -----------------------------------------------

        "what is expense record?",

        "pending expenses?",

        "rejected expense",

        "approved expenses?",

        "what is my total expense?",

        # -----------------------------------------------
        # EMPLOYEE
        # -----------------------------------------------

        "what are my employee details?",

        "who is my manager?",

        # -----------------------------------------------
        # OFFICE
        # -----------------------------------------------

        "office location",

        "office location?",

        "offfice loaction",

        "bangalore office location",

        "working hours?",

        # -----------------------------------------------
        # IT ASSETS
        # -----------------------------------------------

        "it assets?",

        "asset name?",

        "assest name?",

        "what laptop do i have?",

        # -----------------------------------------------
        # LEAVE
        # -----------------------------------------------

        "leave details?",

        "what is my leave details?",

        "what is my sick leave balance?",

        "what is my earned leave balance?",

        "what is my casual leave balance?",

        "what is my leave balance?",

        # -----------------------------------------------
        # LEAVE POLICY
        # -----------------------------------------------

        "can i carry forward my leaves?",

        "what is the leave policy?",

        "what is the work-from-home policy?",

        # -----------------------------------------------
        # MULTI INTENT
        # -----------------------------------------------

        "what is my leave balance and what is the leave policy?",
    ]

    print("=" * 70)
    print("BLUE ORBIT OFFICE ASSISTANT")
    print("=" * 70)

    for question in tests:

        print("\nUSER:")
        print(question)

        answer = ask_employee(
            question=question,
            employee_id="EMP054",
            thread_id="local_test",
        )

        print("\nASSISTANT:")
        print(answer)

        print("-" * 70)