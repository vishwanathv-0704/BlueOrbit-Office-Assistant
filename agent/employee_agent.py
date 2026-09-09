
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver


# ============================================================
# EMPLOYEE TOOLS
# ============================================================

from tools.employee_tools_clean import (
    get_employee_details,
    get_leave_balance,
    get_expense_records,
    get_total_expenses,
    get_expense_summary,
    get_it_assets,
    get_office_details,
)

import tools.employee_tools_clean as employee_tools_module


# ============================================================
# POLICY RAG
# ============================================================

try:
    from rag.retrieval import retrieve_policy
except ImportError:
    try:
        from rag.retrieval import search_policy as retrieve_policy
    except ImportError:
        retrieve_policy = None


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_EMPLOYEE_ID = "EMP054"

BASE_DIR = Path(__file__).resolve().parent.parent

LEAVE_FILE = (
    BASE_DIR
    / "data"
    / "csvs"
    / "leave_balance.csv"
)


# ============================================================
# STATE
# ============================================================

class AgentState(TypedDict, total=False):
    question: str
    employee_id: str
    final_answer: str

    # Used for conversational leave requests.
    pending_leave_days: int


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(text: str) -> str:
    """
    Normalize user text and fix common typos.

    IMPORTANT:
    Do not use:
        text.replace("employe", "employee")

    because "employee" itself contains "employe".
    """

    if not text:
        return ""

    text = str(text).strip().lower()

    typo_map = {
        r"\bempolyee\b": "employee",
        r"\bemploye\b": "employee",
        r"\bmaneger\b": "manager",
        r"\bexpence\b": "expense",
        r"\bexpences\b": "expenses",
        r"\bbanglore\b": "bangalore",
        r"\bbangalor\b": "bangalore",
        r"\bleavee\b": "leave",
        r"\battendence\b": "attendance",
    }

    for pattern, replacement in typo_map.items():
        text = re.sub(
            pattern,
            replacement,
            text,
            flags=re.IGNORECASE,
        )

    text = re.sub(r"\s+", " ", text).strip()

    return text


# ============================================================
# TOOL RESULT HELPERS
# ============================================================

def unwrap_tool_result(
    result: Any,
    default: Any = None,
) -> Any:

    if result is None:
        return default

    if isinstance(result, dict):

        if "data" in result:
            return result.get(
                "data",
                default,
            )

        return result

    return result


def tool_success(result: Any) -> bool:

    if isinstance(result, dict):
        return result.get(
            "success",
            True,
        )

    return True


def first_value(
    data: Dict[str, Any],
    *keys: str,
    default: Any = "Not available",
) -> Any:

    if not isinstance(data, dict):
        return default

    for key in keys:

        value = data.get(key)

        if value is None:
            continue

        if str(value).strip().lower() in {
            "",
            "nan",
            "none",
            "null",
            "n/a",
            "na",
        }:
            continue

        return value

    return default


def clean_number(value: Any) -> str:

    try:

        number = float(value)

        if number.is_integer():
            return str(int(number))

        return f"{number:.2f}"

    except Exception:

        return str(value)


def safe_int(
    value: Any,
    default: int = 0,
) -> int:

    try:
        return int(float(value))

    except (TypeError, ValueError):
        return default


def day_word(value: Any) -> str:

    number = safe_int(value)

    return "day" if number == 1 else "days"


# ============================================================
# GREETING
# ============================================================

def is_greeting(text: str) -> bool:

    text = normalize_text(text)

    text = text.rstrip("!.,? ")

    return text in {
        "hi",
        "hello",
        "hey",
        "hiya",
        "good morning",
        "good afternoon",
        "good evening",
    }


def greeting_response() -> str:

    return (
        "Hello! 👋\n\n"
        "How can I help you with Blue Orbit today?"
    )


# ============================================================
# SPECIAL REQUESTS
# ============================================================

def is_self_harm(text: str) -> bool:

    patterns = [
        r"\bi wanna die\b",
        r"\bi want to die\b",
        r"\bi want die\b",
        r"\bi want to kill myself\b",
        r"\bkill myself\b",
        r"\bend my life\b",
        r"\bsuicide\b",
        r"\bcommit suicide\b",
        r"\bhurt myself\b",
        r"\bwant to hurt myself\b",
    ]

    return any(
        re.search(pattern, text)
        for pattern in patterns
    )


def self_harm_response() -> str:

    return (
        "I'm really sorry you're going through this. "
        "You don't have to handle it alone.\n\n"
        "- If you may hurt yourself right now, call **112** in India or go to the nearest emergency department.\n"
        "- You can call **Tele-MANAS: 14416** for mental-health support.\n"
        "- If possible, stay with someone you trust and move away from anything you could use to hurt yourself."
    )


def is_password_help(text: str) -> bool:

    patterns = [
        r"\bhow to change password\b",
        r"\bhow do i change my password\b",
        r"\bchange my password\b",
        r"\breset my password\b",
        r"\bforgot my password\b",
        r"\bpassword reset\b",
        r"\bpassword change\b",
    ]

    return any(
        re.search(pattern, text)
        for pattern in patterns
    )


def password_response() -> str:

    return (
        "**Password Change**\n\n"
        "- Open the company account/security portal.\n"
        "- Go to **Security** or **Account Settings**.\n"
        "- Select **Change Password**.\n"
        "- Enter your current password.\n"
        "- Enter and confirm your new password.\n"
        "- Save the changes.\n"
        "- If you cannot access your account, use the official password-reset option or contact IT support."
    )


def is_confidential_request(text: str) -> bool:

    if is_password_help(text):
        return False

    patterns = [
        r"\bconfidential file\b",
        r"\bconfidential files\b",
        r"\bprivate file\b",
        r"\bprivate files\b",
        r"\bsecret file\b",
        r"\bsecret files\b",
        r"\bapi key\b",
        r"\bapi keys\b",
        r"\baccess token\b",
        r"\baccess tokens\b",
        r"\bcredentials\b",
        r"\badmin password\b",
        r"\bdatabase password\b",
        r"\bbypass security\b",
        r"\bhack\b",
        r"\bprivate employee data\b",
    ]

    return any(
        re.search(pattern, text)
        for pattern in patterns
    )


def confidential_response() -> str:

    return (
        "**Access Restricted**\n\n"
        "- I can't provide confidential files, passwords, credentials, API keys, tokens, or private security information.\n"
        "- For authorized access, contact the appropriate IT/HR administrator."
    )


# ============================================================
# QUESTION SPLITTING
# ============================================================

def split_questions(text: str) -> List[str]:

    if not text:
        return []

    text = str(text).strip()

    # --------------------------------------------------------
    # STEP 1: HARD SEPARATORS
    # --------------------------------------------------------

    # New lines
    text = re.sub(
        r"[\r\n]+",
        "?",
        text,
    )

    # Question / exclamation marks
    text = re.sub(
        r"[?!]+",
        "?",
        text,
    )

    # Slash / backslash / pipe / semicolon
    text = re.sub(
        r"[;/\\|]+",
        "?",
        text,
    )

    # Periods, but not decimal numbers such as 3.5
    text = re.sub(
        r"(?<!\d)\.(?!\d)",
        "?",
        text,
    )

    # --------------------------------------------------------
    # STEP 2: COMMA AS QUESTION SEPARATOR
    # --------------------------------------------------------

    text = re.sub(
        r",\s*(?="
        r"(?:what|why|how|where|who|when|which|whose|"
        r"can|could|would|should|is|are|do|does|did|"
        r"show|give|tell|list|"
        r"employee|manager|leave|expense|expenses|"
        r"laptop|asset|assets|office|"
        r"wfh|attendance|holiday|security|"
        r"pending|approved|rejected)"
        r"\b)",
        "?",
        text,
        flags=re.IGNORECASE,
    )

    # --------------------------------------------------------
    # STEP 3: SPLIT
    # --------------------------------------------------------

    parts = re.split(
        r"\?",
        text,
    )

    result: List[str] = []

    for part in parts:

        part = part.strip(
            " \t\r\n,.;!?/\\|:-"
        )

        if not part:
            continue

        result.append(part)

    # --------------------------------------------------------
    # STEP 4: AND / ALSO / PLUS
    # --------------------------------------------------------

    final: List[str] = []

    for part in result:

        subparts = re.split(
            r"\s+"
            r"(?:and\s+also|also|plus|and)"
            r"\s+"
            r"(?="
            r"(?:what|why|how|where|who|when|which|whose|"
            r"can|could|would|should|is|are|do|does|did|"
            r"show|give|tell|list|"
            r"employee|manager|leave|expense|expenses|"
            r"laptop|asset|assets|office|"
            r"wfh|attendance|holiday|security|"
            r"pending|approved|rejected)"
            r"\b)",
            part,
            flags=re.IGNORECASE,
        )

        for subpart in subparts:

            subpart = subpart.strip(
                " \t\r\n,.;!?/\\|:-"
            )

            if subpart:
                final.append(subpart)

    # --------------------------------------------------------
    # STEP 5: REMOVE DUPLICATES
    # --------------------------------------------------------

    unique_questions: List[str] = []

    seen = set()

    for question in final:

        normalized = normalize_text(
            question
        )

        if normalized and normalized not in seen:

            seen.add(normalized)

            unique_questions.append(
                question
            )

    return unique_questions


# ============================================================
# NUMBER EXTRACTION
# ============================================================

NUMBER_WORDS = {
    "one": 1,
    "a": 1,
    "an": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
}


def extract_leave_days(
    question: str,
) -> Optional[int]:

    text = normalize_text(question)

    # --------------------------------------------------------
    # Numeric form
    # --------------------------------------------------------

    patterns = [
        r"\b(\d+)\s*(?:days?|leaves?)\b",
        r"\bfor\s+(\d+)\b",
        r"\bneed\s+(\d+)\b",
        r"\btake\s+(\d+)\b",
        r"\btaking\s+(\d+)\b",
        r"\bwant\s+to\s+take\s+(\d+)\b",
        r"\bwant\s+(\d+)\b",
        r"\brequest\s+(\d+)\b",
        r"\bapply\s+for\s+(\d+)\b",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )

        if match:

            days = safe_int(
                match.group(1),
                0,
            )

            if days > 0:
                return days

    # --------------------------------------------------------
    # Word form
    # --------------------------------------------------------

    for word, number in NUMBER_WORDS.items():

        patterns = [
            rf"\b{re.escape(word)}\s+(?:days?|leaves?)\b",
            rf"\bneed\s+{re.escape(word)}\b",
            rf"\btake\s+{re.escape(word)}\b",
            rf"\btaking\s+{re.escape(word)}\b",
            rf"\bwant\s+{re.escape(word)}\b",
        ]

        for pattern in patterns:

            if re.search(
                pattern,
                text,
                flags=re.IGNORECASE,
            ):
                return number

    return None


# ============================================================
# LEAVE TYPE EXTRACTION
# ============================================================

def extract_leave_type(
    question: str,
) -> Optional[str]:

    text = normalize_text(question)

    # Sick leave
    if (
        "sick leave" in text
        or "sick balance" in text
        or re.search(r"\bsick\b", text)
    ):
        return "sick_leave"

    # Casual leave
    if (
        "casual leave" in text
        or "casual balance" in text
        or re.search(r"\bcasual\b", text)
    ):
        return "casual_leave"

    # Earned leave
    if (
        "earned leave" in text
        or "earned balance" in text
        or re.search(r"\bearned\b", text)
    ):
        return "earned_leave"

    # Annual leave
    if (
        "annual leave" in text
        or "annual balance" in text
        or re.search(r"\bannual\b", text)
    ):
        return "annual_leave"

    # CL / EL / SL
    if re.search(r"\bcl\b", text):
        return "casual_leave"

    if re.search(r"\bel\b", text):
        return "earned_leave"

    if re.search(r"\bsl\b", text):
        return "sick_leave"

    return None


def leave_type_display(
    leave_type: str,
) -> str:

    labels = {
        "casual_leave": "Casual Leave",
        "earned_leave": "Earned Leave",
        "sick_leave": "Sick Leave",
        "annual_leave": "Annual Leave",
    }

    return labels.get(
        leave_type,
        leave_type.replace("_", " ").title(),
    )


# ============================================================
# LEAVE COLUMN HELPERS
# ============================================================

LEAVE_COLUMN_ALIASES = {

    "casual_leave": [
        "casual_leave",
        "casual",
        "cl",
        "CL",
    ],

    "earned_leave": [
        "earned_leave",
        "earned",
        "el",
        "EL",
    ],

    "sick_leave": [
        "sick_leave",
        "sick",
        "sl",
        "SL",
    ],

    "annual_leave": [
        "annual_leave",
        "annual",
        "al",
        "AL",
    ],
}


def find_leave_column(
    data: Dict[str, Any],
    leave_type: str,
) -> Optional[str]:

    aliases = LEAVE_COLUMN_ALIASES.get(
        leave_type,
        [],
    )

    for key in aliases:

        if key in data:
            return key

    return None


# ============================================================
# LEAVE BALANCE
# ============================================================

def answer_leave_balance(
    employee_id: str,
) -> str:

    result = get_leave_balance(
        employee_id
    )

    if not tool_success(result):
        return "- Leave balance could not be retrieved."

    data = unwrap_tool_result(
        result,
        {},
    )

    if not isinstance(data, dict):
        return "- Leave balance is not available."

    casual = first_value(
        data,
        "casual_leave",
        "casual",
        "cl",
        "CL",
        default=0,
    )

    earned = first_value(
        data,
        "earned_leave",
        "earned",
        "el",
        "EL",
        default=0,
    )

    sick = first_value(
        data,
        "sick_leave",
        "sick",
        "sl",
        "SL",
        default=0,
    )

    annual = first_value(
        data,
        "annual_leave",
        "annual",
        "al",
        "AL",
        default=None,
    )

    total = (
        safe_int(casual)
        + safe_int(earned)
        + safe_int(sick)
    )

    if annual is not None:
        total += safe_int(annual)

    lines = [
        "**Leave Balance**",
        "",
        f"- **Casual Leave (CL):** {clean_number(casual)}",
        f"- **Earned Leave (EL):** {clean_number(earned)}",
        f"- **Sick Leave (SL):** {clean_number(sick)}",
    ]

    if annual is not None:
        lines.append(
            f"- **Annual Leave:** {clean_number(annual)}"
        )

    lines.append(
        f"- **Total Leave:** {clean_number(total)}"
    )

    return "\n".join(lines)


def answer_specific_leave(
    employee_id: str,
    question: str,
) -> str:

    result = get_leave_balance(
        employee_id
    )

    if not tool_success(result):
        return "- Leave balance could not be retrieved."

    data = unwrap_tool_result(
        result,
        {},
    )

    if not isinstance(data, dict):
        return "- Leave balance is not available."

    leave_type = extract_leave_type(
        question
    )

    if not leave_type:
        return answer_leave_balance(
            employee_id
        )

    column = find_leave_column(
        data,
        leave_type,
    )

    if not column:
        return (
            f"- {leave_type_display(leave_type)} "
            "balance is not available."
        )

    value = data.get(
        column,
        0,
    )

    return (
        f"**{leave_type_display(leave_type)}**\n\n"
        f"- **Available:** {clean_number(value)}"
    )


# ============================================================
# LEAVE REQUEST DETECTION
# ============================================================

def is_leave_request(
    question: str,
) -> bool:

    text = normalize_text(question)

    # Don't treat balance questions as requests.
    if any(
        phrase in text
        for phrase in [
            "leave balance",
            "remaining leave",
            "available leave",
            "how much leave",
            "how many leaves",
        ]
    ):
        return False

    request_patterns = [

        r"\bi need .*leave\b",
        r"\bi need .* day",
        r"\bi want .*leave\b",
        r"\bi want .* day",
        r"\bi would like .*leave\b",
        r"\bi would like .* day",

        r"\bi am taking .*leave\b",
        r"\bi'm taking .*leave\b",
        r"\bi am taking .* day",
        r"\bi'm taking .* day",

        r"\btake .*leave\b",
        r"\brequest .*leave\b",
        r"\bapply .*leave\b",
        r"\bapply for .*leave\b",

        r"\bneed leave\b",
        r"\bwant leave\b",
        r"\btaking leave\b",
        r"\brequesting leave\b",

        # Number + leave type
        r"\b\d+\s+(?:days?|leaves?)\b",
        r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten)"
        r"\s+(?:days?|leaves?)\b",
    ]

    return any(
        re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )
        for pattern in request_patterns
    )


# ============================================================
# LEAVE TYPE QUESTION
# ============================================================

def leave_type_question(
    days: int,
) -> str:

    return (
        f"Sure! You want to take **{days} {day_word(days)}** of leave.\n\n"
        "Which type of leave would you like to use?\n\n"
        "- **Casual Leave (CL)**\n"
        "- **Earned Leave (EL)**\n"
        "- **Sick Leave (SL)**"
    )


# ============================================================
# APPLY / DEDUCT LEAVE
# ============================================================

def apply_leave(
    employee_id: str,
    leave_type: str,
    days: int,
) -> str:

    if days <= 0:
        return "- Please provide a valid number of leave days."

    # --------------------------------------------------------
    # Get current balance
    # --------------------------------------------------------

    result = get_leave_balance(
        employee_id
    )

    if not tool_success(result):
        return "- I couldn't find your leave balance."

    data = unwrap_tool_result(
        result,
        {},
    )

    if not isinstance(data, dict):
        return "- I couldn't read your leave balance."

    # --------------------------------------------------------
    # Find actual CSV column
    # --------------------------------------------------------

    column = find_leave_column(
        data,
        leave_type,
    )

    if not column:
        return (
            f"- {leave_type_display(leave_type)} "
            "is not available for this employee."
        )

    current_balance = safe_int(
        data.get(column),
        0,
    )

    # --------------------------------------------------------
    # Check sufficient balance
    # --------------------------------------------------------

    if days > current_balance:

        return (
            "**Leave Request Cannot Be Approved**\n\n"
            f"- **Requested:** {days} {day_word(days)}\n"
            f"- **Leave Type:** {leave_type_display(leave_type)}\n"
            f"- **Available:** {current_balance} {day_word(current_balance)}\n\n"
            "- You do not have enough balance for this request."
        )

    # --------------------------------------------------------
    # Calculate new balance
    # --------------------------------------------------------

    new_balance = current_balance - days

    # --------------------------------------------------------
    # Update pandas dataframe in memory
    # --------------------------------------------------------

    leave_df = employee_tools_module.leave_df

    employee_rows = leave_df[
        leave_df["employee_id"]
        .astype(str)
        .str.strip()
        .str.upper()
        == str(employee_id).strip().upper()
    ]

    if employee_rows.empty:
        return (
            "- Leave record could not be updated because "
            "the employee was not found."
        )

    row_index = employee_rows.index[0]

    leave_df.at[
        row_index,
        column,
    ] = new_balance

    # --------------------------------------------------------
    # Update total leave
    # --------------------------------------------------------

    total_columns = [
        "total_leave",
        "total_leaves",
        "total",
    ]

    total_column = None

    for possible_column in total_columns:

        if possible_column in leave_df.columns:

            total_column = possible_column
            break

    if total_column:

        old_total = safe_int(
            leave_df.at[
                row_index,
                total_column,
            ],
            0,
        )

        new_total = max(
            old_total - days,
            0,
        )

        leave_df.at[
            row_index,
            total_column,
        ] = new_total

    # --------------------------------------------------------
    # SAVE TO CSV
    # --------------------------------------------------------

    try:

        leave_df.to_csv(
            LEAVE_FILE,
            index=False,
        )

    except Exception as exc:

        print(
            f"[Leave CSV Update Error] {exc}"
        )

        # Restore in-memory dataframe from CSV
        try:

            import pandas as pd

            employee_tools_module.leave_df = (
                pd.read_csv(
                    LEAVE_FILE,
                    keep_default_na=False,
                )
            )

            employee_tools_module.leave_df[
                "employee_id"
            ] = (
                employee_tools_module.leave_df[
                    "employee_id"
                ]
                .astype(str)
                .str.strip()
                .str.upper()
            )

        except Exception:
            pass

        return (
            "- I couldn't save the leave request. "
            "Your leave balance has not been permanently changed."
        )

    # --------------------------------------------------------
    # SUCCESS
    # --------------------------------------------------------

    # Re-read the updated record so that the response
    # always reflects the persisted CSV value.
    updated_result = get_leave_balance(
        employee_id
    )

    updated_data = unwrap_tool_result(
        updated_result,
        {},
    )

    updated_remaining = new_balance

    updated_total = None

    if isinstance(updated_data, dict):

        updated_remaining = safe_int(
            updated_data.get(
                column,
                new_balance,
            ),
            new_balance,
        )

        for total_key in [
            "total_leave",
            "total_leaves",
            "total",
        ]:

            if total_key in updated_data:

                updated_total = safe_int(
                    updated_data.get(
                        total_key
                    ),
                    0,
                )

                break

    response_lines = [
        "**Leave Request Approved**",
        "",
        f"- **Leave Type:** {leave_type_display(leave_type)}",
        f"- **Days Taken:** {days} {day_word(days)}",
        f"- **Remaining {leave_type_display(leave_type)}:** "
        f"{updated_remaining} {day_word(updated_remaining)}",
    ]

    if updated_total is not None:

        response_lines.extend(
            [
                "",
                "**Updated Leave Balance**",
                "",
            ]
        )

        for balance_type, balance_keys, label in [
            (
                "casual_leave",
                [
                    "casual_leave",
                    "casual",
                    "cl",
                    "CL",
                ],
                "Casual Leave (CL)",
            ),
            (
                "earned_leave",
                [
                    "earned_leave",
                    "earned",
                    "el",
                    "EL",
                ],
                "Earned Leave (EL)",
            ),
            (
                "sick_leave",
                [
                    "sick_leave",
                    "sick",
                    "sl",
                    "SL",
                ],
                "Sick Leave (SL)",
            ),
        ]:

            balance_value = first_value(
                updated_data,
                *balance_keys,
                default=0,
            )

            response_lines.append(
                f"- **{label}:** "
                f"{clean_number(balance_value)}"
            )

        response_lines.append(
            f"- **Total Leave:** "
            f"{clean_number(updated_total)}"
        )

    return "\n".join(response_lines)


# ============================================================
# LEAVE REQUEST FLOW
# ============================================================

def answer_leave_request(
    question: str,
    employee_id: str,
    pending_days: Optional[int] = None,
) -> tuple[str, Optional[int], Optional[str]]:

    text = normalize_text(
        question
    )

    # --------------------------------------------------------
    # ALWAYS TRY TO EXTRACT A NEW NUMBER FIRST
    # --------------------------------------------------------

    days = extract_leave_days(
        text
    )

    # --------------------------------------------------------
    # If no new number, use pending number
    # --------------------------------------------------------

    if days is None:
        days = pending_days

    # --------------------------------------------------------
    # If no number available
    # --------------------------------------------------------

    if days is None:

        return (
            "How many days of leave would you like to take?",
            None,
            None,
        )

    # --------------------------------------------------------
    # Extract leave type
    # --------------------------------------------------------

    leave_type = extract_leave_type(
        text
    )

    # --------------------------------------------------------
    # No leave type yet
    # --------------------------------------------------------

    if leave_type is None:

        return (
            leave_type_question(days),
            days,
            None,
        )

    # --------------------------------------------------------
    # Both days + type available
    # --------------------------------------------------------

    response = apply_leave(
        employee_id,
        leave_type,
        days,
    )

    return (
        response,
        None,
        None,
    )


# ============================================================
# EXPENSE DATA
# ============================================================

def get_expenses(
    employee_id: str,
) -> List[Dict[str, Any]]:

    result = get_expense_records(
        employee_id
    )

    if not tool_success(result):
        return []

    data = unwrap_tool_result(
        result,
        [],
    )

    if isinstance(data, list):

        return [
            item
            for item in data
            if isinstance(item, dict)
        ]

    return []


def format_expenses(
    records: List[Dict[str, Any]],
    heading: str,
) -> str:

    if not records:

        return (
            f"**{heading}**\n\n"
            "- No matching expense records found."
        )

    lines = [
        f"**{heading}**",
        "",
    ]

    for record in records:

        expense_id = first_value(
            record,
            "expense_id",
            "expenseId",
            "id",
            default="N/A",
        )

        category = first_value(
            record,
            "category",
            "expense_type",
            "type",
            default="Expense",
        )

        amount = first_value(
            record,
            "amount",
            "expense_amount",
            default=0,
        )

        status = first_value(
            record,
            "status",
            "expense_status",
            default="Unknown",
        )

        date = first_value(
            record,
            "date",
            "expense_date",
            "submitted_date",
            default="N/A",
        )

        lines.append(
            f"- **{expense_id}** | "
            f"{category} | "
            f"₹{clean_number(amount)} | "
            f"{status} | "
            f"{date}"
        )

    return "\n".join(lines)


def answer_expenses(
    employee_id: str,
) -> str:

    records = get_expenses(
        employee_id
    )

    return format_expenses(
        records,
        "Expense Records",
    )


def answer_expense_status(
    employee_id: str,
    status: str,
) -> str:

    records = get_expenses(
        employee_id
    )

    filtered = []

    for record in records:

        record_status = str(
            first_value(
                record,
                "status",
                "expense_status",
                default="",
            )
        ).strip().lower()

        if record_status == status.lower():

            filtered.append(record)

    return format_expenses(
        filtered,
        f"{status.capitalize()} Expenses",
    )


# ============================================================
# TOTAL EXPENSE
# ============================================================

def answer_total_expenses(
    employee_id: str,
) -> str:

    result = get_total_expenses(
        employee_id
    )

    if not tool_success(result):
        return (
            "- Total expenses could not be retrieved."
        )

    data = unwrap_tool_result(
        result,
        {},
    )

    if isinstance(data, dict):

        total = first_value(
            data,
            "total_expenses",
            "total_expense",
            "total",
            default=0,
        )

    else:

        total = data

    return (
        "**Total Expenses**\n\n"
        f"- **Total:** ₹{clean_number(total)}"
    )


# ============================================================
# EXPENSE SUMMARY
# ============================================================

def answer_expense_summary(
    employee_id: str,
) -> str:

    result = get_expense_summary(
        employee_id
    )

    if not tool_success(result):
        return (
            "- Expense summary could not be retrieved."
        )

    data = unwrap_tool_result(
        result,
        {},
    )

    if not isinstance(data, dict):
        return (
            "- Expense summary is not available."
        )

    total = first_value(
        data,
        "total_expenses",
        "total_expense",
        "total",
        default=0,
    )

    approved = first_value(
        data,
        "approved",
        "approved_expenses",
        default=0,
    )

    pending = first_value(
        data,
        "pending",
        "pending_expenses",
        default=0,
    )

    rejected = first_value(
        data,
        "rejected",
        "rejected_expenses",
        default=0,
    )

    return "\n".join(
        [
            "**Expense Summary**",
            "",
            f"- **Total:** ₹{clean_number(total)}",
            f"- **Approved:** ₹{clean_number(approved)}",
            f"- **Pending:** ₹{clean_number(pending)}",
            f"- **Rejected:** ₹{clean_number(rejected)}",
        ]
    )


# ============================================================
# IT ASSETS
# ============================================================

def get_asset_records(
    employee_id: str,
) -> List[Dict[str, Any]]:

    result = get_it_assets(
        employee_id
    )

    if not tool_success(result):
        return []

    data = unwrap_tool_result(
        result,
        [],
    )

    if isinstance(data, list):

        return [
            item
            for item in data
            if isinstance(item, dict)
        ]

    return []


def answer_it_assets(
    employee_id: str,
    question: str,
) -> str:

    records = get_asset_records(
        employee_id
    )

    if not records:

        return (
            "**IT Assets**\n\n"
            "- No assigned IT assets found."
        )

    text = normalize_text(
        question
    )

    laptop_records = []

    for record in records:

        asset_type = str(
            first_value(
                record,
                "asset_type",
                "type",
                "category",
                default="",
            )
        ).lower()

        asset_name = str(
            first_value(
                record,
                "asset_name",
                "name",
                "device_name",
                "model",
                default="",
            )
        ).lower()

        combined = (
            f"{asset_type} {asset_name}"
        )

        if "laptop" in combined:
            laptop_records.append(record)

    # --------------------------------------------------------
    # Specific laptop request
    # --------------------------------------------------------

    if (
        "laptop" in text
        or "computer" in text
        or "device" in text
    ) and laptop_records:

        record = laptop_records[0]

        asset_id = first_value(
            record,
            "asset_id",
            "assetId",
            "id",
            default="N/A",
        )

        asset_name = first_value(
            record,
            "asset_name",
            "name",
            "device_name",
            "model",
            default="N/A",
        )

        asset_type = first_value(
            record,
            "asset_type",
            "type",
            "category",
            default="Laptop",
        )

        serial = first_value(
            record,
            "serial_number",
            "serial",
            default="N/A",
        )

        status = first_value(
            record,
            "status",
            "asset_status",
            default="N/A",
        )

        return "\n".join(
            [
                "**Assigned Laptop**",
                "",
                f"- **Asset ID:** {asset_id}",
                f"- **Device:** {asset_name}",
                f"- **Type:** {asset_type}",
                f"- **Serial Number:** {serial}",
                f"- **Status:** {status}",
            ]
        )

    # --------------------------------------------------------
    # All assets
    # --------------------------------------------------------

    lines = [
        "**IT Assets**",
        "",
    ]

    for record in records:

        asset_id = first_value(
            record,
            "asset_id",
            "assetId",
            "id",
            default="N/A",
        )

        asset_name = first_value(
            record,
            "asset_name",
            "name",
            "device_name",
            "model",
            default="N/A",
        )

        asset_type = first_value(
            record,
            "asset_type",
            "type",
            "category",
            default="N/A",
        )

        serial = first_value(
            record,
            "serial_number",
            "serial",
            default="N/A",
        )

        status = first_value(
            record,
            "status",
            "asset_status",
            default="N/A",
        )

        lines.extend(
            [
                f"- **Asset ID:** {asset_id}",
                f"- **Device:** {asset_name}",
                f"- **Type:** {asset_type}",
                f"- **Serial Number:** {serial}",
                f"- **Status:** {status}",
                "",
            ]
        )

    return "\n".join(lines).strip()


# ============================================================
# EMPLOYEE DETAILS
# ============================================================

def answer_employee_details(
    employee_id: str,
) -> str:

    result = get_employee_details(
        employee_id
    )

    if not tool_success(result):

        return (
            "- Employee details could not be retrieved."
        )

    data = unwrap_tool_result(
        result,
        {},
    )

    if not isinstance(data, dict):

        return (
            "- Employee details are not available."
        )

    return "\n".join(
        [
            "**Employee Details**",
            "",
            f"- **Employee ID:** "
            f"{first_value(data, 'employee_id', 'employeeId', 'id', default=employee_id)}",

            f"- **Name:** "
            f"{first_value(data, 'name', 'employee_name')}",

            f"- **Department:** "
            f"{first_value(data, 'department')}",

            f"- **Designation:** "
            f"{first_value(data, 'designation', 'job_title', 'role')}",

            f"- **Manager ID:** "
            f"{first_value(data, 'manager_id', 'managerId')}",

            f"- **Manager Name:** "
            f"{first_value(data, 'manager_name', 'managerName')}",

            f"- **Manager Designation:** "
            f"{first_value(data, 'manager_designation', 'managerDesignation')}",

            f"- **Joining Date:** "
            f"{first_value(data, 'joining_date', 'date_of_joining', 'doj')}",

            f"- **Location:** "
            f"{first_value(data, 'location_id', 'location', 'office_location')}",

            f"- **Employment Status:** "
            f"{first_value(data, 'employment_status', 'status')}",
        ]
    )


# ============================================================
# MANAGER
# ============================================================

def answer_manager(
    employee_id: str,
) -> str:

    result = get_employee_details(
        employee_id
    )

    if not tool_success(result):

        return (
            "- Manager details could not be retrieved."
        )

    data = unwrap_tool_result(
        result,
        {},
    )

    if not isinstance(data, dict):

        return (
            "- Manager details are not available."
        )

    return "\n".join(
        [
            "**Manager Details**",
            "",
            f"- **Manager ID:** "
            f"{first_value(data, 'manager_id', 'managerId')}",

            f"- **Name:** "
            f"{first_value(data, 'manager_name', 'managerName')}",

            f"- **Designation:** "
            f"{first_value(data, 'manager_designation', 'managerDesignation')}",
        ]
    )


# ============================================================
# OFFICE
# ============================================================

def extract_city(
    question: str,
) -> str:

    text = normalize_text(
        question
    )

    cities = [
        "bangalore",
        "bengaluru",
        "chennai",
        "mumbai",
        "delhi",
        "hyderabad",
        "pune",
        "kochi",
        "kolkata",
        "noida",
        "gurgaon",
        "gurugram",
    ]

    for city in cities:

        if city in text:
            return city

    return "bangalore"


def answer_office(
    question: str,
) -> str:

    city = extract_city(
        question
    )

    result = get_office_details(
        city
    )

    if not tool_success(result):

        return (
            "- Office information could not be retrieved."
        )

    data = unwrap_tool_result(
        result,
        {},
    )

    if not isinstance(data, dict):

        return (
            "- Office information is not available."
        )

    lines = [
        f"**Office Information - {city.title()}**",
        "",
    ]

    for key, value in data.items():

        if value is None:
            continue

        label = str(key).replace(
            "_",
            " ",
        ).title()

        lines.append(
            f"- **{label}:** {value}"
        )

    return "\n".join(lines)


# ============================================================
# POLICY RAG
# ============================================================

POLICY_QUERIES = {

    "leave_policy":
        "leave policy leave entitlement leave rules carry forward unused leave",

    "wfh_policy":
        "work from home WFH remote work policy consecutive days approval",

    "attendance_policy":
        "attendance policy attendance rules late arrival working hours office attendance",

    "security_policy":
        "IT security information security password MFA account security policy",

    "holiday_policy":
        "company holiday policy holiday list holidays",
}


def extract_policy_context(
    result: Any,
) -> str:

    if result is None:
        return ""

    if isinstance(result, str):
        return result.strip()

    if isinstance(result, dict):

        for key in [
            "context",
            "retrieved_context",
            "text",
            "content",
            "answer",
        ]:

            value = result.get(key)

            if value:

                if isinstance(value, list):

                    return "\n".join(
                        str(x)
                        for x in value
                    )

                return str(value).strip()

        data = result.get(
            "data"
        )

        if data:
            return extract_policy_context(
                data
            )

    if isinstance(result, list):

        chunks = []

        for item in result:

            if isinstance(item, str):

                chunks.append(item)

            elif isinstance(item, dict):

                value = (
                    item.get("text")
                    or item.get("content")
                    or item.get("page_content")
                )

                if value:
                    chunks.append(
                        str(value)
                    )

        return "\n".join(chunks)

    return str(result).strip()


def policy_to_bullets(
    text: str,
) -> str:

    if not text:

        return (
            "- No relevant policy information found."
        )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        str(text).strip(),
    )

    bullets = []

    for line in text.splitlines():

        line = line.strip()

        if not line:
            continue

        if line.startswith("#"):
            continue

        line = re.sub(
            r"^[\-\*•]\s*",
            "",
            line,
        )

        sentences = re.split(
            r"(?<=[.!?])\s+",
            line,
        )

        for sentence in sentences:

            sentence = sentence.strip()

            if sentence:
                bullets.append(
                    f"- {sentence}"
                )

    bullets = bullets[:8]

    if not bullets:

        return (
            "- No relevant policy information found."
        )

    return "\n".join(bullets)


def answer_policy(
    intent: str,
    question: str,
) -> str:

    if retrieve_policy is None:

        return (
            "- Policy retrieval is not available."
        )

    query = POLICY_QUERIES.get(
        intent,
        question,
    )

    try:

        result = retrieve_policy(
            query
        )

    except Exception as exc:

        print(
            f"[Policy Retrieval Error] {exc}"
        )

        return (
            "- I couldn't retrieve the relevant company policy right now."
        )

    context = extract_policy_context(
        result
    )

    if not context:

        return (
            "- No relevant policy information found."
        )

    return policy_to_bullets(
        context
    )


# ============================================================
# INTENT DETECTION
# ============================================================

def detect_intents(
    text: str,
) -> List[str]:

    text = normalize_text(
        text
    )

    intents: List[str] = []

    # --------------------------------------------------------
    # EMPLOYEE DETAILS
    # --------------------------------------------------------

    if any(
        re.search(pattern, text)
        for pattern in [

            r"\bemployee details?\b",
            r"\bmy employee details?\b",
            r"\bemployee information\b",
            r"\bmy employee information\b",
            r"\bemployee info\b",
            r"\bmy employee info\b",
            r"\bemployee profile\b",
            r"\bmy profile\b",
        ]
    ):

        intents.append(
            "employee_details"
        )

    # --------------------------------------------------------
    # MANAGER
    # --------------------------------------------------------

    if any(
        re.search(pattern, text)
        for pattern in [

            r"\bmanager details?\b",
            r"\bmy manager\b",
            r"\bmanager information\b",
            r"\bmanager info\b",
            r"\bwho is my manager\b",
            r"\bmanager name\b",
            r"\bmanager id\b",
            r"\breporting manager\b",
        ]
    ):

        intents.append(
            "manager_details"
        )

    # --------------------------------------------------------
    # LEAVE POLICY
    # --------------------------------------------------------

    if any(
        re.search(pattern, text)
        for pattern in [

            r"\bleave policy\b",
            r"\bleave policies\b",
            r"\bleave rules\b",
            r"\bcarry forward.*leave\b",
            r"\bleave.*carry forward\b",
            r"\bcarry-forward.*leave\b",
            r"\bcarryforward.*leave\b",
            r"\bcan i carry forward\b",
            r"\bunused leave.*carry\b",
            r"\bhow many.*leave.*carry\b",
        ]
    ):

        intents.append(
            "leave_policy"
        )

    # --------------------------------------------------------
    # SPECIFIC LEAVE BALANCE
    # --------------------------------------------------------

    if any(
        re.search(pattern, text)
        for pattern in [

            r"\bcasual leave\b",
            r"\bearned leave\b",
            r"\bsick leave\b",
            r"\bcl balance\b",
            r"\bel balance\b",
            r"\bsl balance\b",
        ]
    ) and not is_leave_request(text):

        intents.append(
            "specific_leave"
        )

    # --------------------------------------------------------
    # LEAVE BALANCE
    # --------------------------------------------------------

    if any(
        re.search(pattern, text)
        for pattern in [

            r"\bleave balance\b",
            r"\bleave balances\b",
            r"\bleave details?\b",
            r"\bleave record\b",
            r"\bleave records\b",
            r"\bhow much leave\b",
            r"\bhow many leaves\b",
            r"\bremaining leave\b",
            r"\bremaining leaves\b",
            r"\bavailable leave\b",
            r"\bavailable leaves\b",
        ]
    ):

        intents.append(
            "leave_balance"
        )

    # --------------------------------------------------------
    # EXPENSE STATUS
    # --------------------------------------------------------

    if any(
        re.search(pattern, text)
        for pattern in [

            r"\bpending expense\b",
            r"\bpending expenses\b",
            r"\bexpense.*pending\b",
            r"\bpending report\b",
            r"\bpending expense report\b",
            r"\bpending reimbursement\b",
            r"\breimbursement.*pending\b",
        ]
    ):

        intents.append(
            "pending_expenses"
        )

    if any(
        re.search(pattern, text)
        for pattern in [

            r"\bapproved expense\b",
            r"\bapproved expenses\b",
            r"\bexpense.*approved\b",
            r"\bapproved report\b",
            r"\bapproved expense report\b",
            r"\bapproved reimbursement\b",
        ]
    ):

        intents.append(
            "approved_expenses"
        )

    if any(
        re.search(pattern, text)
        for pattern in [

            r"\brejected expense\b",
            r"\brejected expenses\b",
            r"\bexpense.*rejected\b",
            r"\brejected report\b",
            r"\brejected expense report\b",
            r"\brejected reimbursement\b",
        ]
    ):

        intents.append(
            "rejected_expenses"
        )

    # --------------------------------------------------------
    # EXPENSE RECORDS
    # --------------------------------------------------------

    if any(
        re.search(pattern, text)
        for pattern in [

            r"\bexpense report\b",
            r"\bexpense reports\b",
            r"\bexpense records?\b",
            r"\bmy expenses?\b",
            r"\bshow.*expenses?\b",
            r"\blist.*expenses?\b",
            r"\bexpense details?\b",
            r"\breimbursement records?\b",
        ]
    ):

        if not any(
            x in intents
            for x in [
                "pending_expenses",
                "approved_expenses",
                "rejected_expenses",
            ]
        ):

            intents.append(
                "expenses"
            )

    # --------------------------------------------------------
    # TOTAL EXPENSE
    # --------------------------------------------------------

    if any(
        re.search(pattern, text)
        for pattern in [

            r"\btotal expenses?\b",
            r"\btotal expense\b",
            r"\bhow much.*spent\b",
            r"\bhow much.*expense\b",
            r"\btotal reimbursement\b",
        ]
    ):

        intents.append(
            "total_expenses"
        )

    # --------------------------------------------------------
    # EXPENSE SUMMARY
    # --------------------------------------------------------

    if any(
        re.search(pattern, text)
        for pattern in [

            r"\bexpense summary\b",
            r"\bsummary of expenses?\b",
            r"\bexpense breakdown\b",
            r"\bbreakdown of expenses?\b",
        ]
    ):

        intents.append(
            "expense_summary"
        )

    # --------------------------------------------------------
    # IT ASSETS
    # --------------------------------------------------------

    if any(
        re.search(pattern, text)
        for pattern in [

            r"\bit assets?\b",
            r"\bmy assets?\b",
            r"\bassigned assets?\b",
            r"\basset details?\b",
            r"\bmy company laptop\b",
            r"\bmy laptop\b",
            r"\bcompany laptop\b",
            r"\blaptop assigned\b",
            r"\bassigned laptop\b",
            r"\bwhat laptop\b",
            r"\bwhich laptop\b",
            r"\blaptop details?\b",
            r"\blaptop serial\b",
            r"\blaptop serial number\b",
            r"\bcomputer assigned\b",
            r"\bcomputer details?\b",
            r"\bdevice assigned\b",
            r"\bwhat device\b",
            r"\bwhich device\b",
        ]
    ):

        intents.append(
            "it_assets"
        )

    # --------------------------------------------------------
    # OFFICE
    # --------------------------------------------------------

    if any(
        re.search(pattern, text)
        for pattern in [

            r"\boffice location\b",
            r"\boffice details?\b",
            r"\boffice information\b",
            r"\bnearest office\b",
            r"\bwhere is the office\b",
            r"\boffice working hours\b",
            r"\bworking hours\b",
        ]
    ):

        intents.append(
            "office_info"
        )

    # --------------------------------------------------------
    # WFH POLICY
    # --------------------------------------------------------

    if any(
        re.search(pattern, text)
        for pattern in [

            r"\bwfh policy\b",
            r"\bwfh\b",
            r"\bwork from home policy\b",
            r"\bwork from home rules\b",
            r"\bremote work policy\b",
            r"\bcan i work from home\b",
            r"\bhow many.*wfh\b",
            r"\bhow many.*work from home\b",
        ]
    ):

        intents.append(
            "wfh_policy"
        )

    # --------------------------------------------------------
    # ATTENDANCE POLICY
    # --------------------------------------------------------

    if any(
        re.search(pattern, text)
        for pattern in [

            r"\battendance policy\b",
            r"\battendance rules\b",
            r"\battendance timing\b",
            r"\battendance.*policy\b",
            r"\bworking hours policy\b",
        ]
    ):

        intents.append(
            "attendance_policy"
        )

    # --------------------------------------------------------
    # SECURITY POLICY
    # --------------------------------------------------------

    if any(
        re.search(pattern, text)
        for pattern in [

            r"\bsecurity policy\b",
            r"\binformation security\b",
            r"\bsecurity rules\b",
            r"\bdata security\b",
            r"\bsecurity guidelines\b",
        ]
    ):

        intents.append(
            "security_policy"
        )

    # --------------------------------------------------------
    # HOLIDAY POLICY
    # --------------------------------------------------------

    if any(
        re.search(pattern, text)
        for pattern in [

            r"\bholiday policy\b",
            r"\bholiday list\b",
            r"\bcompany holidays\b",
        ]
    ):

        intents.append(
            "holiday_policy"
        )

    # --------------------------------------------------------
    # REMOVE DUPLICATES
    # --------------------------------------------------------

    final = []

    for intent in intents:

        if intent not in final:
            final.append(intent)

    return final


# ============================================================
# ANSWER ONE QUESTION
# ============================================================

def answer_one_question(
    question: str,
    employee_id: str,
    pending_leave_days: Optional[int] = None,
) -> tuple[str, Optional[int]]:

    text = normalize_text(
        question
    )

    # --------------------------------------------------------
    # SAFETY
    # --------------------------------------------------------

    if is_self_harm(text):

        return (
            self_harm_response(),
            pending_leave_days,
        )

    # --------------------------------------------------------
    # CONFIDENTIAL
    # --------------------------------------------------------

    if is_confidential_request(text):

        return (
            confidential_response(),
            pending_leave_days,
        )

    # --------------------------------------------------------
    # PASSWORD
    # --------------------------------------------------------

    if is_password_help(text):

        return (
            password_response(),
            pending_leave_days,
        )

    # --------------------------------------------------------
    # GREETING
    # --------------------------------------------------------

    if is_greeting(text):

        return (
            greeting_response(),
            pending_leave_days,
        )

    # ========================================================
    # LEAVE HANDLING
    # ========================================================

    # IMPORTANT:
    # Always extract the NEW number before using any
    # previously pending number.
    #
    # Example:
    #
    # Previous pending = 2
    #
    # User:
    # "I want to take 10 casual leave"
    #
    # extracted_days = 10
    #
    # Therefore 10 MUST be used.
    # ========================================================

    extracted_days = extract_leave_days(
        text
    )

    extracted_leave_type = extract_leave_type(
        text
    )

    # --------------------------------------------------------
    # CASE 1:
    # NEW number + leave type
    #
    # Example:
    # "I want to take 10 casual leave"
    #
    # Immediately apply 10.
    # --------------------------------------------------------

    if (
        extracted_days is not None
        and extracted_leave_type is not None
    ):

        response = apply_leave(
            employee_id,
            extracted_leave_type,
            extracted_days,
        )

        return (
            response,
            None,
        )

    # --------------------------------------------------------
    # CASE 2:
    # NEW number but NO leave type
    #
    # Example:
    # "I need 10 days leave"
    #
    # Store 10 as pending.
    # --------------------------------------------------------

    if (
        extracted_days is not None
        and extracted_leave_type is None
    ):

        return (
            leave_type_question(
                extracted_days
            ),
            extracted_days,
        )

    # --------------------------------------------------------
    # CASE 3:
    # No NEW number + leave type + pending number
    #
    # Example:
    #
    # Previous:
    # "I need 2 days leave"
    #
    # pending = 2
    #
    # User:
    # "casual leave"
    #
    # Apply pending 2.
    # --------------------------------------------------------

    if (
        extracted_days is None
        and pending_leave_days is not None
        and extracted_leave_type is not None
    ):

        response = apply_leave(
            employee_id,
            extracted_leave_type,
            pending_leave_days,
        )

        return (
            response,
            None,
        )

    # --------------------------------------------------------
    # CASE 4:
    # Leave request without number
    # --------------------------------------------------------

    if is_leave_request(text):

        response, new_pending_days, _ = (
            answer_leave_request(
                text,
                employee_id,
                pending_leave_days,
            )
        )

        return (
            response,
            new_pending_days,
        )

    # ========================================================
    # NORMAL INTENTS
    # ========================================================

    intents = detect_intents(
        text
    )

    if not intents:

        return (
            "- I can help with Blue Orbit workplace information, "
            "but this question is outside my scope.",
            pending_leave_days,
        )

    answers = []

    for intent in intents:

        if intent == "employee_details":

            answers.append(
                answer_employee_details(
                    employee_id
                )
            )

        elif intent == "manager_details":

            answers.append(
                answer_manager(
                    employee_id
                )
            )

        elif intent == "leave_policy":

            answers.append(
                answer_policy(
                    "leave_policy",
                    text,
                )
            )

        elif intent == "specific_leave":

            answers.append(
                answer_specific_leave(
                    employee_id,
                    text,
                )
            )

        elif intent == "leave_balance":

            answers.append(
                answer_leave_balance(
                    employee_id
                )
            )

        elif intent == "pending_expenses":

            answers.append(
                answer_expense_status(
                    employee_id,
                    "pending",
                )
            )

        elif intent == "approved_expenses":

            answers.append(
                answer_expense_status(
                    employee_id,
                    "approved",
                )
            )

        elif intent == "rejected_expenses":

            answers.append(
                answer_expense_status(
                    employee_id,
                    "rejected",
                )
            )

        elif intent == "expenses":

            answers.append(
                answer_expenses(
                    employee_id
                )
            )

        elif intent == "total_expenses":

            answers.append(
                answer_total_expenses(
                    employee_id
                )
            )

        elif intent == "expense_summary":

            answers.append(
                answer_expense_summary(
                    employee_id
                )
            )

        elif intent == "it_assets":

            answers.append(
                answer_it_assets(
                    employee_id,
                    text,
                )
            )

        elif intent == "office_info":

            answers.append(
                answer_office(
                    text
                )
            )

        elif intent == "wfh_policy":

            answers.append(
                answer_policy(
                    "wfh_policy",
                    text,
                )
            )

        elif intent == "attendance_policy":

            answers.append(
                answer_policy(
                    "attendance_policy",
                    text,
                )
            )

        elif intent == "security_policy":

            answers.append(
                answer_policy(
                    "security_policy",
                    text,
                )
            )

        elif intent == "holiday_policy":

            answers.append(
                answer_policy(
                    "holiday_policy",
                    text,
                )
            )

    # --------------------------------------------------------
    # REMOVE EMPTY / DUPLICATE ANSWERS
    # --------------------------------------------------------

    final_answers = []

    for answer in answers:

        if (
            answer
            and answer not in final_answers
        ):

            final_answers.append(
                answer
            )

    if not final_answers:

        return (
            "- I couldn't find an answer for that request.",
            pending_leave_days,
        )

    return (
        "\n\n".join(
            final_answers
        ),
        pending_leave_days,
    )


# ============================================================
# MIXED QUESTION HANDLING
# ============================================================

def process_full_question(
    question: str,
    employee_id: str,
    pending_leave_days: Optional[int] = None,
) -> tuple[str, Optional[int]]:

    questions = split_questions(
        question
    )

    if not questions:

        questions = [
            question
        ]

    answers = []

    current_pending_days = (
        pending_leave_days
    )

    for individual_question in questions:

        answer, current_pending_days = (
            answer_one_question(
                individual_question,
                employee_id,
                current_pending_days,
            )
        )

        if (
            answer
            and answer not in answers
        ):

            answers.append(
                answer
            )

    if not answers:

        return (
            "- I couldn't process that request.",
            current_pending_days,
        )

    return (
        "\n\n---\n\n".join(
            answers
        ),
        current_pending_days,
    )


# ============================================================
# LANGGRAPH NODE
# ============================================================

def agent_node(
    state: AgentState,
) -> AgentState:

    question = state.get(
        "question",
        "",
    )

    employee_id = state.get(
        "employee_id",
        DEFAULT_EMPLOYEE_ID,
    )

    pending_leave_days = state.get(
        "pending_leave_days"
    )

    final_answer, new_pending_days = (
        process_full_question(
            question,
            employee_id,
            pending_leave_days,
        )
    )

    new_state = {
        **state,
        "final_answer": final_answer,
    }

    # --------------------------------------------------------
    # Save conversational leave state
    # --------------------------------------------------------

    if new_pending_days is not None:

        new_state[
            "pending_leave_days"
        ] = new_pending_days

    else:

        new_state.pop(
            "pending_leave_days",
            None,
        )

    return new_state


# ============================================================
# LANGGRAPH
# ============================================================

graph = StateGraph(
    AgentState
)

graph.add_node(
    "agent",
    agent_node,
)

graph.add_edge(
    START,
    "agent",
)

graph.add_edge(
    "agent",
    END,
)

checkpointer = MemorySaver()

app = graph.compile(
    checkpointer=checkpointer
)


# ============================================================
# PUBLIC FUNCTION
# ============================================================

def ask_employee(
    question: str,
    employee_id: str = DEFAULT_EMPLOYEE_ID,
    thread_id: str = "default",
) -> str:

    if not question or not str(question).strip():

        return "Please enter a question."

    if not employee_id:

        employee_id = DEFAULT_EMPLOYEE_ID

    employee_id = str(
        employee_id
    ).strip().upper()

    # Make sure the thread ID is valid and stable.
    thread_id = str(
        thread_id or "default"
    ).strip()

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    try:

        result = app.invoke(
            {
                "question": question,
                "employee_id": employee_id,
            },
            config=config,
        )

        return result.get(
            "final_answer",
            "I couldn't process that request.",
        )

    except Exception as exc:

        print(
            f"[Blue Orbit Agent Error] {exc}"
        )

        return (
            "Sorry, I couldn't process that request right now. "
            "Please try again."
        )


# ============================================================
# DIRECT TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("BLUE ORBIT OFFICE ASSISTANT")
    print("=" * 70)

    print(
        f"Employee: {DEFAULT_EMPLOYEE_ID}"
    )

    print(
        f"Leave file: {LEAVE_FILE}"
    )

    print(
        "Type 'exit' to quit."
    )

    print()

    thread_id = "leave-test"

    while True:

        question = input(
            "You: "
        ).strip()

        if question.lower() in {
            "exit",
            "quit",
        }:

            print(
                "Goodbye! 👋"
            )

            break

        answer = ask_employee(
            question,
            DEFAULT_EMPLOYEE_ID,
            thread_id,
        )

        print(
            "\nAssistant:"
        )

        print(
            answer
        )

        print()

