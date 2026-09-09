from pathlib import Path
from typing import Any, Dict

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "csvs"

# Expected project structure:
#
# BlueOrbit-Office-Assistant/
# ├── data/
# │   └── csvs/
# │       ├── employees_with_null.csv
# │       ├── leave_balance.csv
# │       ├── expense_records.csv
# │       ├── IT_assets.csv
# │       └── office_locations.csv


# ============================================================
# HELPERS
# ============================================================

def _normalize_id(value: str) -> str:
    """Normalize an employee/location identifier."""
    return str(value).strip().upper()


def _result(
    success: bool,
    data: Any = None,
    message: str = ""
) -> Dict[str, Any]:
    """Return a consistent result structure for every tool."""
    return {
        "success": success,
        "data": data,
        "message": message,
    }


def _employee_rows(employee_id: str) -> pd.DataFrame:
    """Return employee rows matching the supplied employee ID."""
    normalized_id = _normalize_id(employee_id)

    return employees_df[
        employees_df["employee_id"] == normalized_id
    ]


# ============================================================
# DATA LOADING
# ============================================================

employees_df = pd.read_csv(
    DATA_DIR / "employees_with_null.csv",
    keep_default_na=False
)

leave_df = pd.read_csv(
    DATA_DIR / "leave_balance.csv",
    keep_default_na=False
)

expenses_df = pd.read_csv(
    DATA_DIR / "expense_records.csv",
    keep_default_na=False
)

assets_df = pd.read_csv(
    DATA_DIR / "IT_assets.csv",
    keep_default_na=False
)

offices_df = pd.read_csv(
    DATA_DIR / "office_locations.csv",
    keep_default_na=False
)


# ============================================================
# NORMALIZE DATA
# ============================================================

# Employee IDs
for df in (
    employees_df,
    leave_df,
    expenses_df,
    assets_df
):
    if "employee_id" in df.columns:
        df["employee_id"] = (
            df["employee_id"]
            .astype(str)
            .str.strip()
            .str.upper()
        )


# Manager IDs
if "manager_id" in employees_df.columns:
    employees_df["manager_id"] = (
        employees_df["manager_id"]
        .astype(str)
        .str.strip()
        .str.upper()
        .replace({
            "": "null",
            "NAN": "null",
            "NONE": "null"
        })
    )


# Office fields
offices_df["location_id"] = (
    offices_df["location_id"]
    .astype(str)
    .str.strip()
    .str.upper()
)

offices_df["city"] = (
    offices_df["city"]
    .astype(str)
    .str.strip()
    .str.lower()
)


# ============================================================
# EMPLOYEE DETAILS
# ============================================================

def get_employee_details(employee_id: str) -> Dict[str, Any]:
    """
    Get employee details.

    Also looks up the employee's manager using manager_id
    and adds the manager's name and designation.
    """

    rows = _employee_rows(employee_id)

    if rows.empty:
        return _result(
            False,
            None,
            f"Employee {_normalize_id(employee_id)} not found."
        )

    # Get employee record
    employee = rows.iloc[0].to_dict()

    # --------------------------------------------------------
    # MANAGER LOOKUP
    # --------------------------------------------------------
    #
    # Example:
    #
    # EMP054 -> manager_id = EMP009
    #
    # Then search employee master:
    #
    # EMP009 -> Tanvi Kapoor
    #
    # This gives us the actual manager name.
    # --------------------------------------------------------

    manager_id = employee.get("manager_id", "")

    if (
        manager_id
        and str(manager_id).strip().lower() not in {
            "null",
            "nan",
            "none",
            ""
        }
    ):
        manager_rows = _employee_rows(manager_id)

        if not manager_rows.empty:
            manager = manager_rows.iloc[0].to_dict()

            employee["manager_name"] = manager.get(
                "name",
                ""
            )

            employee["manager_designation"] = manager.get(
                "designation",
                ""
            )

        else:
            # Manager ID exists but manager record
            # could not be found.
            employee["manager_name"] = ""
            employee["manager_designation"] = ""

    else:
        # Top-level employees have no manager.
        employee["manager_name"] = ""
        employee["manager_designation"] = ""

    return _result(
        True,
        employee,
        "Employee found."
    )


# ============================================================
# LEAVE BALANCE
# ============================================================

def get_leave_balance(employee_id: str) -> Dict[str, Any]:
    """Get the current leave balance for one employee."""

    normalized_id = _normalize_id(employee_id)

    rows = leave_df[
        leave_df["employee_id"] == normalized_id
    ]

    if rows.empty:
        return _result(
            False,
            None,
            f"Leave record for {normalized_id} not found."
        )

    return _result(
        True,
        rows.iloc[0].to_dict(),
        "Leave balance found."
    )
# ============================================================
# APPLY / GRANT LEAVE
# ============================================================

def apply_leave(
    employee_id: str,
    leave_type: str,
    days: int,
) -> Dict[str, Any]:
    """
    Deduct requested leave days from the employee's leave balance.

    leave_type:
        casual_leave
        earned_leave
        sick_leave

    Returns the updated balance.
    """

    normalized_id = _normalize_id(employee_id)

    # --------------------------------------------------------
    # Validate number of days
    # --------------------------------------------------------

    try:
        days = int(days)
    except (TypeError, ValueError):
        return _result(
            False,
            None,
            "Number of leave days must be a valid integer.",
        )

    if days <= 0:
        return _result(
            False,
            None,
            "Leave days must be greater than zero.",
        )

    # --------------------------------------------------------
    # Normalize leave type
    # --------------------------------------------------------

    aliases = {
        "casual": "casual_leave",
        "casual leave": "casual_leave",
        "cl": "casual_leave",
        "casual_leave": "casual_leave",

        "earned": "earned_leave",
        "earned leave": "earned_leave",
        "el": "earned_leave",
        "earned_leave": "earned_leave",

        "sick": "sick_leave",
        "sick leave": "sick_leave",
        "sl": "sick_leave",
        "sick_leave": "sick_leave",
    }

    normalized_leave_type = aliases.get(
        str(leave_type).strip().lower()
    )

    if normalized_leave_type is None:
        return _result(
            False,
            None,
            "Invalid leave type. Use Casual, Earned, or Sick Leave.",
        )

    # --------------------------------------------------------
    # Find employee leave record
    # --------------------------------------------------------

    matching_rows = leave_df.index[
        leave_df["employee_id"] == normalized_id
    ].tolist()

    if not matching_rows:
        return _result(
            False,
            None,
            f"Leave record for {normalized_id} not found.",
        )

    row_index = matching_rows[0]

    # --------------------------------------------------------
    # Find correct column
    # --------------------------------------------------------

    column_aliases = {
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
    }

    balance_column = None

    for column in column_aliases[normalized_leave_type]:
        if column in leave_df.columns:
            balance_column = column
            break

    if balance_column is None:
        return _result(
            False,
            None,
            f"Could not find the {normalized_leave_type} column.",
        )

    # --------------------------------------------------------
    # Current balance
    # --------------------------------------------------------

    try:
        current_balance = float(
            leave_df.at[row_index, balance_column]
        )
    except (TypeError, ValueError):
        return _result(
            False,
            None,
            "Current leave balance is invalid.",
        )

    # --------------------------------------------------------
    # Check sufficient balance
    # --------------------------------------------------------

    if current_balance < days:
        return _result(
            False,
            {
                "employee_id": normalized_id,
                "leave_type": normalized_leave_type,
                "requested_days": days,
                "available_days": current_balance,
            },
            f"Insufficient {normalized_leave_type.replace('_', ' ').title()} balance.",
        )

    # --------------------------------------------------------
    # Deduct leave
    # --------------------------------------------------------

    new_balance = current_balance - days

    leave_df.at[
        row_index,
        balance_column
    ] = new_balance

    # --------------------------------------------------------
    # Update total leave if the column exists
    # --------------------------------------------------------

    total_columns = [
        "total_leave",
        "total_leaves",
        "total",
    ]

    total_column = None

    for column in total_columns:
        if column in leave_df.columns:
            total_column = column
            break

    if total_column:
        try:
            current_total = float(
                leave_df.at[row_index, total_column]
            )

            leave_df.at[
                row_index,
                total_column
            ] = current_total - days

        except (TypeError, ValueError):
            pass

    # --------------------------------------------------------
    # IMPORTANT:
    # Persist the change to CSV
    # --------------------------------------------------------

    leave_file = DATA_DIR / "leave_balance.csv"

    try:
        leave_df.to_csv(
            leave_file,
            index=False,
        )
    except Exception as exc:
        return _result(
            False,
            None,
            f"Leave balance was changed in memory but could not be saved: {exc}",
        )

    # --------------------------------------------------------
    # Return updated record
    # --------------------------------------------------------

    updated_record = leave_df.loc[
        row_index
    ].to_dict()

    return _result(
        True,
        {
            "employee_id": normalized_id,
            "leave_type": normalized_leave_type,
            "requested_days": days,
            "previous_balance": current_balance,
            "remaining_balance": new_balance,
            "updated_record": updated_record,
        },
        "Leave granted successfully.",
    )

# ============================================================
# EXPENSE RECORDS
# ============================================================

def get_expense_records(employee_id: str) -> Dict[str, Any]:
    """Get all expense records for one employee."""

    normalized_id = _normalize_id(employee_id)

    rows = expenses_df[
        expenses_df["employee_id"] == normalized_id
    ]

    if rows.empty:
        return _result(
            False,
            [],
            f"No expense records found for {normalized_id}."
        )

    return _result(
        True,
        rows.to_dict(orient="records"),
        "Expense records found."
    )


# ============================================================
# TOTAL EXPENSES
# ============================================================

def get_total_expenses(employee_id: str) -> Dict[str, Any]:
    """Calculate total expense amount for one employee."""

    normalized_id = _normalize_id(employee_id)

    rows = expenses_df[
        expenses_df["employee_id"] == normalized_id
    ]

    if rows.empty:
        return _result(
            False,
            None,
            f"No expense records found for {normalized_id}."
        )

    amounts = pd.to_numeric(
        rows["amount"],
        errors="coerce"
    ).fillna(0)

    total = float(amounts.sum())

    return _result(
        True,
        {
            "employee_id": normalized_id,
            "total_expenses": total,
        },
        "Total expenses calculated."
    )


# ============================================================
# EXPENSE SUMMARY
# ============================================================

def get_expense_summary(employee_id: str) -> Dict[str, Any]:
    """
    Return total, approved, pending,
    and rejected expense amounts.
    """

    normalized_id = _normalize_id(employee_id)

    rows = expenses_df[
        expenses_df["employee_id"] == normalized_id
    ]

    if rows.empty:
        return _result(
            False,
            None,
            f"No expense records found for {normalized_id}."
        )

    # Convert amounts safely to numeric
    amounts = pd.to_numeric(
        rows["amount"],
        errors="coerce"
    ).fillna(0)

    # Normalize status
    status = (
        rows["status"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    summary = {
        "employee_id": normalized_id,

        "total_expenses": float(
            amounts.sum()
        ),

        "approved": float(
            amounts[status == "approved"].sum()
        ),

        "pending": float(
            amounts[status == "pending"].sum()
        ),

        "rejected": float(
            amounts[status == "rejected"].sum()
        ),
    }

    return _result(
        True,
        summary,
        "Expense summary calculated."
    )


# ============================================================
# IT ASSETS
# ============================================================

def get_it_assets(employee_id: str) -> Dict[str, Any]:
    """Get all IT assets assigned to one employee."""

    normalized_id = _normalize_id(employee_id)

    rows = assets_df[
        assets_df["employee_id"] == normalized_id
    ]

    if rows.empty:
        return _result(
            False,
            [],
            f"No IT assets found for {normalized_id}."
        )

    return _result(
        True,
        rows.to_dict(orient="records"),
        "IT assets found."
    )


# ============================================================
# OFFICE DETAILS
# ============================================================

def get_office_details(city: str) -> Dict[str, Any]:
    """Get office details using a city name."""

    normalized_city = (
        str(city)
        .strip()
        .lower()
    )

    rows = offices_df[
        offices_df["city"] == normalized_city
    ]

    if rows.empty:
        return _result(
            False,
            None,
            f"No BlueOrbit office found in {city}."
        )

    return _result(
        True,
        rows.iloc[0].to_dict(),
        "Office found."
    )