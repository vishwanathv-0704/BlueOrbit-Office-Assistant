from pathlib import Path
from typing import Any, Dict

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "csvs"

# For local testing outside the project structure, place the CSVs in
# data/csvs/ as expected by the project.


# ============================================================
# HELPERS
# ============================================================

def _normalize_id(value: str) -> str:
    """Normalize an identifier for reliable case-insensitive lookup."""
    return str(value).strip().upper()


def _result(success: bool, data: Any = None, message: str = "") -> Dict[str, Any]:
    """Return a consistent result structure for every tool."""
    return {
        "success": success,
        "data": data,
        "message": message,
    }


def _employee_rows(employee_id: str) -> pd.DataFrame:
    """Return employee master rows matching the supplied employee ID."""
    normalized_id = _normalize_id(employee_id)
    return employees_df[employees_df["employee_id"] == normalized_id]


# ============================================================
# DATA LOADING
# ============================================================

employees_df = pd.read_csv(DATA_DIR / "employees_with_null.csv", keep_default_na=False)
leave_df = pd.read_csv(DATA_DIR / "leave_balance.csv", keep_default_na=False)
expenses_df = pd.read_csv(DATA_DIR / "expense_records.csv", keep_default_na=False)
assets_df = pd.read_csv(DATA_DIR / "IT_assets.csv", keep_default_na=False)
offices_df = pd.read_csv(DATA_DIR / "office_locations.csv", keep_default_na=False)

# Normalize lookup fields once instead of repeating string operations in
# every tool call.
for df in (employees_df, leave_df, expenses_df, assets_df):
    df["employee_id"] = df["employee_id"].astype(str).str.strip().str.upper()

offices_df["location_id"] = offices_df["location_id"].astype(str).str.strip().str.upper()
offices_df["city"] = offices_df["city"].astype(str).str.strip().str.lower()

if "manager_id" in employees_df.columns:
    employees_df["manager_id"] = (
        employees_df["manager_id"]
        .astype(str)
        .str.strip()
        .str.upper()
        .replace({"": "null"})
    )


# ============================================================
# EMPLOYEE DETAILS
# ============================================================

def get_employee_details(employee_id: str) -> Dict[str, Any]:
    """Get the employee master record for one employee."""
    rows = _employee_rows(employee_id)

    if rows.empty:
        return _result(False, None, f"Employee {_normalize_id(employee_id)} not found.")

    return _result(True, rows.iloc[0].to_dict(), "Employee found.")


# ============================================================
# LEAVE BALANCE
# ============================================================

def get_leave_balance(employee_id: str) -> Dict[str, Any]:
    """Get the current leave balance for one employee."""
    normalized_id = _normalize_id(employee_id)
    rows = leave_df[leave_df["employee_id"] == normalized_id]

    if rows.empty:
        return _result(False, None, f"Leave record for {normalized_id} not found.")

    return _result(True, rows.iloc[0].to_dict(), "Leave balance found.")


# ============================================================
# EXPENSE RECORDS
# ============================================================

def get_expense_records(employee_id: str) -> Dict[str, Any]:
    """Get all expense records for one employee."""
    normalized_id = _normalize_id(employee_id)
    rows = expenses_df[expenses_df["employee_id"] == normalized_id]

    if rows.empty:
        return _result(False, [], f"No expense records found for {normalized_id}.")

    return _result(True, rows.to_dict(orient="records"), "Expense records found.")


def get_total_expenses(employee_id: str) -> Dict[str, Any]:
    """Calculate total expense amount for one employee."""
    normalized_id = _normalize_id(employee_id)
    rows = expenses_df[expenses_df["employee_id"] == normalized_id]

    if rows.empty:
        return _result(False, None, f"No expense records found for {normalized_id}.")

    amounts = pd.to_numeric(rows["amount"], errors="coerce").fillna(0)

    return _result(
        True,
        {
            "employee_id": normalized_id,
            "total_expenses": float(amounts.sum()),
        },
        "Total expenses calculated.",
    )


def get_expense_summary(employee_id: str) -> Dict[str, Any]:
    """Return total, approved, pending, and rejected expenses."""
    normalized_id = _normalize_id(employee_id)
    rows = expenses_df[expenses_df["employee_id"] == normalized_id]

    if rows.empty:
        return _result(False, None, f"No expense records found for {normalized_id}.")

    amounts = pd.to_numeric(rows["amount"], errors="coerce").fillna(0)
    status = rows["status"].astype(str).str.strip().str.lower()

    summary = {
        "employee_id": normalized_id,
        "total_expenses": float(amounts.sum()),
        "approved": float(amounts[status == "approved"].sum()),
        "pending": float(amounts[status == "pending"].sum()),
        "rejected": float(amounts[status == "rejected"].sum()),
    }

    return _result(True, summary, "Expense summary calculated.")


# ============================================================
# IT ASSETS
# ============================================================

def get_it_assets(employee_id: str) -> Dict[str, Any]:
    """Get all IT assets assigned to one employee."""
    normalized_id = _normalize_id(employee_id)
    rows = assets_df[assets_df["employee_id"] == normalized_id]

    if rows.empty:
        return _result(False, [], f"No IT assets found for {normalized_id}.")

    return _result(True, rows.to_dict(orient="records"), "IT assets found.")


# ============================================================
# OFFICE DETAILS
# ============================================================

def get_office_details(city: str) -> Dict[str, Any]:
    """Get office details using a city name."""
    normalized_city = str(city).strip().lower()
    rows = offices_df[offices_df["city"] == normalized_city]

    if rows.empty:
        return _result(False, None, f"No BlueOrbit office found in {city}.")

    return _result(True, rows.iloc[0].to_dict(), "Office found.")
