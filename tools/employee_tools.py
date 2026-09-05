import pandas as pd
from pathlib import Path


# -----------------------------
# DATA PATHS
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "csvs"


# -----------------------------
# LOAD DATA
# -----------------------------

employees_df = pd.read_csv(DATA_DIR / "employees_with_null.csv")
leave_df = pd.read_csv(DATA_DIR / "leave_balance.csv")
expenses_df = pd.read_csv(DATA_DIR / "expense_records.csv")
assets_df = pd.read_csv(DATA_DIR / "IT_assets.csv")
offices_df = pd.read_csv(DATA_DIR / "office_locations.csv")


# -----------------------------
# EMPLOYEE DETAILS
# -----------------------------

def get_employee_details(employee_id):
    employee = employees_df[
        employees_df["employee_id"].str.upper() == employee_id.upper()
    ]

    if employee.empty:
        return f"Employee {employee_id} not found."

    return employee.iloc[0].to_dict()


# -----------------------------
# LEAVE BALANCE
# -----------------------------

def get_leave_balance(employee_id):
    leave = leave_df[
        leave_df["employee_id"].str.upper() == employee_id.upper()
    ]

    if leave.empty:
        return f"Leave record for {employee_id} not found."

    return leave.iloc[0].to_dict()


# -----------------------------
# EXPENSE RECORDS
# -----------------------------

def get_expense_records(employee_id):
    expenses = expenses_df[
        expenses_df["employee_id"].str.upper() == employee_id.upper()
    ]

    if expenses.empty:
        return f"No expense records found for {employee_id}."

    return expenses.to_dict(orient="records")


# -----------------------------
# IT ASSETS
# -----------------------------

def get_it_assets(employee_id):
    assets = assets_df[
        assets_df["employee_id"].str.upper() == employee_id.upper()
    ]

    if assets.empty:
        return f"No IT assets found for {employee_id}."

    return assets.to_dict(orient="records")


# -----------------------------
# OFFICE DETAILS
# -----------------------------

def get_office_details(city):
    office = offices_df[
        offices_df["city"].str.lower() == city.lower()
    ]

    if office.empty:
        return f"No BlueOrbit office found in {city}."

    return office.iloc[0].to_dict()


def get_total_expenses(employee_id):
    expenses = expenses_df[
        expenses_df["employee_id"].str.upper() == employee_id.upper()
    ]

    if expenses.empty:
        return f"No expense records found for {employee_id}."

    total = expenses["amount"].sum()

    return {
        "employee_id": employee_id.upper(),
        "total_expenses": float(total)
    }

def get_expense_summary(employee_id):
    expenses = expenses_df[
        expenses_df["employee_id"].str.upper() == employee_id.upper()
    ]

    if expenses.empty:
        return f"No expense records found for {employee_id}."

    return {
        "employee_id": employee_id.upper(),
        "total_expenses": float(expenses["amount"].sum()),
        "approved": float(
            expenses.loc[expenses["status"] == "Approved", "amount"].sum()
        ),
        "pending": float(
            expenses.loc[expenses["status"] == "Pending", "amount"].sum()
        ),
        "rejected": float(
            expenses.loc[expenses["status"] == "Rejected", "amount"].sum()
        )
    }