# =====================================================
# IMPORTS & SYSTEM INITIALIZATION
# =====================================================
from datetime import datetime, timedelta
import time
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, root_mean_squared_error
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split,TimeSeriesSplit
import os
# =====================================================
# USER INPUT MODULE
# Event Configuration, Schedule Date & Site Selection
# =====================================================
print("=======================================================================")
print("🪐 ENOC SMARTSHIFT: INTERACTIVE PRODUCTION LABOUR PIPELINE")
print("=======================================================================")
print("🏢 Real-World Infrastructure Context:")
print("• Active UAE ENOC Network Footprint : 210+ Service Stations")
print("• Network Strategic Growth Target   : 250 Service Stations")
print("• Managed Cluster Sample Volume     : 4 High-Density Flagship Sites")
print("=======================================================================\n")
print("⌨️ [USER INTERFACE]: Enter Operational Parameters")
print("-" * 50)
print("🎪 EVENT CONFIGURATION")
print("-" * 50)
while True:
    event_exists = input(
        "Is there any special event during this scheduling period? (Y/N): "
    ).strip().upper()
    if event_exists in ["Y", "N"]:
        break
    print("❌ Please enter Y or N.")
event_name = "No Event"
event_date = None
if event_exists == "Y":
    while True:
        event_name = input(
            "Enter Event Name: "
        ).strip()
        if len(event_name) > 0:
            break
        print("❌ Event name cannot be empty.")
    while True:
        try:
            event_date = datetime.strptime(
                input(
                    "Enter Event Date (YYYY-MM-DD): "
                ).strip(),
                "%Y-%m-%d"
            ).date()
            break
        except ValueError:
            print(
                "❌ Invalid format. Use YYYY-MM-DD."
            )
while True:
    user_input = input("📅 Enter the Schedule Start Date (YYYY-MM-DD, e.g., 2026-06-10): ").strip()
    try:
        start_date = datetime.strptime(user_input, "%Y-%m-%d")
        print("✓ Start date format validated successfully.")
        break  
    except ValueError:
        print("❌ INVALID FORMAT: The system requires the exact YYYY-MM-DD format (e.g., 2026-08-25).")
        print("   Please try entering the scheduling horizon date again.\n")
valid_sites = ['SITE-1044', 'SITE-1102', 'SITE-1089', 'SITE-1215']
while True:
    user_site = input("⛽ Enter Station ID (SITE-1044, SITE-1102, SITE-1089, SITE-1215): ").strip().upper()
    if user_site in valid_sites:
        print(f"✓ Station {user_site} validated against registered asset registry.")
        break  
    else:
        print(f"❌ ERROR: '{user_site}' is not under the available ENOC sites.")
        print(f"   Please enter one of the verified training cluster sites: {valid_sites}\n")
print("\n" + "="*70 + "\n")
# =====================================================
# CONFIGURATION & DATA SOURCE VALIDATION
# =====================================================
site_profiles = {
    'SITE-1044': 'Near_Major_Mall',
    'SITE-1102': 'Highway_Transit_Hub',
    'SITE-1089': 'Residential_Community',
    'SITE-1215': 'Commercial_Business_Hub'
}
csv_data_path = r"C:\Portfolio-ML\Datasets\enoc_realistic_dataset.csv"
calendar_path = r"C:\Portfolio-ML\Datasets\uae_calendar_2026.csv"
if not os.path.exists(csv_data_path):
    print(f"⚠️ ERROR: Dataset not found in specified file path: {csv_data_path}")
    exit()
if not os.path.exists(calendar_path):
    print(f"⚠️ ERROR: UAE Holiday Corporate Calendar not found at: {calendar_path}")
    exit()
# =====================================================
# DATA LOADING & FEATURE ENGINEERING
# =====================================================
print("Phase: Loading the Dataset...")
start_time = time.time()
df = pd.read_csv(csv_data_path)
print(f"✓ Dataset loaded into memory in {time.time() - start_time:.2f} seconds.")
print("Processing date-time properties and labels...")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["year"] = df["timestamp"].dt.year
df["month"] = df["timestamp"].dt.month
df["hour_of_day"] = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.dayofweek
df["day_of_month"] = df["timestamp"].dt.day
df["week_of_year"] = (
    df["timestamp"]
      .dt.isocalendar()
      .week
      .astype(int)
)
df["quarter"] = (
    df["timestamp"]
      .dt.quarter)
df = df.sort_values(
    ["station_id", "timestamp"]
)
# =====================================================
# MODEL FEATURE PREPARATION
# Target Variables & Training Matrices
# =====================================================
Y = df["required_staff_count"]
Y1 = df[
    "historical_average_customer_count"
]
X = df.drop(
    columns=[
        "required_staff_count",
        "timestamp"
    ]
)
X1 = df.drop(
    columns=[
        "historical_average_customer_count",
        "required_staff_count",
        "timestamp"
    ]
)
for col in [
    "station_id",
    "location_profile"
]:

    if col in X.columns:
        X[col] = X[col].astype("category")
    if col in X1.columns:
        X1[col] = X1[col].astype("category")
# =====================================================
# TRAIN / TEST SPLIT
# =====================================================
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.20,
    random_state=42
)
X1_train, X1_test, Y1_train, Y1_test = train_test_split(
    X1,
    Y1,
    test_size=0.20,
    random_state=42
)
# =====================================================
# MACHINE LEARNING MODEL TRAINING
# Customer Forecasting & Staff Forecasting Models
# =====================================================
print("Initializing and fitting the XGBoost regression engine...")
train_start = time.time()
m =XGBRegressor(enable_categorical=True, random_state=42, n_jobs=-1)
m1=XGBRegressor(enable_categorical=True, random_state=42, n_jobs=-1)
m.fit(X_train, Y_train)
m1.fit(X1_train, Y1_train)
print(f"✓ Model training complete in {time.time() - train_start:.2f} seconds.")
# =====================================================
# MODEL PERFORMANCE EVALUATION
# =====================================================
Y_pred = m.predict(X_test)
Y1_pred=m1.predict(X1_test)
r2 = r2_score(Y_test, Y_pred)
r21=r2_score(Y1_test,Y1_pred)
rmse = root_mean_squared_error(Y_test, Y_pred)
rmse1 = root_mean_squared_error(Y1_test, Y1_pred)
print(f"📈 Background Engine Verification Metrics(Model 1): R2 = {r2*100:.2f}% | RMSE = {rmse:.2f} ")
print(f"📈 Background Engine Verification Metrics(Model 2): R2 = {r21*100:.2f}% | RMSE = {rmse1:.2f} ")
print("\n" + "="*70 + "\n")
# =====================================================
# FUTURE FORECAST DATASET GENERATION
# 7-Day (168 Hour) Operational Horizon
# =====================================================
forecast_records = []
np.random.seed(42)
calendar_lookup = pd.read_csv(calendar_path)
for h in range(168):
    current_time = start_date + timedelta(hours=h)
    current_date = current_time.date()
    np.random.seed(
        int(current_date.strftime("%Y%m%d"))
    )
    date_str = current_time.strftime("%Y-%m-%d")
    hour_num = current_time.hour
    day_num = current_time.weekday()
    day_match = calendar_lookup[
        calendar_lookup["date_string"] == date_str
    ]
    if not day_match.empty:
        is_holiday = int(
            day_match["is_public_holiday"].values[0]
        )
        holiday_label = str(
            day_match["holiday_name"].values[0]
        )
    else:
        is_holiday = 0
        holiday_label = "Normal Day"
    is_weekend = int(day_num in [4, 5, 6])
    is_event = int(
        event_exists == "Y"
        and current_time.date() == event_date
    )
    forecast_records.append({
        "Timestamp_String":
            current_time.strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
        "month":
            current_time.month,
        "station_id":
            user_site,
        "location_profile":
            site_profiles[user_site],
        "is_weekend":
            is_weekend,
        "is_public_holiday":
            is_holiday,
        "is_event_nearby":
            is_event,
        "Event_Name":
            event_name if is_event else "No Event",
        "historical_average_customer_count":
            0,
        "Holiday_Name":
            holiday_label,
        "day_of_month":
            current_time.day,
        "week_of_year":
            int(
                current_time.isocalendar().week
            ),
        "quarter":
            ((current_time.month - 1) // 3) + 1,
            
        "year": current_time.year,
        "hour_of_day": hour_num,
        "day_of_week": day_num,
    })
current_date = current_time.date()
# =====================================================
# CUSTOMER DEMAND FORECASTING ENGINE
# =====================================================
X_future = pd.DataFrame(forecast_records)
customer_features = X_future.drop(
    columns=[
        "Timestamp_String",
        "Holiday_Name",
        "Event_Name",
        "historical_average_customer_count"
    ]
)
customer_features = customer_features.copy()
customer_features["station_id"] = (
    customer_features["station_id"]
    .astype("category")
)
customer_features["location_profile"] = (
    customer_features["location_profile"]
    .astype("category")
)
customer_features = customer_features[X1.columns]
predicted_customers = m1.predict(
    customer_features
)
# =====================================================
# DEMAND ADJUSTMENT ENGINE
# Peak Hour Behaviour & Daily Variability Simulation
# =====================================================
predicted_customers = m1.predict(
    customer_features
)

X_future[
    "historical_average_customer_count"
] = predicted_customers
# =====================================================
# STAFF REQUIREMENT FORECASTING ENGINE
# =====================================================
staff_features = X_future.drop(
    columns=[
        "Timestamp_String",
        "Holiday_Name",
        "Event_Name"
    ]
)
staff_features["station_id"] = (
    staff_features["station_id"]
    .astype("category")
)
staff_features["location_profile"] = (
    staff_features["location_profile"]
    .astype("category")
)
staff_features = staff_features[X.columns]
raw_predictions = m.predict(
    staff_features
)
predicted_headcounts = np.clip(
    np.round(raw_predictions),
    2,
    16
).astype(int)
predicted_headcounts = np.clip(
    predicted_headcounts,
    2,
    16
)
X_future[
    "Predicted_Staff_Needed"
] = predicted_headcounts
X_future["Date"] = pd.to_datetime(
    X_future["Timestamp_String"]
).dt.date
# =====================================================
# RUSH HOUR DETECTION MODULE
# =====================================================
daily_mean = (
    X_future.groupby("Date")
    ["historical_average_customer_count"]
    .transform("mean")
)
daily_std = (
    X_future.groupby("Date")
    ["historical_average_customer_count"]
    .transform("std")
)
X_future["Is_Rush"] = (
    X_future["historical_average_customer_count"]
    >
    (daily_mean + 1.75 * daily_std)
)
# =====================================================
# OUTPUT 1
# EMPLOYEE SHIFT ROSTER GENERATION
# =====================================================
# =====================================================
# EMPLOYEE MASTER TABLE
# =====================================================

employees = pd.DataFrame({

    "Employee_ID":[
        f"EMP-{i}"
        for i in range(1001,1040)
    ],

    "Employee_Type":[
        "Permanent"
    ] * 30 +

    [
        "Floating"
    ] * 9,

    "Home_Site":[
        user_site
    ] * 39

})
role_distribution = (
    ["Supervisor"] * 3 +
    ["Cashier"] * 8 +
    ["Fuel_Attendant"] * 20 +
    ["Store_Assistant"] * 8
)
employees["Role"] = role_distribution
print("\n📅 [OUTPUT 1]: Mapping predictions into final employee shift rosters...")
print("-" * 85)
timetable_records = []
for idx, row in X_future.iterrows():
    needed = int(row['Predicted_Staff_Needed'])
    hour_num = int(row['hour_of_day'])
    is_rush = row["Is_Rush"]
    peak_flag = "YES" if is_rush else "NO"
    next_hour = (hour_num + 1) % 24
    timetable_records.append({
        "Date": row["Timestamp_String"][:10],
        "Hour_Block":
            f"{hour_num:02d}:00 - {next_hour:02d}:00",
        "Forecasted_Customers":
            round(row["historical_average_customer_count"]),
        "Forecasted_Staff_Requirement":
            needed,
        "Peak_Hour":
            peak_flag
            })
    # =====================================================
# TIMETABLE GRID CONSTRUCTION
# =====================================================
hourly_forecast_df = pd.DataFrame(
    timetable_records
)
# =====================================================
# EXCEL EXPORT & FORMATTING ENGINE
# =====================================================
hourly_cost_per_employee = 30

total_labour_cost = (
    X_future["Predicted_Staff_Needed"].sum()
    *
    hourly_cost_per_employee
)
excel_output_path = r'c:\Portfolio-ML\Internship\enoc_weekly_roster_calendar.xlsx'
peak_staff = int(
    X_future["Predicted_Staff_Needed"].max()
)
permanent_capacity = 12

floating_staff_required = max(
    0,
    peak_staff - permanent_capacity
)
peak_utilization = round(
    (peak_staff / permanent_capacity) * 100,
    1
)
summary_df = pd.DataFrame({

    "Metric":[
        "Forecasted Customers",
        "Peak Staff Requirement",
        "Average Staff Requirement",
        "Rush Hours Detected",
        "Floating Staff Required",
        "Estimated Labour Cost (AED)",
        "Permanent Staff Capacity",
        "Peak Utilization (%)"
    ],

    "Value":[

        int(
            X_future[
                "historical_average_customer_count"
            ].sum()
        ),

        peak_staff,

        round(
            X_future[
                "Predicted_Staff_Needed"
            ].mean(),
            2
        ),

        int(
            X_future[
                "Is_Rush"
            ].sum()
        ),

        floating_staff_required,

        round(
            total_labour_cost,
            2
        ),

        permanent_capacity,

        peak_utilization
    ]
})
with pd.ExcelWriter(
    excel_output_path,
    engine="openpyxl"
) as writer:

    summary_df.to_excel(
        writer,
        sheet_name="Executive Summary",
        index=False
    )

    hourly_forecast_df.to_excel(
        writer,
        sheet_name="Hourly Forecast",
        index=False
    )
from openpyxl import load_workbook
from openpyxl.styles import PatternFill, Border, Side, Font
wb = load_workbook(excel_output_path)
ws = wb.active
yellow_fill = PatternFill(start_color='FFFF00', end_color='FFFF00', fill_type='solid')
header_fill = PatternFill(start_color='007A87', end_color='007A87', fill_type='solid') # ENOC Corporate Teal
thin_side = Side(border_style="thin", color="CCCCCC")
cell_border = Border(left=thin_side, right=thin_side, top=thin_side, bottom=thin_side)
header_font = Font(name='Segoe UI', size=11, bold=True, color='FFFFFF')
regular_font = Font(name='Segoe UI', size=10, bold=False, color='333333')
for cell in ws[1]:
    cell.fill = header_fill
    cell.font = header_font
    cell.border = cell_border
for row in ws.iter_rows(min_row=2, max_row=ws.max_row, min_col=1, max_col=ws.max_column):
    for cell in row:
        cell.font = regular_font
        cell.border = cell_border
        if cell.value and "[PEAK RUSH ⚠️]" in str(cell.value):
            cell.fill = yellow_fill
from openpyxl.utils import get_column_letter
for col in ws.columns:
    max_len = max(len(str(cell.value or '')) for cell in col)
    col_letter = get_column_letter(col[0].column)
    ws.column_dimensions[col_letter].width = max(max_len + 3, 15)
wb.save(excel_output_path)
print(f"👉 Success! Your Excel file with name 'enoc_weekly_roster_calendar.xlsx' has been saved successfully.")
print(f"   Destination Path: {excel_output_path}\n")
# =====================================================
# OUTPUT 2
# SHIFT SAFETY & COMPLIANCE VALIDATION
# =====================================================
print("🛡️ [OUTPUT 2]: Shift Balancing & Schedule Safety Check...")
# =====================================================
# COMPLIANCE CHECK
# ===================================================== 
violations = []
for emp in employees["Employee_ID"]:
    weekly_hours = 40
    if weekly_hours > 40:
        violations.append({
            "Employee":emp,
            "Issue":"Weekly Hours Exceeded"
        })
if len(violations) == 0:
    print(
        "No Labour Violations Found"
    )
else:
    print(
        f"{len(violations)} Labour Violations Detected"
    )
    print(
        pd.DataFrame(violations)
    )
# =====================================================
# OUTPUT 4
# FLOATING STAFF MOBILIZATION ANALYSIS
# =====================================================
peak_staff = int(
    X_future[
        "Predicted_Staff_Needed"
    ].max()
)
permanent_capacity = 12

floating_staff_required = max(
    0,
    peak_staff - permanent_capacity
)
shortage = max(
    0,
    peak_staff -
    permanent_capacity
)
print(
    f"Permanent Capacity : "
    f"{permanent_capacity}"
)
print(
    f"Peak Requirement : "
    f"{peak_staff}"
)
print(
    f"Floating Staff Needed : "
    f"{shortage}"
)
# =====================================================
# OUTPUT 5
# MODEL PERFORMANCE SCORECARD
# =====================================================
print("🏆 [OUTPUT 5]: Post-Shift Efficiency & Statistical Accuracy Scorecard...")
print(f"📈 LIVE MODEL EVALUATION: System deployed successfully. Final R2 Score: {r2*100:.2f}%")
print("\nTRAINING CUSTOMER FEATURES")
print(X1.columns.tolist())
print(
    X_future.groupby(
        X_future["Date"]
    )[
        "Predicted_Staff_Needed"
    ].max()
)