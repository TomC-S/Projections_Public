import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# --- Streamlit Page Settings ---
st.title("📊 Projections + DAU +LTV +ROAS")

# --- Constants ---
DEFAULT_GRAPH_WIDTH = 800
DEFAULT_GRAPH_HEIGHT = 400

days_to_calc= 365
days_in_year =365
days_in_month = int(days_in_year/12)
months_to_calc = int(days_in_year/30)

# -----------------------------------------
# 📌 RETENTION MODEL SELECTION
# -----------------------------------------

ret_days= [0, 1, 3, 7, 14, 30, 60, 90, 180, 365]
# --- Retention Data ---
retention_models = {
    "Default": { # basic PC data chagpt research dtaa 
         "retention_rates": [1.0, 0.45, 0.35, 0.25, 0.175, 0.15, 0.075, 0.04, 0.02, 0.015],
    },
    "Rust": {
        "retention_rates": [1.0, 0.5, 0.3476, 0.23, 0.1451, 0.09, 0.0692, 0.0563, 0.0395, 0.0276],
    },
    "Fortnite": {
        "retention_rates": [1.0, 0.32, 0.2396, 0.16, 0.1373, 0.11, 0.0858, 0.0753, 0.0602, 0.0479],
    },
    "Apex Legends": {
       "retention_rates": [1.0, 0.30, 0.2045, 0.11, 0.0775, 0.05, 0.0352, 0.0283, 0.0194, 0.0132],
    },
      "Day_Z": {
         "retention_rates": [1.0, 0.40, 0.35, 0.33, 0.3, 0.25, 0.20, 0.15, 0.12, 0.10],
    },
}

# --- Select Retention Model ---
option = st.sidebar.selectbox("Select Retention Model", list(retention_models.keys()))
retention_days =  ret_days
retention_rate = retention_models[option]["retention_rates"]

# --- Interpolate Retention Curve ---
retention_on_days = np.interp(range(days_to_calc), retention_days, retention_rate)

# --- Display Retention Graph ---
df_retention = pd.DataFrame({"Days": range(days_to_calc), "Retention": retention_on_days})
fig_retention = px.line(df_retention, x="Days", y="Retention", title=f"Retention Curve - {option}",
                        width=DEFAULT_GRAPH_WIDTH, height=DEFAULT_GRAPH_HEIGHT)
st.plotly_chart(fig_retention)

# -----------------------------------------
# 📌 DAILY ACTIVE USERS (DAU) SIMULATION
# -----------------------------------------
st.title("DAU")

st.sidebar.title("📈 Monthly Campaign Installs")

monthly_campaigns = []
months_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                 "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

months_default_installs = [ 
    200_000,  # Month 1
    50_000,  # Month 2
    30_000,   # Month 3
    40_000,   # Month 4
    20_000,   # Month 5
    60_000,   # Month 6
    25_000,   # Month 7
    15_000,   # Month 8
    20_000,   # Month 9
    15_000,   # Month 10
    30_000,   # Month 11
    10_000    # Month 12
    ]


for i in range(12):
    val = st.sidebar.number_input(
        f"{months_labels[i]} Campaign Installs", 
        min_value=0, value=months_default_installs[i], step=500
    )
    monthly_campaigns.append(val)

daily_installs = []
for month_index in range(12):
    installs_per_day = monthly_campaigns[month_index] // days_in_month
    for _ in range(days_in_month):
        daily_installs.append(installs_per_day)

# In case 12 * 30 < 365, pad the remaining days
while len(daily_installs) < days_to_calc:
    daily_installs.append(0)

st.write("### 📊 Daily Installs from Monthly Campaigns")
st.line_chart(pd.DataFrame({
    "Day": range(days_to_calc),
    "Installs": daily_installs
}))

# install_val = st.slider("New Players per Day", 1, 10000, 100)
# daily_installs = [install_val] *days_to_calc

def install_func(install_day):
    return daily_installs[install_day]

def calculate(days_to_calc, months_to_calc, retention_over_days, install_function):
    """
    Projects DAU, MAU, and installs over time based on retention data.

    Args:
        days_to_calc (int): Number of days to simulate.
        months_to_calc (int): Number of months to simulate.
        retention_over_days (array): Retention rates for each day.
        install_function (function): Function returning installs for each day.

    Returns:
        dict: DataFrames for DAU, MAU, and installs.
    """
    dau = np.zeros(days_to_calc, dtype=int)
    mau = np.zeros(months_to_calc, dtype=int)
    installs_on_day = np.zeros(days_to_calc, dtype=int)

    for install_day in range(days_to_calc):
        new_installs = install_function(install_day)
        installs_on_day[install_day] += new_installs

        retained_users = (new_installs * retention_over_days[:days_to_calc - install_day]).astype(int)
        dau[install_day:install_day + len(retained_users)] += retained_users

        for retained_day, users in enumerate(retained_users):
            month_index = (install_day + retained_day) // 30
            if month_index < months_to_calc and retained_day % 30 == 0:
                mau[month_index] += users

    return {
        'mau': pd.DataFrame({'month': np.arange(months_to_calc), 'mau': mau}),
        'dau': pd.DataFrame({'day': np.arange(days_to_calc), 'dau': dau}),
        'installs_on_day': pd.DataFrame({'day': np.arange(days_to_calc), 'installs': installs_on_day})
    }




# --- Project DAU ---
resultDAU = calculate(days_to_calc, months_to_calc, retention_on_days, install_func)

# --- DAU Graph ---
fig_dau = px.line(resultDAU["dau"], x="day", y="dau", title="DAU Over Time",
                  width=DEFAULT_GRAPH_WIDTH, height=DEFAULT_GRAPH_HEIGHT)
st.plotly_chart(fig_dau)

# -----------------------------------------
# 📌 PRICING & REVENUE SIMULATION
# -----------------------------------------
st.sidebar.title("Pricing & Revenue")

# --- User Inputs ---
arpdau = st.sidebar.number_input("ARPDAU ($)", min_value=0.1, value=0.50, step=0.01)
paid_price = st.sidebar.number_input("Paid Price ($)", min_value=0.1, value=29.99, step=0.10)
battle_pass_price = st.sidebar.number_input("BP Price ($)", min_value=0.1, value=9.99, step=0.10)
cpi = st.sidebar.number_input("CPI ($)", min_value=0.1, value=1.87, step=0.01)


# Truncate extra if too many (e.g., leap year)
daily_installs = daily_installs[:days_to_calc]

# --- Compute Daily Revenue ---
df_revenue = pd.DataFrame({
    "Day": range(days_to_calc),
    "DAU": resultDAU["dau"]["dau"],
    "Daily Revenue ($)": [d * arpdau for d in resultDAU["dau"]["dau"]],
    "Paid Revenue ($)": np.array(daily_installs) * paid_price,
})

df_revenue["Total Daily Revenue ($)"] = df_revenue["Daily Revenue ($)"] + df_revenue["Paid Revenue ($)"]

# --- Compute Monthly Revenue ---
df_revenue["Monthly Revenue ($)"] = df_revenue["Total Daily Revenue ($)"].rolling(window=30).sum().round(2)
df_revenue["Monthly Revenue ($)"] = df_revenue["Monthly Revenue ($)"].fillna(df_revenue["Total Daily Revenue ($)"].cumsum())

df_monthly_revenue = df_revenue[df_revenue["Day"] % 30 == 0]

# --- Display Revenue Data ---
st.title("Total Monthly Revenue Calculation")
st.dataframe(df_monthly_revenue)

# --- Monthly Revenue Graph ---
fig_revenue = px.line(df_monthly_revenue, x="Day", y="Monthly Revenue ($)",
                      title="Monthly Revenue Over Time (Including Paid Price)",
                      width=DEFAULT_GRAPH_WIDTH, height=DEFAULT_GRAPH_HEIGHT)
st.plotly_chart(fig_revenue)

# --- Download Revenue Data ---
csv = df_monthly_revenue.to_csv(index=False).encode("utf-8")
st.download_button("Download Monthly Revenue Data", csv, "monthly_revenue.csv", "text/csv")

# -----------------------------------------
# 📌 LIFETIME VALUE (LTV) CALCULATION
# -----------------------------------------
st.title("LTV")

days_since_install = np.arange(1, 365)  # Days from 1 to 365
retention_on_days = np.interp(days_since_install, retention_days, retention_rate)

# --- Compute ARPU ---
arpu = np.cumsum(arpdau * retention_on_days)  # Cumulative revenue per user

# --- Compute LTV ---
churn_rate = max(1 - retention_on_days[29], 0.01)  # Avoid divide by zero
customer_lifetime = 1 / churn_rate
ltv = arpu * customer_lifetime
ltv_with_paid = ltv + paid_price  # Adding paid price to each LTV point

# --- LTV Data ---
df_ltv = pd.DataFrame({
    "Days Since Install": days_since_install,
    "LTV ($)": ltv,
    "LTV + Paid Price ($)": ltv_with_paid
})

# --- Display LTV Table ---
st.write("### 📊 LTV Predictions Over Time")
st.dataframe(df_ltv)

# --- LTV Graph with Both Curves ---
fig_ltv = px.line(df_ltv, x="Days Since Install", 
                  y=["LTV ($)", "LTV + Paid Price ($)"],
                  title="LTV vs. Days Since Install (With and Without Paid Price)",
                  labels={"value": "LTV ($)", "Days Since Install": "Days"},
                  width=DEFAULT_GRAPH_WIDTH, height=500)

st.plotly_chart(fig_ltv)

# --- Download LTV Data ---
csv = df_ltv.to_csv(index=False).encode("utf-8")
st.download_button("Download LTV Data", csv, "ltv_data.csv", "text/csv")

# --- CPI and ROAS
st.write("### 📊 CPI and ROAS")
# --- slider for organic vs paid 
organic = st.slider("Organic Users (%)", 0, 100, 50)
paid_users_percentage = 100 - organic  # Percentage of paid users

# --- Days for ROAS Calculation ---
days = [1, 3, 7, 14, 30, 60, 90]
ltv_values = np.interp(days, np.arange(1, 365), np.cumsum(arpdau * np.interp(np.arange(1, 365), retention_days, retention_rate)))

# --- Calculate ROAS ---
roas = (ltv_values + paid_price) / cpi

# --- Create DataFrame ---
df_roas = pd.DataFrame({
    "Day": days,
    "LTV ($)": ltv_values.round(2),
    "ROAS": roas.round(2),
    "ROAS %": (roas * 100).round(2)
})

# --- Display ROAS Table ---
st.write("### 📊 ROAS Calculation Table")
st.dataframe(df_roas)

# --- Plot ROAS Over Time ---
fig_roas = px.line(df_roas, x="Day", y="ROAS %", title="ROAS % Over Time",
                   labels={"ROAS %": "ROAS (%)", "Day": "Days"},
                   markers=True)
st.plotly_chart(fig_roas)

# --- Download ROAS Data ---
csv = df_roas.to_csv(index=False).encode("utf-8")
st.download_button("Download ROAS Data", csv, "roas_data.csv", "text/csv")

st.sidebar.title("🎯 Seasonal Campaign Cost (2-Month Seasons)")

# --- Define 2-month seasons
season_labels = ["S1 (Jan-Feb)", "S2 (Mar-Apr)", "S3 (May-Jun)",
                 "S4 (Jul-Aug)", "S5 (Sep-Oct)", "S6 (Nov-Dec)"]

# --- Aggregate monthly installs into 2-month seasons
seasonal_installs = [
    monthly_campaigns[i] + monthly_campaigns[i + 1]
    for i in range(0, 12, 2)
]
season_costs = []
# --- Automatically calculate season costs based on CPI
for i in range(6):
    cost = st.sidebar.number_input(
        f"{months_labels[i]} Season Cost ($)",
        min_value=0.0,
        value=93528.0,
        step=500.0
    )
    season_costs.append(cost)

# --- Battle Pass Conversion
battle_pass_conversion = st.sidebar.slider("Battle Pass Conversion Rate (%)", 0, 100, 10)
battle_pass_conversion_rate = battle_pass_conversion / 100

# --- Calculate BP revenue and ROI for each 2-month season
monthly_bp_revenue = []
monthly_roi = []

for i in range(6):
    # Calculate the day range for the two-month season
    start_day = i * 60
    end_day = start_day + 60

    # Sum up all active players (DAU) in those days
    active_players = resultDAU["dau"]["dau"][start_day:end_day].sum()
    cost = season_costs[i]

    # Assume a percentage of active players purchase the battle pass
    bp_buyers = active_players * battle_pass_conversion_rate
    revenue = bp_buyers * battle_pass_price
    roi = (revenue - cost) / cost if cost > 0 else 0

    monthly_bp_revenue.append(round(revenue, 2))
    monthly_roi.append(round(roi, 2))

# --- Create DataFrame for display
df_roi = pd.DataFrame({
    "Season": season_labels,
    "Season Cost ($)": season_costs,
    "Battle Pass Revenue ($)": monthly_bp_revenue,
    "ROI": monthly_roi,
    "ROI %": [f"{r * 100:.2f}%" for r in monthly_roi]
})

# --- Show table and chart
st.write("### 📈 Season Cost vs ROI Table (2-Month Seasons)")
st.dataframe(df_roi)

fig_roi = px.bar(df_roi, x="Season", y="ROI", title="📊 ROI by 2-Month Season",
                 labels={"ROI": "Return on Investment"},
                 text="ROI %", height=400)
st.plotly_chart(fig_roi)
