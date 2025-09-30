import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# --- Load Data ---
@st.cache_data
def load_data():
    dropouts = pd.read_csv("all_dropouts.csv")
    summary = pd.read_csv("summary_by_group.csv")
    financial = pd.read_csv("financial_summary.csv")
    multidim = pd.read_csv("multi_dim_summary.csv")
    return dropouts, summary, financial, multidim

all_dropouts, summary_by_group, financial_summary, multi_dim_summary = load_data()

# --- Sidebar Filters ---
st.sidebar.title("Filters")

# Basic filters
program_filter = st.sidebar.multiselect("Select Program", all_dropouts["Program"].dropna().unique())
gender_filter = st.sidebar.multiselect("Select Gender", all_dropouts["Gender"].dropna().unique())
dropout_filter = st.sidebar.radio("Dropout Timing", ["All", "Early", "Late"])

# Session/Semester filters
joining_sem_filter = st.sidebar.multiselect(
    "Joining Semester", 
    all_dropouts["Joining_Semester"].dropna().unique()
)
dropout_sem_filter = st.sidebar.multiselect(
    "Dropout Semester", 
    all_dropouts["Dropout_Semester"].dropna().unique()
)

# Advanced filters
st.sidebar.markdown("### Advanced Filters")
gpa_min, gpa_max = st.sidebar.slider(
    "CGPA Range",
    min_value=float(all_dropouts["CGPA"].min()),
    max_value=float(all_dropouts["CGPA"].max()),
    value=(float(all_dropouts["CGPA"].min()), float(all_dropouts["CGPA"].max()))
)

attendance_min, attendance_max = st.sidebar.slider(
    "Attendance % Range",
    min_value=float(all_dropouts["Present_Percentage"].min()),
    max_value=float(all_dropouts["Present_Percentage"].max()),
    value=(float(all_dropouts["Present_Percentage"].min()), float(all_dropouts["Present_Percentage"].max()))
)

# Optional reason filter (only if column exists)
reason_filter = []
if "Dropout_Reason" in all_dropouts.columns:
    reason_filter = st.sidebar.multiselect(
        "Dropout Reason", 
        all_dropouts["Dropout_Reason"].dropna().unique()
    )

# --- Apply Filters ---
filtered = all_dropouts.copy()

if program_filter:
    filtered = filtered[filtered["Program"].isin(program_filter)]
if gender_filter:
    filtered = filtered[filtered["Gender"].isin(gender_filter)]
if dropout_filter != "All":
    filtered = filtered[filtered["Is_Early_Dropout"] == dropout_filter]
if joining_sem_filter:
    filtered = filtered[filtered["Joining_Semester"].isin(joining_sem_filter)]
if dropout_sem_filter:
    filtered = filtered[filtered["Dropout_Semester"].isin(dropout_sem_filter)]
if reason_filter:
    filtered = filtered[filtered["Dropout_Reason"].isin(reason_filter)]

filtered = filtered[
    (filtered["CGPA"] >= gpa_min) & (filtered["CGPA"] <= gpa_max) &
    (filtered["Present_Percentage"] >= attendance_min) & (filtered["Present_Percentage"] <= attendance_max)
]


# --- Dashboard ---
st.title("🎓 Student Dropout Dashboard")

# KPIs
col1, col2, col3 = st.columns(3)
col1.metric("Total Dropouts", len(filtered))
col2.metric("Avg CGPA", f"{filtered['CGPA'].mean():.2f}")
col3.metric("Avg Pending Dues", f"{filtered['Pending_Overall'].mean():,.0f} PKR")

# Dropout by Program
st.subheader("Dropout Counts by Program")
if not filtered.empty:
    program_plot = px.bar(
        filtered.groupby(["Program", "Is_Early_Dropout"]).size().reset_index(name="Count"),
        x="Count", y="Program", color="Is_Early_Dropout", orientation="h", barmode="stack"
    )
    st.plotly_chart(program_plot, use_container_width=True)
else:
    st.warning("No data matches the selected filters.")

# Dropout by Session
st.subheader("Dropout Counts by Session")
if not filtered.empty:
    session_plot = px.bar(
        filtered.groupby(["Joining_Semester", "Is_Early_Dropout"]).size().reset_index(name="Count"),
        x="Joining_Semester", y="Count", color="Is_Early_Dropout", barmode="stack"
    )
    session_plot.update_xaxes(type="category")
    st.plotly_chart(session_plot, use_container_width=True)

# Financial Patterns
st.subheader("Financial Patterns")
if not filtered.empty:
    fin_plot = px.box(filtered, x="Is_Early_Dropout", y="Pending_Overall", color="Is_Early_Dropout")
    st.plotly_chart(fin_plot, use_container_width=True)

# Academic Correlations
st.subheader("Correlation Heatmap")
if not filtered.empty:
    numeric_cols = ["CGPA", "Previous_Percentage", "Present_Percentage", "Total_Absent", "Pending_Overall", "Age_at_Join"]
    corr = filtered[numeric_cols].corr()

    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap="Blues", ax=ax)
    st.pyplot(fig)
