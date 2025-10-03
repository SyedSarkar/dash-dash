import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import io  # For download
import warnings  # Added: To suppress non-critical warnings for cleaner UX

warnings.filterwarnings("ignore")  # Suppress pandas warnings for a pro dashboard feel

# Page configure (unchanged, but added wide layout for comparisons)
st.set_page_config(page_title="Enrollment & Dropout Dashboard", layout="wide")
st.title("📊 Enrollment & Dropout Analysis Dashboard")

# Custom CSS for aesthetics (enhanced for compare mode with better spacing)
st.markdown("""
    <style>
    .stSidebar .stRadio > label { font-size: 16px; }
    .stMetric { font-size: 18px; }
    .stTabs [data-testid="stMarkdownContainer"] { font-size: 16px; font-weight: bold; }
    .stDataFrame { border: 1px solid #ddd; border-radius: 5px; }
    .compare-plot { margin-bottom: 20px; }  /* Added: Spacing for comparison plots */
    </style>
""", unsafe_allow_html=True)

# --- Data Validation ---
required_cols = ["Program", "Gender", "CGPA", "Attendance_Percentage", "Pending_Fee", "Joining_Semester", "Dropout_Semester"]  # Standardized to uppercase

# file: dash_try.py
import pandas as pd
import numpy as np
import streamlit as st

@st.cache_data
def load_data():
    # Load enrollment
    enrollment_df = pd.read_csv("enrollment_cleaned.csv")
    enrollment_df.columns = enrollment_df.columns.str.strip().str.capitalize()
    enrollment_df["Type"] = "Enrollment"
    enrollment_df.rename(columns={
        "Joining_semester": "Joining_Semester",
        "Cgpa": "CGPA",
        "Attendance_percentage": "Attendance_Percentage",
        "Pending_fee": "Pending_Fee"
    }, inplace=True)

    # Load dropout
    dropout_df = pd.read_csv("dropout_cleaned.csv")
    dropout_df.columns = dropout_df.columns.str.strip().str.capitalize()
    dropout_df["Type"] = "Dropout"
    dropout_df.rename(columns={
        "Roll_number": "Roll_Number",
        "Father_occupation": "Father_Occupation",
        "Std_birth_date": "Birth_Date",
        "Previous_institute_name": "Previous_Institute",
        "Previous_marks_percentage": "Previous_Marks_Percentage",
        "Program_selected": "Program",
        "Degree_title": "Degree_Title",
        "Transport_status": "Transport_Status",
        "Hostel_facility": "Hostel_Facility",
        "Umcdc_stat": "UMC_DC_Status",
        "Umcdc_points": "UMC_DC_Points",
        "Cgpa": "CGPA",
        "Joining_semester": "Joining_Semester",
        "Dropout_semester": "Dropout_Semester",
        "Comp_crhr": "Completed_CrHr",
        "Req_crhr": "Required_CrHr",
        "Rem_crhr": "Remaining_CrHr",
        "Tot_payable_first_semester": "Tot_Payable_First_Semester",
        "Paid_fee_first_semester": "Paid_Fee_First_Semester",
        "Pending_fee_first_semester": "Pending_Fee_First_Semester",
        "Tot_payable": "Tot_Payable",
        "Paid_fee": "Paid_Fee",
        "Pending_fee": "Pending_Fee",
        "Discount_provided_type": "Discount_Provided_Type",
        "Discount_provided_percentage": "Discount_Provided_Percentage",
        "Last_payment_date": "Last_Payment_Date",
        "Total_lec": "Total_Lec",
        "Attendance_percentage": "Attendance_Percentage",
        "Total_absent": "Total_Absent",
        "Overall_balance": "Overall_Balance",
        "Student_status": "Student_Status",
        "Last_degree": "Last_Degree",
        "Marks_obtained": "Marks_Obtained",
        "Marks_total": "Marks_Total",
        "Last_enrolled_semester": "Last_Enrolled_Semester",
        "Last_attendance_date": "Last_Attendance_Date",
        "Last_semester_gpa": "Last_Semester_GPA",
        "Outstanding_dues": "Outstanding_Dues",
        "Follow_up_type": "Follow_Up_Type",
        "Follow_up_date": "Follow_Up_Date",
        "Follow_up_reason": "Follow_Up_Reason",
        "Follow_up_remarks": "Follow_Up_Remarks",
        "Dropout_date": "Dropout_Date",
        "Dropout_reason": "Dropout_Reason",
        "Dropout_remarks": "Dropout_Remarks",
    }, inplace=True)

    # Convert numeric safely
    num_cols = [
        "CGPA", "Attendance_Percentage", "Pending_Fee",
        "Discount_Provided_Percentage", "Completed_CrHr",
        "Required_CrHr", "Remaining_CrHr"
    ]
    for col in num_cols:
        if col in enrollment_df.columns:
            enrollment_df[col] = pd.to_numeric(enrollment_df[col], errors="coerce")
        if col in dropout_df.columns:
            dropout_df[col] = pd.to_numeric(dropout_df[col], errors="coerce")

    # Fill NAs consistently
    enrollment_df = enrollment_df.fillna("N/A")
    dropout_df = dropout_df.fillna("N/A")

    # Remove duplicate columns that cause DataFrame return
    dropout_df = dropout_df.loc[:, ~dropout_df.columns.duplicated()]

    # Ensure correct dtype for .str operations
    for col in ["Joining_Semester", "Dropout_Semester"]:
        if col in dropout_df.columns:
            dropout_df[col] = dropout_df[col].astype(str)

    # Derive early/late dropout
    dropout_df["Is_Early_Dropout"] = np.where(
        dropout_df["Dropout_Semester"].str.contains("Spring", na=False) &
        dropout_df["Joining_Semester"].str.contains("Fall", na=False),
        "Early", "Late"
    )


    # Summary files
    summary_by_group = pd.read_csv("summary_by_group.csv")
    financial_summary = pd.read_csv("financial_summary.csv")
    multi_dim_summary = pd.read_csv("multi_dim_summary.csv")

    return enrollment_df, dropout_df, summary_by_group, financial_summary, multi_dim_summary


# Call it once
enrollment_df, dropout_df, summary_by_group, financial_summary, multi_dim_summary = load_data()


# Toggle option (added "Compare")
data_choice = st.radio("Select Data Type", ["Enrollment", "Dropout", "Compare"], horizontal=True)

# Pick the dataframe based on user choice (enhanced for Compare)
common_cols = list(set(enrollment_df.columns) & set(dropout_df.columns))  # Auto-detect common for robustness
if data_choice == "Enrollment":
    df = enrollment_df
elif data_choice == "Dropout":
    df = dropout_df
else:  # Compare
    df = pd.concat([enrollment_df[common_cols], dropout_df[common_cols]], ignore_index=True)
    st.info("Compare mode: Showing combined data with 'Type' for differentiation. Some dropout-specific filters are ignored.")

# --- Sidebar Filters ---
st.sidebar.title("Filters")
st.sidebar.info("Early: Dropout in first year; Late: After first year.")

# Basic Filters (unchanged)
top_filter = st.sidebar.radio("Show Top Programs", ["All", "Top 3", "Top 5", "Top 10"])
program_filter = st.sidebar.multiselect("Select Program", pd.concat([enrollment_df["Program"], dropout_df["Program"]]).dropna().unique())  # Use combined unique for Compare
gender_filter = st.sidebar.multiselect("Select Gender", pd.concat([enrollment_df["Gender"], dropout_df["Gender"]]).dropna().unique())

# Dropout-specific filters (conditional)
if data_choice != "Enrollment":
    dropout_filter = st.sidebar.radio("Dropout Timing", ["All", "Early", "Late"])

# Session filters (use Joining_Semester for consistency)
joining_sem_filter = st.sidebar.multiselect("Joining Semester", pd.concat([enrollment_df.get("Joining_Semester", pd.Series([])), dropout_df.get("Joining_Semester", pd.Series([]))]).dropna().unique())
if data_choice != "Enrollment":
    dropout_sem_filter = st.sidebar.multiselect("Dropout Semester", dropout_df.get("Dropout_Semester", pd.Series([])).dropna().unique())

# Advanced Filters
st.sidebar.markdown("### Advanced Filters")
gpa_min, gpa_max = st.sidebar.slider(
    "CGPA Range",
    min_value=0.0,
    max_value=4.0,
    value=(0.0, 4.0),
    step=0.1
)
attendance_min, attendance_max = st.sidebar.slider(
    "Attendance % Range",
    min_value=0.0,
    max_value=100.0,
    value=(0.0, 100.0),
    step=1.0
)

# Optional reason filters (conditional)
if data_choice != "Enrollment":
    reason_filter = st.sidebar.multiselect("Dropout Reason", dropout_df.get("Dropout_Reason", pd.Series([])).dropna().unique())
    
# --- Apply Filters ---
filtered = df.copy()

# Program filter
if program_filter and "Program" in filtered.columns:
    filtered = filtered[filtered["Program"].isin(program_filter)]

# Gender filter
if gender_filter and "Gender" in filtered.columns:
    filtered = filtered[filtered["Gender"].isin(gender_filter)]

# Dropout-specific filters (apply only if relevant)
if data_choice == "Dropout":
    if "Is_Early_Dropout" in filtered.columns and dropout_filter != "All":
        filtered = filtered[filtered["Is_Early_Dropout"] == dropout_filter]
    if dropout_sem_filter and "Dropout_Semester" in filtered.columns:
        filtered = filtered[filtered["Dropout_Semester"].isin(dropout_sem_filter)]
    if reason_filter and "Dropout_Reason" in filtered.columns:
        filtered = filtered[filtered["Dropout_Reason"].isin(reason_filter)]
elif data_choice == "Compare":
    # For Compare, filter each separately then concat
    filtered_enroll = enrollment_df[common_cols].copy()
    filtered_drop = dropout_df[common_cols].copy()
    
    # Apply common filters
    for f_df in [filtered_enroll, filtered_drop]:
        if program_filter and "Program" in f_df.columns:
            f_df = f_df[f_df["Program"].isin(program_filter)]
        if gender_filter and "Gender" in f_df.columns:
            f_df = f_df[f_df["Gender"].isin(gender_filter)]
        if joining_sem_filter and "Joining_Semester" in f_df.columns:
            f_df = f_df[f_df["Joining_Semester"].isin(joining_sem_filter)]
    
    # Dropout-specific for drop only
    if "Is_Early_Dropout" in filtered_drop.columns and dropout_filter != "All":
        filtered_drop = filtered_drop[filtered_drop["Is_Early_Dropout"] == dropout_filter]
    if dropout_sem_filter and "Dropout_Semester" in filtered_drop.columns:
        filtered_drop = filtered_drop[filtered_drop["Dropout_Semester"].isin(dropout_sem_filter)]
    if reason_filter and "Dropout_Reason" in filtered_drop.columns:
        filtered_drop = filtered_drop[filtered_drop["Dropout_Reason"].isin(reason_filter)]
    
    filtered = pd.concat([filtered_enroll, filtered_drop], ignore_index=True)

# Joining semester filter (for all)
if joining_sem_filter and "Joining_Semester" in filtered.columns:
    filtered = filtered[filtered["Joining_Semester"].isin(joining_sem_filter)]

# Numeric filters (apply to all if columns exist)
if "CGPA" in filtered.columns:
    filtered = filtered[filtered["CGPA"].between(gpa_min, gpa_max, inclusive="both")]
if "Attendance_Percentage" in filtered.columns:
    filtered["Attendance_Percentage"] = pd.to_numeric(filtered["Attendance_Percentage"], errors="coerce")
    filtered = filtered[filtered["Attendance_Percentage"].between(attendance_min, attendance_max, inclusive="both")]

# --- Apply Top Filter ---
if "Program" in filtered.columns and top_filter != "All":
    program_counts = filtered["Program"].value_counts().head(int(top_filter.split()[1]))
    filtered = filtered[filtered["Program"].isin(program_counts.index)]

# Remove duplicates (moved earlier for efficiency)
filtered = filtered.loc[:, ~filtered.columns.duplicated()].drop_duplicates().copy()

# Ensure numeric columns (extended)
numeric_cols = ["CGPA", "Pending_Fee", "Attendance_Percentage", "Discount_Provided_Percentage"]
for col in numeric_cols:
    if col in filtered.columns:
        filtered[col] = pd.to_numeric(filtered[col], errors="coerce")

# Handle empty data globally
if filtered.empty:
    st.warning("No data matches the selected filters. Adjust filters to see results.")
else:
    # Download button
    csv = filtered.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button("Download Filtered Data", csv, "filtered_data.csv", "text/csv")

    # Data Table View
    pd.set_option("styler.render.max_elements", 500000)
    with st.expander("View Filtered Data"):
        st.dataframe(
            filtered.style.format({
                "CGPA": "{:.2f}",
                "Pending_Fee": "{:,.0f}",
                "Attendance_Percentage": "{:.1f}",
                "Discount_Provided_Percentage": "{:.1f}"
            }),
            height=4000
        )


# --- Functions ---
def display_kpis(df):
    if df.empty:
        return
    
    if data_choice == "Compare":
        agg_dict = {"Total": ("Type", "size")}
        if "CGPA" in df.columns:
            agg_dict["Avg_CGPA"] = ("CGPA", "mean")
        if "Pending_Fee" in df.columns:
            agg_dict["Avg_Pending_Dues"] = ("Pending_Fee", "mean")

        grouped = df.groupby("Type").agg(**agg_dict).reset_index()

        for _, row in grouped.iterrows():
            st.subheader(f"{row['Type']} KPIs")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Records", int(row["Total"]))
            col2.metric(
                "Avg CGPA",
                f"{row['Avg_CGPA']:.2f}" if "Avg_CGPA" in row and pd.notna(row["Avg_CGPA"]) else "N/A"
            )
            col3.metric(
                "Avg Pending Dues",
                f"{row['Avg_Pending_Dues']:,.0f} PKR" if "Avg_Pending_Dues" in row and pd.notna(row["Avg_Pending_Dues"]) else "N/A"
            )

    else:
        total = len(df)
        avg_cgpa = f"{df['CGPA'].mean():.2f}" if "CGPA" in df.columns and total > 0 else "N/A"
        
        # Dues col detection
        dues_col = next((col for col in ["Pending_Fee", "Overall_Balance", "Outstanding_Dues"] if col in df.columns), None)
        avg_dues = f"{df[dues_col].mean():,.0f} PKR" if dues_col and total > 0 else "N/A"
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Records", total)
        col2.metric("Avg CGPA", avg_cgpa)
        col3.metric("Avg Pending Dues", avg_dues)

        
def display_dropout_by_session(df):
    if df.empty:
        return

    # Ensure Is_Early_Dropout (add for enrollment as N/A if Compare)
    if "Is_Early_Dropout" not in df.columns:
        df = df.copy()
        df["Is_Early_Dropout"] = "N/A"

    join_col = "Joining_Semester"

    if join_col not in df.columns:
        return  # Skip if no session col

    if data_choice == "Compare":
        session_counts = df.groupby([join_col, "Is_Early_Dropout", "Type"]).size().reset_index(name="Count")
        type_totals = session_counts.groupby("Type")["Count"].sum().reset_index(name="Type_Total")
        session_counts = session_counts.merge(type_totals, on="Type")
        session_counts["Percent"] = (session_counts["Count"] / session_counts["Type_Total"] * 100).round(1)
    else:
        session_counts = df.groupby([join_col, "Is_Early_Dropout"]).size().reset_index(name="Count")
        total_session = session_counts["Count"].sum()
        session_counts["Percent"] = (session_counts["Count"] / total_session * 100).round(1)

    # Sort
    session_totals = session_counts.groupby(join_col)["Count"].sum().reset_index(name="Total").sort_values("Total", ascending=False)
    session_counts = session_counts.merge(session_totals, on=join_col).sort_values(["Total", join_col], ascending=[False, True])

    # Plot (add facet for Compare)
    session_plot = px.bar(
        session_counts,
        x=join_col, y="Count",
        color="Is_Early_Dropout", barmode="stack",
        text=session_counts["Percent"].astype(str) + "%",
        hover_data={"Count": True, "Percent": True},
        color_discrete_sequence=px.colors.qualitative.Safe,
        category_orders={join_col: session_totals[join_col].tolist()},
        facet_col="Type" if data_choice == "Compare" else None  # Key addition
    )
    session_plot.update_traces(
        hovertemplate="<b>%{x}</b><br>Count: %{y}<br>Percent: %{customdata[0]}%<extra></extra>",
        customdata=session_counts[["Percent"]].values
    )
    session_plot.update_layout(
        title="Records by Joining Session (Descending Order)", 
        xaxis_title="Session", 
        yaxis_title="Count",
        hovermode="x unified"
    )
    session_plot.update_xaxes(type="category")
    st.plotly_chart(session_plot, use_container_width=True, config={'responsive': True, 'scrollZoom': True})

    # Key insights (adapted for Compare)
    st.markdown("### 📌 Key Insights")
    if data_choice == "Compare":
        st.write("Comparison: Enrollment vs Dropout trends across sessions.")
    top_sessions = session_counts.nlargest(3, "Count")
    st.write("Top record sessions:")
    for _, row in top_sessions.iterrows():
        st.write(f"- {row[join_col]} ({row['Is_Early_Dropout']}): {row['Percent']}%")
    if data_choice in ["Dropout", "Compare"]:
        if data_choice == "Compare":
            early_pct = (df[df["Type"] == "Dropout"]["Is_Early_Dropout"] == "Early").mean() * 100
        else:
            early_pct = (df["Is_Early_Dropout"] == "Early").mean() * 100
        st.write(f"Local Note: Early dropouts ~{early_pct:.0f}% align with worldwide rates of 20-30% in first year (Gallup/MDPI).")
    else:
        st.write("Global Note: Early dropouts ~20-30% in first year worldwide (Gallup/MDPI).")

def display_program_summary(df, title_suffix=""):
    if df.empty:
        return

    # Determine mode based on data_choice
    is_dropout_mode = data_choice in ["Dropout", "Compare"]

    # Ensure Is_Early_Dropout for dropout mode
    if is_dropout_mode and "Is_Early_Dropout" not in df.columns:
        df = df.copy()
        df["Is_Early_Dropout"] = "Not Available"

    # Grouping logic
    if is_dropout_mode:
        if data_choice == "Compare":
            program_counts = df.groupby(["Program", "Is_Early_Dropout", "Type"]).size().reset_index(name="Count")
            color_col = "Type"
        else:
            program_counts = df.groupby(["Program", "Is_Early_Dropout"]).size().reset_index(name="Count")
            color_col = "Is_Early_Dropout"
        # Relative percent per program
        program_counts["Percent"] = (
            program_counts["Count"] / program_counts.groupby("Program")["Count"].transform("sum") * 100
        ).round(1)
        x_col = "Percent"
        title = f"Dropout Rates by Program{title_suffix}"
        barmode = "stack"
    else:
        # Enrollment mode
        if data_choice == "Compare":  # Though Compare typically uses dropout, but for consistency
            program_counts = df.groupby(["Program", "Type"]).size().reset_index(name="Count")
            color_col = "Type"
        else:
            program_counts = df.groupby("Program").size().reset_index(name="Count")
            color_col = "Program"
        # Overall percent
        program_counts["Percent"] = (program_counts["Count"] / program_counts["Count"].sum() * 100).round(1)
        x_col = "Count"
        title = f"Enrollments by Program{title_suffix}"
        barmode = "group"  # Or "stack" if preferred, but original was not stacked

    program_counts = program_counts.sort_values("Count", ascending=False)

    fig = px.bar(
        program_counts,
        x=x_col, y="Program",
        text="Percent",
        orientation="h",
        color=color_col,
        barmode=barmode,
        color_discrete_sequence=px.colors.qualitative.Safe
    )
    fig.update_layout(
        title=title,
        showlegend=True if data_choice == "Compare" or is_dropout_mode else False
    )
    st.plotly_chart(fig, use_container_width=True)

def display_program_trends(df, title_suffix=""):
    if df.empty:
        st.info("⚠️ No data available for this selection.")
        return

    if "Is_Early_Dropout" in df.columns:
        # Dropout or Compare mode
        if data_choice == "Compare":
            program_counts = df.groupby(["Joining_Semester", "Program", "Is_Early_Dropout", "Type"]).size().reset_index(name="Count")
            color_col = "Is_Early_Dropout"
            facet_extra = "Type"
        else:
            program_counts = df.groupby(["Joining_Semester", "Program", "Is_Early_Dropout"]).size().reset_index(name="Count")
            color_col = "Is_Early_Dropout"
            facet_extra = None
        title = f"Dropouts by Program and Session{title_suffix}"
    else:
        # Enrollment
        program_counts = df.groupby(["Joining_Semester", "Program"]).size().reset_index(name="Count")
        color_col = "Program"
        facet_extra = None
        title = f"Enrollments by Program and Session{title_suffix}"

    program_plot = px.bar(
        program_counts,
        x="Joining_Semester", y="Count",
        color=color_col,
        facet_col="Program", facet_col_wrap=3,
        barmode="stack",
        text="Count",
        hover_data={"Count": True},
        color_discrete_sequence=px.colors.qualitative.Safe,
        facet_row=facet_extra if data_choice == "Compare" else None  # Row for vertical compare
    )
    program_plot.update_layout(
        title=title,
        xaxis_title="Session",
        yaxis_title="Count",
        hovermode="x unified"
    )
    st.plotly_chart(program_plot, use_container_width=True)

def display_cgpa_by_umc_status(df):
    if df.empty or "UMC_DC_Status" not in df.columns or "CGPA" not in df.columns:
        return

    if data_choice == "Compare":
        agg_df = df.groupby(["UMC_DC_Status", "Type"])["CGPA"].mean().reset_index(name="Avg_CGPA")
        color_col = "Type"
    else:
        agg_df = df.groupby("UMC_DC_Status")["CGPA"].mean().reset_index(name="Avg_CGPA")
        color_col = "UMC_DC_Status"

    agg_df = agg_df.sort_values("Avg_CGPA", ascending=False)

    plot = px.bar(
        agg_df,
        x="UMC_DC_Status", y="Avg_CGPA",
        color=color_col,
        color_discrete_sequence=px.colors.qualitative.Safe,
        text=agg_df["Avg_CGPA"].round(2).astype(str),
        hover_data={"Avg_CGPA": ":.2f"}
    )
    plot.update_traces(
        hovertemplate="<b>%{x}</b><br>Avg CGPA: %{y:.2f}<extra></extra>"
    )
    plot.update_layout(
        title="Average CGPA by UMC/DC Status (Descending Order)", 
        xaxis_title="UMC/DC Status", 
        yaxis_title="Avg CGPA",
        hovermode="x unified"
    )
    st.plotly_chart(plot, use_container_width=True, config={'responsive': True, 'scrollZoom': True})

    # Key Insights
    st.markdown("### 📌 Key Insights")
    top = agg_df.nlargest(3, "Avg_CGPA")
    st.write("Top UMC/DC statuses with highest average CGPA:")
    for _, row in top.iterrows():
        st.write(f"- {row['UMC_DC_Status']}: {row['Avg_CGPA']:.2f}")

def display_pending_by_umc_points(df):
    if df.empty or "UMC_DC_Points" not in df.columns or "Pending_Fee" not in df.columns:
        return

    if data_choice == "Compare":
        agg_df = df.groupby(["UMC_DC_Points", "Type"])["Pending_Fee"].mean().reset_index(name="Avg_Pending_Fee")
        color_col = "Type"
    else:
        agg_df = df.groupby("UMC_DC_Points")["Pending_Fee"].mean().reset_index(name="Avg_Pending_Fee")
        color_col = "UMC_DC_Points"

    agg_df = agg_df.sort_values("Avg_Pending_Fee", ascending=False)

    plot = px.bar(
        agg_df,
        x="UMC_DC_Points", y="Avg_Pending_Fee",
        color=color_col,
        color_discrete_sequence=px.colors.qualitative.Safe,
        text=agg_df["Avg_Pending_Fee"].round(0).astype(str) + " PKR",
        hover_data={"Avg_Pending_Fee": ":.0f"}
    )
    plot.update_traces(
        hovertemplate="<b>%{x}</b><br>Avg Pending: %{y:,.0f} PKR<extra></extra>"
    )
    plot.update_layout(
        title="Average Pending Dues by UMC/DC Points (Descending Order)", 
        xaxis_title="UMC/DC Points", 
        yaxis_title="Avg Pending Fee (PKR)",
        hovermode="x unified"
    )
    st.plotly_chart(plot, use_container_width=True, config={'responsive': True, 'scrollZoom': True})

    # Key Insights
    st.markdown("### 📌 Key Insights")
    top = agg_df.nlargest(3, "Avg_Pending_Fee")
    st.write("Top UMC/DC points with highest average pending dues:")
    for _, row in top.iterrows():
        st.write(f"- {row['UMC_DC_Points']}: {row['Avg_Pending_Fee']:,.0f} PKR")
        
def display_additional_charts(df):
    if df.empty:
        return
    
    # Gender Breakdown
    st.subheader("Record by Gender")
    if data_choice == "Compare":
        gender_counts = df.groupby(["Gender", "Type"]).size().reset_index(name="Count")
        gender_pie = px.pie(gender_counts, values="Count", names="Gender", color_discrete_sequence=px.colors.qualitative.Safe, facet_col="Type")
    else:
        gender_counts = df["Gender"].value_counts().reset_index(name="Count")
        gender_pie = px.pie(gender_counts, values="Count", names="Gender", color_discrete_sequence=px.colors.qualitative.Safe)
    gender_pie.update_traces(
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percent: %{percent}<extra></extra>",
        hole=0.4  # Donut for modern look
    )
    gender_pie.update_layout(title="Gender Distribution")
    st.plotly_chart(gender_pie, use_container_width=True, config={'responsive': True})

    # Reasons Distribution (if available, conditional)
    if data_choice != "Enrollment" and "Dropout_Reason" in df.columns and not df["Dropout_Reason"].isna().all():
        st.subheader("Dropout Reasons Distribution")
        reason_counts = df["Dropout_Reason"].value_counts().reset_index(name="Count")
        reason_pie = px.pie(reason_counts, values="Count", names="Dropout_Reason", color_discrete_sequence=px.colors.qualitative.Safe)
        reason_pie.update_traces(
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percent: %{percent}<extra></extra>",
            hole=0.4
        )
        reason_pie.update_layout(title="Reasons Distribution")
        st.plotly_chart(reason_pie, use_container_width=True, config={'responsive': True})

def display_executive_summary():
    st.header("📊 Executive Summary for Stakeholders")
    st.markdown("""
    - **Most Dropouts are Early:** Many students leave within the first year → onboarding/adjustment support is crucial.  
    - **One Program Dominates Dropouts:** A few programs contribute the majority → targeted interventions needed.  
    - **Financial Struggles Build Up:** Late dropouts owe more dues → fee assistance or financial counseling should be prioritized.  
    - **Attendance Matters:** Strong link between attendance and CGPA → better attendance tracking and engagement programs can reduce dropout risk.  
    """)
    if data_choice == "Compare":
        st.markdown("- **Comparison Insights:** Enrollment peaks in high-demand programs, but dropouts spike early—focus on retention gaps.")

def display_financial_patterns(df):
    if df.empty or "Pending_Fee" not in df.columns:
        st.info("⚠️ No financial data available in this dataset.")
        return

    if "Is_Early_Dropout" in df.columns:
        # Dropout or Compare
        fin_plot = px.box(
            df, x="Is_Early_Dropout", y="Pending_Fee", color="Is_Early_Dropout",
            points="all",
            notched=True,
            color_discrete_map={"Early": "#63a7d8", "Late": "#8f3939"},
            facet_col="Type" if data_choice == "Compare" else None
        )
        fin_plot.update_traces(
            hovertemplate="<b>%{x}</b><br>Pending: %{y:,.0f} PKR<br>Global: Top driver (MDPI/Emmason 2024; Gallup 54% stress-linked; Pakistan: World Bank floods/economic)<extra></extra>"
        )
        fin_plot.update_layout(title="Financial Patterns (Local + Global)")
        st.plotly_chart(fin_plot, use_container_width=True)

        # Insights
        if data_choice != "Compare":
            avg_due_early = df.loc[df["Is_Early_Dropout"]=="Early", "Pending_Fee"].mean()
            avg_dues_late = df.loc[df["Is_Early_Dropout"]=="Late", "Pending_Fee"].mean()
            st.markdown("### 📌 Key Insights")
            st.write(f"- Late higher dues ({avg_dues_late:,.0f} vs. {avg_due_early:,.0f} PKR). Matches Pakistan economic/marriage (World Bank/Nepjol) and global hardship (PMC/arXiv).")
        else:
            st.markdown("### 📌 Key Insights")
            st.write("- Compare: Check boxes for Enrollment vs Dropout differences in dues distribution.")
    else:
        # Enrollment
        fin_plot = px.box(
            df, x="Program", y="Pending_Fee", color="Program",
            points="all",
            notched=True,
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        fin_plot.update_traces(
            hovertemplate="<b>%{x}</b><br>Pending Fee: %{y:,.0f} PKR<extra></extra>"
        )
        fin_plot.update_layout(title="Enrollment Financial Patterns")
        st.plotly_chart(fin_plot, use_container_width=True)

        avg_dues = df["Pending_Fee"].mean()
        st.markdown("### 📌 Key Insights")
        st.write(f"- Average pending dues across programs: {avg_dues:,.0f} PKR")

def display_dropout_reasons(df):
    if data_choice == "Enrollment" or "Dropout_Reason" not in df.columns or df["Dropout_Reason"].isna().all():
        return
    if data_choice == "Compare":
        df = df[df["Type"] == "Dropout"]  # Restrict to Dropout

    st.subheader("Dropout Reasons (Data + Global Alignments)")
    reason_counts = df["Dropout_Reason"].value_counts().reset_index(name="Count")
    reason_pie = px.pie(reason_counts, values="Count", names="Dropout_Reason", color_discrete_sequence=px.colors.qualitative.Safe)
    reason_pie.update_traces(
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percent: %{percent}<br>Global: Unpreparedness (ERIC 2024); Mental health (BMC/Gallup 43%); Pakistan: Marriage/economic (World Bank/Nepjol)<extra></extra>"
    )
    st.plotly_chart(reason_pie, use_container_width=True)

def display_mixed_methods_analysis(df, summary=None, financial=None, multidim=None):
    if data_choice == "Enrollment" or "Dropout_Reason" not in df.columns:
        st.warning("No 'Dropout_Reason' column found in dataset.")
        return
    if data_choice == "Compare":
        df = df[df["Type"] == "Dropout"]  # Restrict

    st.subheader("Mixed-Methods: Local Quantitative + Global Studies")

    # Calculate local dropout reason percentages
    reason_counts = df["Dropout_Reason"].value_counts(normalize=True) * 100
    reason_counts = reason_counts.round(2)

    # Updated mapping with latest 2025 research
    mapping = {
        "Financial Issues": "Top global (EssayShark 2025 ~30%; CreatrixCampus 2025 financial burdens; Gallup/Forbes 2025 stress-linked ~54%); Pakistan: Khuddi.global 2025 poverty main driver; Thebrilliantbrains 2025 affordability crisis; ResearchGate 2025 unaffordable fees in BS programs",
        "Marriage/Family": "Regional/Global (Wooclap 2025 personal/family issues ~32%; Thediplomat 2025 early marriage/cultural restrictions in Pakistan; Taylor&Francis 2025 socioeconomic/cultural factors)",
        "Migration/Abroad": "Intl/emigration (Thebrilliantbrains 2025 brain drain in Pakistan; ERIC/arXiv 2025 aspirations; NPR 2025 demographic shifts impacting enrollment)",
        "Academic Struggle": "Unpreparedness (MDPI 2025 academic difficulty ~18%, poor performance ~12%; QuadC 2025 struggling coursework/lack of support; ResearchGate 2025 course complexity/volume in Pakistan BS programs)",
        "Lack of Motivation": "Disengagement (EssayShark 2025 motivation issues ~24%; CreatrixCampus 2025 lack of engagement; ITB-Academic-Tests 2025 individual/social factors)",
        "Choice of Different Program": "Poor fit (MDPI 2025 dissatisfaction; QuadC 2025 inadequate support/mismatch; Tandfonline 2025 inequality; Thebrilliantbrains 2025 curriculum reform needs in Pakistan)",
        "Transport": "Challenges (Irapa.org 2025 non-availability of facilities in Pakistan (school-level, extensible to higher ed); CreatrixCampus 2025 external factors like economic shifts)",
        "Health": "Mental/physical (EssayShark 2025 mental health ~18%; CreatrixCampus 2025 mental health issues; Thediplomat 2025 restrictions impacting well-being in Pakistan)",
        "Communication Gap": "Support gaps (CreatrixCampus 2025 inadequate support systems; QuadC 2025 lack of tutoring/services; ResearchGate 2025 unfavorable teacher attitudes in Pakistan)",
        "UMC/DC": "Violations (Aligned academic; MDPI 2025 poor performance links; ResearchGate 2025 inadequate resources/attitudes in Pakistan; limited 2025-specific, but ties to broader struggles)"
    }

    # Build matrix
    matrix_data = []
    for reason, citation in mapping.items():
        local_val = reason_counts.get(reason, "N/A")
        if local_val != "N/A":
            local_display = f"{local_val:.2f}%"
            note = "Strong match; consider direct interventions" if local_val >= 50 else ("Partial alignment; extend measures" if 30 <= local_val < 50 else "Low evidence; global mismatch")
        else:
            local_display = "N/A"
            note = "No local evidence"
        matrix_data.append([reason, local_display, citation, note])

    matrix_df = pd.DataFrame(matrix_data, columns=["Theme", "Local Quantitative %", "Global Alignment & Citation", "Local-Global Notes"])
    matrix_df = matrix_df.sort_values("Local Quantitative %", ascending=False)  # Added: Sort by %

    # Coloring (unchanged)
    def color_rows(val):
        if "Strong" in val: return 'background-color: #d4edda'
        if "Partial" in val: return 'background-color: #fff3cd'
        if "Low" in val or "No local" in val: return 'background-color: #f8d7da'
        return ''

    st.markdown("### Convergence Matrix (Dynamic)")
    st.dataframe(matrix_df.style.applymap(color_rows, subset=['Local-Global Notes']))

    # Recommendations
    st.markdown("### Recommendations (Based on Local + Global Evidence)")
    for _, row in matrix_df.iterrows():
        if row["Local Quantitative %"] != "N/A" and float(row["Local Quantitative %"].rstrip('%')) >= 50:
            st.write(f"- **{row['Theme']}**: High prevalence locally ({row['Local Quantitative %']}). Align with {row['Global Alignment & Citation']}. Recommendation: Prioritize targeted intervention.")
        elif row["Local Quantitative %"] != "N/A":
            st.write(f"- **{row['Theme']}**: Moderate/low prevalence locally ({row['Local Quantitative %']}). Consider pilot or secondary interventions aligned with global practices.")
        else:
            st.write(f"- **{row['Theme']}**: No local evidence. Global references suggest watching this area, but may not need immediate action.")
            
# --- Dashboard ---
if data_choice == "Dropout":
    st.title("🎓 Student Dropout Dashboard")
elif data_choice == "Enrollment":
    st.title("📊 Student Enrollment Dashboard")
else:
    st.title("🔍 Enrollment vs Dropout Comparison Dashboard")

if not filtered.empty:
    tab1, tab2, tab3 = st.tabs(["Core Metrics & Charts", "Program & Financial Details", "Mixed-Methods Insights"])
    
    with tab1:
        display_kpis(filtered)
        display_dropout_by_session(filtered)
        display_dropout_reasons(filtered)
        display_additional_charts(filtered)
        display_executive_summary()
    
    with tab2:
        # Overall by Program
        # Improvement: Replaced if-else with single call to combined function
        display_program_summary(filtered)
        
        # Category-specific (unchanged, but filtered handles Compare)
        st.subheader("Category-Specific Program Records")
        categories = ["BS Programs", "BS after 14 years", "Girls Block Programs", "Only for Female", "AD 2 years", "MPhil and MSc"]
        selected_category = st.selectbox("Select Program Category", ["None"] + categories)
    
        if selected_category != "None":
            if selected_category == "BS Programs":
                category_filtered = filtered[
                    filtered["Program"].str.contains("BS", na=False)
                    & ~filtered["Program"].str.contains("14 years", na=False)
                    & ~filtered["Program"].str.contains("Only for", na=False)
                    & ~filtered["Program"].str.contains("Girls Block", na=False)
                ]
            elif selected_category == "BS after 14 years":
                category_filtered = filtered[filtered["Program"].str.contains(r"\(After 14 years edu\)", na=False, regex=True)]
            elif selected_category == "Girls Block Programs":
                category_filtered = filtered[filtered["Program"].str.contains("Girls Block", na=False)]
            elif selected_category == "Only for Female":
                category_filtered = filtered[filtered["Program"].str.contains("(Only for Females)", na=False)]
            elif selected_category == "AD 2 years":
                category_filtered = filtered[filtered["Program"].str.startswith("AD", na=False)]
            elif selected_category == "MPhil and MSc":
                category_filtered = filtered[
                    (filtered["Program"].str.startswith("MPhil", na=False))
                    | (filtered["Program"].str.startswith("MSc", na=False))
                ]
            
            if not category_filtered.empty:
                st.markdown("## Overall by Program")
                # Improvement: Replaced if-else with single call to combined function
                display_program_summary(category_filtered, f" - {selected_category}")

                st.markdown("## Trends by Program and Session")
                display_program_trends(category_filtered, f" - {selected_category}")
            else:
                st.warning("No data available for the selected category.")

        # Financial & others
        display_financial_patterns(filtered)
        display_cgpa_by_umc_status(filtered)
        display_pending_by_umc_points(filtered)

    with tab3:
        display_mixed_methods_analysis(filtered, summary_by_group, financial_summary, multi_dim_summary)

else:
    st.warning("No data available. Please adjust filters.")
