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
required_cols = ["program", "gender", "cgpa", "attendance_percentage", "pending_fee", "joining_semester", "dropout_semester"]  # Standardized to uppercase

@st.cache_data
def load_data(enrollment_file, dropout_file):
    import re
    import numpy as np
    import pandas as pd

    # Load enrollment
    enrollment_df = pd.read_csv(enrollment_file)
    dropout_df = pd.read_csv(dropout_file)

    # ✅ Standardize all columns to lowercase + underscores
    enrollment_df.columns = (
        enrollment_df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )
    dropout_df.columns = (
        dropout_df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    # Add type column
    enrollment_df["type"] = "enrollment"
    dropout_df["type"] = "dropout"

    # Convert common numeric fields safely
    numeric_cols = [
        "cgpa", "attendance_percentage", "pending_fee",
        "discount_provided_percentage", "completed_crhr",
        "required_crhr", "remaining_crhr"
    ]
    for df in [enrollment_df, dropout_df]:
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fill NAs consistently
    enrollment_df = enrollment_df.fillna("N/A")
    dropout_df = dropout_df.fillna("N/A")

    # Ensure correct dtype for string ops
    for col in ["joining_semester", "dropout_semester"]:
        if col in dropout_df.columns:
            dropout_df[col] = dropout_df[col].astype(str)

    # --- Normalize column names between files ---
    for df in [enrollment_df, dropout_df]:
        # Ensure consistent 'joining_semester' column
        if "joining_semester" not in df.columns:
            if "session" in df.columns:
                df["joining_semester"] = df["session"]
            else:
                df["joining_semester"] = "Unknown"

    # --- Define helper functions for early dropout logic ---
    def parse_semester(sem):
        """Extracts ('spring' or 'fall', year as int) from a semester string like 'Fall 2023'."""
        if not isinstance(sem, str):
            return None, None
        match = re.search(r'(spring|fall)\s*(\d{4})', sem.lower())
        if match:
            return match.group(1), int(match.group(2))
        return None, None

    def classify_early_dropout(row):
        join_sem, join_year = parse_semester(row.get("joining_semester"))
        drop_sem, drop_year = parse_semester(row.get("dropout_semester"))

        if join_sem is None or drop_sem is None or join_year is None or drop_year is None:
            return "Unknown"

        year_diff = drop_year - join_year

        # Early dropout = within 1 academic year
        if year_diff == 0:
            return "Early"
        elif year_diff == 1:
            if join_sem == "fall" and drop_sem in ["spring", "fall"]:
                return "Early"
            if join_sem == "spring" and drop_sem == "spring":
                return "Early"
        return "Late"

    # --- Apply the early/late dropout logic ---
    dropout_df["is_early_dropout"] = dropout_df.apply(classify_early_dropout, axis=1)

    # --- Generate summary tables dynamically ---
    # Safely pick the best available pending fee column
    pending_col = None
    for cand in ["pending_fee", "pending_amt", "pending_amt_first_semester", "outstanding_dues"]:
        if cand in dropout_df.columns:
            pending_col = cand
            break

    # Ensure numeric columns (extended)
    numeric_cols = ["cgpa", "pending_fee", "attendance_percentage", "discount_provided_percentage"]
    for col in numeric_cols:
        if col in dropout_df.columns:
            dropout_df[col] = pd.to_numeric(dropout_df[col], errors="coerce")

    # Build summary tables
    summary_by_group = dropout_df.groupby("program", dropna=False).agg(
        total_dropouts=("roll_no", "count"),
        avg_cgpa=("cgpa", "mean"),
        **({"avg_pending_fee": (pending_col, "mean")} if pending_col else {})
    ).reset_index()

    financial_summary = dropout_df.groupby("program", dropna=False).agg(
        total_paid=("paid_amt", "sum"),
        total_pending=("pending_amt", "sum"),
    ).reset_index()

    multi_dim_summary = dropout_df.groupby(["program", "joining_semester"], dropna=False).agg(
        total_students=("roll_no", "count"),
        avg_cgpa=("cgpa", "mean"),
    ).reset_index()

    return enrollment_df, dropout_df, summary_by_group, financial_summary, multi_dim_summary

# --- Sidebar Uploads ---
st.sidebar.header("📂 Upload Data Files")
enrollment_file = st.sidebar.file_uploader("Upload Enrollment Data (.csv)", type=["csv"])
dropout_file = st.sidebar.file_uploader("Upload Dropout Data (.csv)", type=["csv"])

# --- Load Data Dynamically ---
if enrollment_file and dropout_file:
    enrollment_df, dropout_df, summary_by_group, financial_summary, multi_dim_summary = load_data(
        enrollment_file, dropout_file
    )
    st.success("✅ Files uploaded and loaded successfully!")
else:
    st.warning("⚠️ Please upload both Enrollment and Dropout CSV files to continue.")
    st.stop()


# Toggle option (added "Compare")
data_choice = st.radio("Select Data type", ["Enrollment", "Dropout", "Compare"], horizontal=True)

# Pick the dataframe based on user choice (enhanced for Compare)
common_cols = list(set(enrollment_df.columns) & set(dropout_df.columns))  # Auto-detect common for robustness
if data_choice == "Enrollment":
    df = enrollment_df
elif data_choice == "Dropout":
    df = dropout_df
else:  # Compare
    df = pd.concat([enrollment_df[common_cols], dropout_df[common_cols]], ignore_index=True)
    st.info("Compare mode: Showing combined data with 'type' for differentiation. Some dropout-specific filters are ignored.")

# --- Sidebar Filters ---
st.sidebar.title("Filters")
st.sidebar.info("Early: Dropout in first year; Late: After first year.")

# Basic Filters (unchanged)
top_filter = st.sidebar.radio("Show Top programs", ["All", "Top 3", "Top 5", "Top 10"])
program_filter = st.sidebar.multiselect("Select program", pd.concat([enrollment_df["program"], dropout_df["program"]]).dropna().unique())  # Use combined unique for Compare
gender_filter = st.sidebar.multiselect("Select gender", pd.concat([enrollment_df["gender"], dropout_df["gender"]]).dropna().unique())

# Dropout-specific filters (conditional)
if data_choice != "Enrollment":
    dropout_filter = st.sidebar.radio("Dropout Timing", ["All", "Early", "Late"])

# Session filters (use joining_semester for consistency)
joining_sem_filter = st.sidebar.multiselect("Joining Semester", pd.concat([enrollment_df.get("session", pd.Series([])), dropout_df.get("joining_semester", pd.Series([]))]).dropna().unique())
if data_choice != "Enrollment":
    dropout_sem_filter = st.sidebar.multiselect("Dropout Semester", dropout_df.get("dropout_semester", pd.Series([])).dropna().unique())

# Advanced Filters
st.sidebar.markdown("### Advanced Filters")
gpa_min, gpa_max = st.sidebar.slider(
    "cgpa Range",
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
    reason_filter = st.sidebar.multiselect("Dropout Reason", dropout_df.get("dropout_reason", pd.Series([])).dropna().unique())
    
# --- Apply Filters ---
filtered = df.copy()

# program filter
if program_filter and "program" in filtered.columns:
    filtered = filtered[filtered["program"].isin(program_filter)]

# gender filter
if gender_filter and "gender" in filtered.columns:
    filtered = filtered[filtered["gender"].isin(gender_filter)]

# --- Dropout-specific logic ---
if data_choice == "Dropout":
    # Early/Late dropout filter
    if "is_early_dropout" in filtered.columns and dropout_filter != "All":
        filtered = filtered[filtered["is_early_dropout"].str.lower() == dropout_filter.lower()]
        
    # Dropout semester filter
    if dropout_sem_filter and "dropout_semester" in filtered.columns:
        filtered = filtered[filtered["dropout_semester"].isin(dropout_sem_filter)]
        
    # Dropout reason filter
    if reason_filter and "dropout_reason" in filtered.columns:
        filtered = filtered[filtered["dropout_reason"].isin(reason_filter)]
        
    # Joining semester filter — apply as well (important!)
    if joining_sem_filter and "joining_semester" in filtered.columns:
        filtered = filtered[filtered["joining_semester"].isin(joining_sem_filter)]

# --- Enrollment logic ---
elif data_choice == "Enrollment":
    if joining_sem_filter and "joining_semester" in filtered.columns:
        filtered = filtered[filtered["joining_semester"].isin(joining_sem_filter)]

# --- Compare mode logic ---
elif data_choice == "Compare":
    filtered_enroll = enrollment_df.copy()
    filtered_drop = dropout_df.copy()

    # Apply program/gender filters
    for f_df in [filtered_enroll, filtered_drop]:
        if program_filter and "program" in f_df.columns:
            f_df = f_df[f_df["program"].isin(program_filter)]
        if gender_filter and "gender" in f_df.columns:
            f_df = f_df[f_df["gender"].isin(gender_filter)]

    # Apply session filters specifically
    if joining_sem_filter:
        if "joining_semester" in filtered_enroll.columns:
            filtered_enroll = filtered_enroll[filtered_enroll["joining_semester"].isin(joining_sem_filter)]
        if "joining_semester" in filtered_drop.columns:
            filtered_drop = filtered_drop[filtered_drop["joining_semester"].isin(joining_sem_filter)]
    if dropout_sem_filter and "dropout_semester" in filtered_drop.columns:
        filtered_drop = filtered_drop[filtered_drop["dropout_semester"].isin(dropout_sem_filter)]

    # Dropout-specific filters
    if "is_early_dropout" in filtered_drop.columns and dropout_filter != "All":
        filtered_drop = filtered_drop[filtered_drop["is_early_dropout"].str.lower() == dropout_filter.lower()]
    if reason_filter and "dropout_reason" in filtered_drop.columns:
        filtered_drop = filtered_drop[filtered_drop["dropout_reason"].isin(reason_filter)]

    filtered = pd.concat([filtered_enroll, filtered_drop], ignore_index=True)

# --- Numeric filters ---
if "cgpa" in filtered.columns:
    filtered = filtered[filtered["cgpa"].between(gpa_min, gpa_max, inclusive="both")]

if "attendance_percentage" in filtered.columns:
    filtered["attendance_percentage"] = pd.to_numeric(filtered["attendance_percentage"], errors="coerce")
    filtered = filtered[filtered["attendance_percentage"].between(attendance_min, attendance_max, inclusive="both")]

# --- Top N programs ---
if "program" in filtered.columns and top_filter != "All":
    top_n = int(top_filter.split()[1])
    program_counts = filtered["program"].value_counts().head(top_n)
    filtered = filtered[filtered["program"].isin(program_counts.index)]


# Remove duplicates (moved earlier for efficiency)
filtered = filtered.loc[:, ~filtered.columns.duplicated()].drop_duplicates().copy()

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
                "cgpa": "{:.2f}",
                "pending_fee": "{:,.0f}",
                "attendance_percentage": "{:.1f}",
                "Discount_Provided_percentage": "{:.1f}"
            }),
            height=4000
        )


# --- Functions ---
def display_kpis(df):
    if df.empty:
        return
    if data_choice == "Compare":
        agg_dict = {"Total": ("type", "size")}
        if "cgpa" in df.columns:
            agg_dict["avg_cgpa"] = ("cgpa", "mean")
        if "pending_fee" in df.columns:
            agg_dict["avg_pending_dues"] = ("pending_fee", "mean")

        grouped = df.groupby("type").agg(**agg_dict).reset_index()

        for _, row in grouped.iterrows():
            st.subheader(f"{row['type']} KPIs")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Records", int(row["Total"]))
            col2.metric(
                "Avg cgpa",
                f"{row['avg_cgpa']:.2f}" if "avg_cgpa" in row and pd.notna(row["avg_cgpa"]) else "N/A"
            )
            col3.metric(
                "Avg Pending Dues",
                f"{row['avg_pending_dues']:,.0f} PKR" if "avg_pending_dues" in row and pd.notna(row["avg_pending_dues"]) else "N/A"
            )

    else:
        total = len(df)
        avg_cgpa = f"{df['cgpa'].mean():.2f}" if "cgpa" in df.columns and total > 0 else "N/A"
        
        # Dues col detection
        dues_col = next((col for col in ["pending_fee", "overall_balance", "outstanding_dues"] if col in df.columns), None)
        avg_dues = f"{df[dues_col].mean():,.0f} PKR" if dues_col and total > 0 else "N/A"
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Records", total)
        col2.metric("Avg cgpa", avg_cgpa)
        col3.metric("Avg Pending Dues", avg_dues)

        
def display_dropout_by_session(df):
    if df.empty:
        return

    # --- Normalize column names ---
    if "is_early_dropout" not in df.columns:
        df = df.copy()
        df["is_early_dropout"] = "N/A"

    join_col = "joining_semester"
    drop_col = "dropout_semester"

    if join_col not in df.columns:
        return  # skip if no semester info

    # --- Detect what to plot on x-axis ---
    # If a joining_semester filter is applied, plot dropout_semesters for those students
    if len(joining_sem_filter) > 0 and drop_col in df.columns and data_choice in ["Dropout", "Compare"]:
        group_col = drop_col
        chart_title = f"Dropout Semesters for Students Who Joined in {', '.join(joining_sem_filter)}"
    else:
        group_col = join_col
        chart_title = "Records by Joining Session (Descending Order)"

    # --- Compute grouped counts ---
    group_fields = [group_col, "is_early_dropout"]
    if data_choice == "Compare" and "type" in df.columns:
        group_fields.append("type")

    session_counts = df.groupby(group_fields, dropna=False).size().reset_index(name="count")
    total_session = session_counts["count"].sum()
    session_counts["percent"] = (session_counts["count"] / total_session * 100).round(1)

    # --- Sort sessions by total descending ---
    session_totals = session_counts.groupby(group_col)["count"].sum().reset_index(name="Total")
    session_totals = session_totals.sort_values("Total", ascending=False)
    session_counts = session_counts.merge(session_totals, on=group_col).sort_values(["Total", group_col], ascending=[False, True])

    # --- Build plot ---
    session_plot = px.bar(
        session_counts,
        x=group_col,
        y="count",
        color="is_early_dropout",
        barmode="stack",
        text=session_counts["percent"].astype(str) + "%",
        hover_data={"count": True, "percent": True},
        color_discrete_sequence=px.colors.qualitative.Safe,
        category_orders={group_col: session_totals[group_col].tolist()},
        facet_col="type" if data_choice == "Compare" else None
    )

    session_plot.update_traces(
        hovertemplate="<b>%{x}</b><br>count: %{y}<br>percent: %{customdata[0]}%<extra></extra>",
        customdata=session_counts[["percent"]].values
    )
    session_plot.update_layout(
        title=chart_title,
        xaxis_title="Session",
        yaxis_title="Count",
        hovermode="x unified"
    )
    session_plot.update_xaxes(type="category")
    st.plotly_chart(session_plot, use_container_width=True, config={'responsive': True, 'scrollZoom': True})

    # --- Key insights section ---
    st.markdown("### 📌 Key Insights")

    if data_choice == "Compare":
        st.write("Comparison: Enrollment vs Dropout trends across sessions.")

    top_sessions = session_counts.nlargest(3, "count")
    st.write("Top record sessions:")
    for _, row in top_sessions.iterrows():
        st.write(f"- {row[group_col]} ({row['is_early_dropout']}): {row['percent']}%")

    if data_choice in ["Dropout", "Compare"]:
        if data_choice == "Compare":
            early_pct = (df[df["type"] == "Dropout"]["is_early_dropout"] == "Early").mean() * 100
        else:
            early_pct = (df["is_early_dropout"] == "Early").mean() * 100
        st.write(f"Local Note: Early dropouts ~{early_pct:.0f}% align with worldwide rates of 20–30% in first year (Gallup/MDPI).")
    else:
        st.write("Global Note: Early dropouts ~20–30% in first year worldwide (Gallup/MDPI).")


def display_program_summary(df, title_suffix=""):
    if df.empty:
        return

    # Determine mode based on data_choice
    is_dropout_mode = data_choice in ["Dropout", "Compare"]

    # Ensure is_early_dropout for dropout mode
    if is_dropout_mode and "is_early_dropout" not in df.columns:
        df = df.copy()
        df["is_early_dropout"] = "Not Available"

    # Grouping logic
    if is_dropout_mode:
        if data_choice == "Compare":
            program_counts = df.groupby(["program", "is_early_dropout", "type"]).size().reset_index(name="count")
            color_col = "type"
        else:
            program_counts = df.groupby(["program", "is_early_dropout"]).size().reset_index(name="count")
            color_col = "is_early_dropout"
        # Relative percent per program
        program_counts["percent"] = (
            program_counts["count"] / program_counts.groupby("program")["count"].transform("sum") * 100
        ).round(1)
        x_col = "percent"
        title = f"Dropout Rates by program{title_suffix}"
        barmode = "stack"
    else:
        # Enrollment mode
        if data_choice == "Compare":  # Though Compare typically uses dropout, but for consistency
            program_counts = df.groupby(["program", "type"]).size().reset_index(name="count")
            color_col = "type"
        else:
            program_counts = df.groupby("program").size().reset_index(name="count")
            color_col = "program"
        # Overall percent
        program_counts["percent"] = (program_counts["count"] / program_counts["count"].sum() * 100).round(1)
        x_col = "count"
        title = f"Enrollments by program{title_suffix}"
        barmode = "group"  # Or "stack" if preferred, but original was not stacked

    program_counts = program_counts.sort_values("count", ascending=False)

    fig = px.bar(
        program_counts,
        x=x_col, y="program",
        text="percent",
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

    if "is_early_dropout" in df.columns:
        # Dropout or Compare mode
        if data_choice == "Compare":
            program_counts = df.groupby(["joining_semester", "program", "is_early_dropout", "type"]).size().reset_index(name="count")
            color_col = "is_early_dropout"
            facet_extra = "type"
        else:
            program_counts = df.groupby(["joining_semester", "program", "is_early_dropout"]).size().reset_index(name="count")
            color_col = "is_early_dropout"
            facet_extra = None
        title = f"Dropouts by program and Session{title_suffix}"
    else:
        # Enrollment
        program_counts = df.groupby(["joining_semester", "program"]).size().reset_index(name="count")
        color_col = "program"
        facet_extra = None
        title = f"Enrollments by program and Session{title_suffix}"

    program_plot = px.bar(
        program_counts,
        x="joining_semester", y="count",
        color=color_col,
        facet_col="program", facet_col_wrap=3,
        barmode="stack",
        text="count",
        hover_data={"count": True},
        color_discrete_sequence=px.colors.qualitative.Safe,
        facet_row=facet_extra if data_choice == "Compare" else None  # Row for vertical compare
    )
    program_plot.update_layout(
        title=title,
        xaxis_title="Session",
        yaxis_title="count",
        hovermode="x unified"
    )
    st.plotly_chart(program_plot, use_container_width=True)

def display_cgpa_by_umc_status(df):
    if df.empty or "UMC_DC_Status" not in df.columns or "cgpa" not in df.columns:
        return

    if data_choice == "Compare":
        agg_df = df.groupby(["UMC_DC_Status", "type"])["cgpa"].mean().reset_index(name="avg_cgpa")
        color_col = "type"
    else:
        agg_df = df.groupby("UMC_DC_Status")["cgpa"].mean().reset_index(name="avg_cgpa")
        color_col = "UMC_DC_Status"

    agg_df = agg_df.sort_values("avg_cgpa", ascending=False)

    plot = px.bar(
        agg_df,
        x="UMC_DC_Status", y="avg_cgpa",
        color=color_col,
        color_discrete_sequence=px.colors.qualitative.Safe,
        text=agg_df["avg_cgpa"].round(2).astype(str),
        hover_data={"avg_cgpa": ":.2f"}
    )
    plot.update_traces(
        hovertemplate="<b>%{x}</b><br>Avg cgpa: %{y:.2f}<extra></extra>"
    )
    plot.update_layout(
        title="Average cgpa by UMC/DC Status (Descending Order)", 
        xaxis_title="UMC/DC Status", 
        yaxis_title="Avg cgpa",
        hovermode="x unified"
    )
    st.plotly_chart(plot, use_container_width=True, config={'responsive': True, 'scrollZoom': True})

    # Key Insights
    st.markdown("### 📌 Key Insights")
    top = agg_df.nlargest(3, "avg_cgpa")
    st.write("Top UMC/DC statuses with highest average cgpa:")
    for _, row in top.iterrows():
        st.write(f"- {row['UMC_DC_Status']}: {row['avg_cgpa']:.2f}")

def display_pending_by_umc_points(df):
    if df.empty or "UMC_DC_Points" not in df.columns or "pending_fee" not in df.columns:
        return

    if data_choice == "Compare":
        agg_df = df.groupby(["UMC_DC_Points", "type"])["pending_fee"].mean().reset_index(name="Avg_pending_fee")
        color_col = "type"
    else:
        agg_df = df.groupby("UMC_DC_Points")["pending_fee"].mean().reset_index(name="Avg_pending_fee")
        color_col = "UMC_DC_Points"

    agg_df = agg_df.sort_values("Avg_pending_fee", ascending=False)

    plot = px.bar(
        agg_df,
        x="UMC_DC_Points", y="Avg_pending_fee",
        color=color_col,
        color_discrete_sequence=px.colors.qualitative.Safe,
        text=agg_df["Avg_pending_fee"].round(0).astype(str) + " PKR",
        hover_data={"Avg_pending_fee": ":.0f"}
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
    top = agg_df.nlargest(3, "Avg_pending_fee")
    st.write("Top UMC/DC points with highest average pending dues:")
    for _, row in top.iterrows():
        st.write(f"- {row['UMC_DC_Points']}: {row['Avg_pending_fee']:,.0f} PKR")
        
def display_gender_chart(df):
    if df.empty:
        return
    
    # gender Breakdown
    st.subheader("Record by gender")
    if data_choice == "Compare":
        gender_counts = df.groupby(["gender", "type"]).size().reset_index(name="count")
        gender_pie = px.pie(gender_counts, values="count", names="gender", color_discrete_sequence=px.colors.qualitative.Safe, facet_col="type")
    else:
        gender_counts = df["gender"].value_counts().reset_index(name="count")
        gender_pie = px.pie(gender_counts, values="count", names="gender", color_discrete_sequence=px.colors.qualitative.Safe)
    gender_pie.update_traces(
        hovertemplate="<b>%{label}</b><br>count: %{value}<br>percent: %{percent}<extra></extra>",
        hole=0.4  # Donut for modern look
    )
    gender_pie.update_layout(title="gender Distribution")
    st.plotly_chart(gender_pie, use_container_width=True, config={'responsive': True})

def display_executive_summary():
    st.header("📊 Executive Summary for Stakeholders")
    st.markdown("""
    - **Most Dropouts are Early:** Many students leave within the first year → onboarding/adjustment support is crucial.  
    - **One program Dominates Dropouts:** A few programs contribute the majority → targeted interventions needed.  
    - **Financial Struggles Build Up:** Late dropouts owe more dues → fee assistance or financial counseling should be prioritized.  
    - **Attendance Matters:** Strong link between attendance and cgpa → better attendance tracking and engagement programs can reduce dropout risk.  
    """)
    if data_choice == "Compare":
        st.markdown("- **Comparison Insights:** Enrollment peaks in high-demand programs, but dropouts spike early—focus on retention gaps.")

def display_financial_patterns(df):
    if df.empty or "pending_amt" not in df.columns:
        st.info("⚠️ No financial data available in this dataset.")
        return

    if "is_early_dropout" in df.columns:
        # Dropout or Compare
        fin_plot = px.box(
            df, x="is_early_dropout", y="pending_amt", color="is_early_dropout",
            points="all",
            notched=True,
            color_discrete_map={"Early": "#63a7d8", "Late": "#8f3939"},
            facet_col="type" if data_choice == "Compare" else None
        )
        fin_plot.update_traces(
            hovertemplate="<b>%{x}</b><br>Pending: %{y:,.0f} PKR<br>Global: Top driver (MDPI/Emmason 2024; Gallup 54% stress-linked; Pakistan: World Bank floods/economic)<extra></extra>"
        )
        fin_plot.update_layout(title="Financial Patterns (Local + Global)")
        st.plotly_chart(fin_plot, use_container_width=True)

        # Insights
        if data_choice != "Compare":
            avg_due_early = df.loc[df["is_early_dropout"]=="Early", "pending_amt"].mean()
            avg_dues_late = df.loc[df["is_early_dropout"]=="Late", "pending_amt"].mean()
            st.markdown("### 📌 Key Insights")
            st.write(f"- Late higher dues ({avg_dues_late:,.0f} vs. {avg_due_early:,.0f} PKR). Matches Pakistan economic/marriage (World Bank/Nepjol) and global hardship (PMC/arXiv).")
        else:
            st.markdown("### 📌 Key Insights")
            st.write("- Compare: Check boxes for Enrollment vs Dropout differences in dues distribution.")
    else:
        # Enrollment
        fin_plot = px.box(
            df, x="program", y="pending_amt", color="program",
            points="all",
            notched=True,
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        fin_plot.update_traces(
            hovertemplate="<b>%{x}</b><br>Pending Fee: %{y:,.0f} PKR<extra></extra>"
        )
        fin_plot.update_layout(title="Enrollment Financial Patterns")
        st.plotly_chart(fin_plot, use_container_width=True)

        avg_dues = df["pending_amt"].mean()
        st.markdown("### 📌 Key Insights")
        st.write(f"- Average pending dues across programs: {avg_dues:,.0f} PKR")

def display_dropout_reasons(df):
    if data_choice == "Enrollment" or "dropout_reason" not in df.columns or df["dropout_reason"].isna().all():
        return

    if data_choice == "Compare" and "type" in df.columns:
        df = df[df["type"].str.lower() == "dropout"]

    st.subheader("Dropout Reasons (Data + Global Alignments)")

    # --- Aggregate ---
    reason_counts = df["dropout_reason"].value_counts(dropna=False).reset_index()
    reason_counts.columns = ["dropout_reason", "count"]
    reason_counts["count"] = pd.to_numeric(reason_counts["count"], errors="coerce").fillna(0)

    # --- Add dynamic reason filter ---
    all_reasons = reason_counts["dropout_reason"].dropna().unique().tolist()
    selected_reasons = st.multiselect("Select Reasons to Display", all_reasons, default=all_reasons)

    reason_counts = reason_counts[reason_counts["dropout_reason"].isin(selected_reasons)]

    total = reason_counts["count"].sum()
    if total == 0:
        st.warning("No valid dropout reason data found.")
        return

    reason_counts["percent"] = (reason_counts["count"].astype(float) / total * 100).round(1)

    # --- Pie Chart ---
    reason_pie = px.pie(
        reason_counts,
        values="count",
        names="dropout_reason",
        color_discrete_sequence=px.colors.qualitative.Safe,
    )
    reason_pie.update_traces(
        textinfo="percent+label",
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percent: %{percent}<extra></extra>",
    )
    st.plotly_chart(reason_pie, use_container_width=True)

    # --- Dynamic Key Insights ---
    st.markdown("### 📊 Key Insights: Dropout Reasons Breakdown")
    st.write(f"Total Dropouts (selected categories): **{int(total):,}**")

    for _, row in reason_counts.iterrows():
        st.write(f"- **{row['dropout_reason']}**: {int(row['count']):,} students ({row['percent']}%)")

    st.info(
        "🌍 *Global Patterns:* Academic unpreparedness and mental health challenges account for ~40–45% "
        "of early dropouts (Gallup/BMC 2024). In South Asia, socio-economic and family factors "
        "remain major contributors (World Bank, Nepjol)."
    )
    
def display_mixed_methods_analysis(df, summary=None, financial=None, multidim=None):
    if data_choice == "Enrollment" or "dropout_reason" not in df.columns:
        st.warning("No 'dropout_reason' column found in dataset.")
        return
    if data_choice == "Compare":
        df = df[df["type"] == "Dropout"]  # Restrict

    st.subheader("Mixed-Methods: Local Quantitative + Global Studies")

    # Calculate local dropout reason percentages
    reason_counts = df["dropout_reason"].value_counts(normalize=True) * 100
    reason_counts = reason_counts.round(2)

    # Updated mapping with latest 2025 research
    mapping = {
        "Financial Issues": "Top global (EssayShark 2025 ~30%; CreatrixCampus 2025 financial burdens; Gallup/Forbes 2025 stress-linked ~54%); Pakistan: Khuddi.global 2025 poverty main driver; Thebrilliantbrains 2025 affordability crisis; ResearchGate 2025 unaffordable fees in BS programs",
        "Marriage/Family": "Regional/Global (Wooclap 2025 personal/family issues ~32%; Thediplomat 2025 early marriage/cultural restrictions in Pakistan; Taylor&Francis 2025 socioeconomic/cultural factors)",
        "Migration/Abroad": "Intl/emigration (Thebrilliantbrains 2025 brain drain in Pakistan; ERIC/arXiv 2025 aspirations; NPR 2025 demographic shifts impacting enrollment)",
        "Academic Struggle": "Unpreparedness (MDPI 2025 academic difficulty ~18%, poor performance ~12%; QuadC 2025 struggling coursework/lack of support; ResearchGate 2025 course complexity/volume in Pakistan BS programs)",
        "Lack of Motivation": "Disengagement (EssayShark 2025 motivation issues ~24%; CreatrixCampus 2025 lack of engagement; ITB-Academic-Tests 2025 individual/social factors)",
        "Choice of Different program": "Poor fit (MDPI 2025 dissatisfaction; QuadC 2025 inadequate support/mismatch; Tandfonline 2025 inequality; Thebrilliantbrains 2025 curriculum reform needs in Pakistan)",
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
    tab1, tab2, tab3 = st.tabs(["Core Metrics & Charts", "program & Financial Details", "Mixed-Methods Insights"])
    
    with tab1:
        display_kpis(filtered)
        display_dropout_by_session(filtered)
        display_dropout_reasons(filtered)
        display_gender_chart(filtered)
        display_executive_summary()
    
    with tab2:
        # Overall by program
        # Improvement: Replaced if-else with single call to combined function
        display_program_summary(filtered)
        
        # Category-specific (unchanged, but filtered handles Compare)
        st.subheader("Category-Specific program Records")
        categories = ["BS programs", "BS after 14 years", "Girls Block programs", "Only for Female", "AD 2 years", "MPhil and MSc"]
        selected_category = st.selectbox("Select program Category", ["None"] + categories)
    
        if selected_category != "None":
            if selected_category == "BS programs":
                category_filtered = filtered[
                    filtered["program"].str.contains("BS", na=False)
                    & ~filtered["program"].str.contains("14 years", na=False)
                    & ~filtered["program"].str.contains("Only for", na=False)
                    & ~filtered["program"].str.contains("Girls Block", na=False)
                ]
            elif selected_category == "BS after 14 years":
                category_filtered = filtered[filtered["program"].str.contains(r"\(After 14 years edu\)", na=False, regex=True)]
            elif selected_category == "Girls Block programs":
                category_filtered = filtered[filtered["program"].str.contains("Girls Block", na=False)]
            elif selected_category == "Only for Female":
                category_filtered = filtered[filtered["program"].str.contains("(Only for Females)", na=False)]
            elif selected_category == "AD 2 years":
                category_filtered = filtered[filtered["program"].str.startswith("AD", na=False)]
            elif selected_category == "MPhil and MSc":
                category_filtered = filtered[
                    (filtered["program"].str.startswith("MPhil", na=False))
                    | (filtered["program"].str.startswith("MSc", na=False))
                ]
            
            if not category_filtered.empty:
                st.markdown("## Overall by program")
                # Improvement: Replaced if-else with single call to combined function
                display_program_summary(category_filtered, f" - {selected_category}")

                st.markdown("## Trends by program and Session")
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
