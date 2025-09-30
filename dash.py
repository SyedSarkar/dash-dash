import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import io  # For download

# Set page config for better UX
st.set_page_config(layout="wide", page_title="Student Dropout Dashboard")

# Custom CSS for aesthetics
st.markdown("""
    <style>
    .stSidebar .stRadio > label { font-size: 16px; }
    .stMetric { font-size: 18px; }
    </style>
""", unsafe_allow_html=True)

# --- Data Validation ---
required_cols = ["Program", "Gender", "CGPA", "Present_Percentage", "Pending_Overall", "Joining_Semester", "Dropout_Semester", "Is_Early_Dropout"]

# --- Load Data ---
@st.cache_data
def load_data():
    # For real-time, replace with database connection, e.g.:
    # conn = st.experimental_connection('dropout_db', type='sql')
    # dropouts = conn.query('SELECT * FROM dropouts')
    # For now, use CSVs
    dropouts = pd.read_csv("all_dropouts.csv")
    summary = pd.read_csv("summary_by_group.csv")
    financial = pd.read_csv("financial_summary.csv")
    multidim = pd.read_csv("multi_dim_summary.csv")
    
    # Validate columns
    missing_cols = [col for col in required_cols if col not in dropouts.columns]
    if missing_cols:
        st.error(f"Missing columns in data: {', '.join(missing_cols)}")
        st.stop()
    
    return dropouts, summary, financial, multidim

all_dropouts, summary_by_group, financial_summary, multi_dim_summary = load_data()

# --- Sidebar Filters ---
st.sidebar.title("Filters")
st.sidebar.info("Early: Dropout in first year; Late: After first year.")

# Basic filters
top_filter = st.sidebar.radio("Show Top Programs", ["All", "Top 3", "Top 5", "Top 10"])
program_filter = st.sidebar.multiselect("Select Program", all_dropouts["Program"].dropna().unique())
gender_filter = st.sidebar.multiselect("Select Gender", all_dropouts["Gender"].dropna().unique())
dropout_filter = st.sidebar.radio("Dropout Timing", ["All", "Early", "Late"])

# Session/Semester filters
joining_sem_filter = st.sidebar.multiselect("Joining Semester", all_dropouts["Joining_Semester"].dropna().unique())
dropout_sem_filter = st.sidebar.multiselect("Dropout Semester", all_dropouts["Dropout_Semester"].dropna().unique())

# Advanced filters
st.sidebar.markdown("### Advanced Filters")
gpa_min, gpa_max = st.sidebar.slider(
    "CGPA Range",
    min_value=float(all_dropouts["CGPA"].min()),
    max_value=float(all_dropouts["CGPA"].max()),
    value=(float(all_dropouts["CGPA"].min()), float(all_dropouts["CGPA"].max())),
    step=0.1
)

attendance_min, attendance_max = st.sidebar.slider(
    "Attendance % Range",
    min_value=0.0,
    max_value=100.0,
    value=(0.0, 100.0),
    step=1.0
)

# Optional reason filter
reason_filter = []
if "Dropout_Reason" in all_dropouts.columns:
    reason_filter = st.sidebar.multiselect("Dropout Reason", all_dropouts["Dropout_Reason"].dropna().unique())

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

# Apply Top N Program filter
if top_filter != "All":
    n = int(top_filter.split(" ")[1])  # 3 or 5
    top_programs = filtered["Program"].value_counts().nlargest(n).index
    filtered = filtered[filtered["Program"].isin(top_programs)]

# Handle empty data globally
if filtered.empty:
    st.warning("No data matches the selected filters. Adjust filters to see results.")
else:
    # Download button
    csv = filtered.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button("Download Filtered Data", csv, "filtered_dropouts.csv", "text/csv")

    # Data Table View
    with st.expander("View Filtered Data"):
        st.dataframe(filtered.style.format({"CGPA": "{:.2f}", "Pending_Overall": "{:,.0f}", "Present_Percentage": "{:.1f}"}))

# --- Functions for Modularization ---
def display_kpis(df):
    try:
        total_dropouts = len(df)
        avg_cgpa = f"{df['CGPA'].mean():.2f}" if total_dropouts > 0 else "N/A"
        avg_dues = f"{df['Pending_Overall'].mean():,.0f} PKR" if total_dropouts > 0 else "N/A"
    except:
        avg_cgpa = "N/A"
        avg_dues = "N/A"
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Dropouts", total_dropouts)
    col2.metric("Avg CGPA", avg_cgpa)
    col3.metric("Avg Pending Dues", avg_dues)

def display_dropout_by_session(df):
    if df.empty:
        return
    session_counts = (
        df.groupby(["Joining_Semester", "Is_Early_Dropout"])
        .size()
        .reset_index(name="Count")
    )
    total_session = session_counts["Count"].sum()
    session_counts["Percent"] = (session_counts["Count"] / total_session * 100).round(1)

    session_plot = px.bar(
        session_counts,
        x="Joining_Semester", y="Count",
        color="Is_Early_Dropout", barmode="stack",
        text=session_counts["Percent"].astype(str) + "%",
        color_discrete_sequence=px.colors.qualitative.Safe  # Colorblind-friendly
    )
    session_plot.update_layout(title="Dropouts by Joining Session", xaxis_title="Session", yaxis_title="Count")
    session_plot.update_xaxes(type="category")
    st.plotly_chart(session_plot, use_container_width=True)

    # Dynamic Key Insights
    st.markdown("### 📌 Key Insights")
    top_sessions = session_counts.nlargest(3, "Count")
    st.write("Top dropout sessions:")
    for _, row in top_sessions.iterrows():
        st.write(f"- {row['Joining_Semester']} ({row['Is_Early_Dropout']}): {row['Percent']}%")

def display_dropout_by_program(df):
    if df.empty:
        return
    program_counts = (
        df.groupby(["Program", "Is_Early_Dropout"])
        .size()
        .reset_index(name="Count")
    )
    total_counts = program_counts["Count"].sum()
    program_counts["Percent"] = (program_counts["Count"] / total_counts * 100).round(1)
    program_counts["Label"] = program_counts["Count"].astype(str) + " (" + program_counts["Percent"].astype(str) + "%)"

    # Custom colors per program (colorblind-friendly)
    unique_programs = program_counts["Program"].unique()
    color_map = {prog: color for prog, color in zip(unique_programs, px.colors.qualitative.Safe[:len(unique_programs)])}

    program_plot = px.bar(
        program_counts,
        x="Count", y="Program", color="Program",
        orientation="h", barmode="stack", text="Label",
        color_discrete_map=color_map
    )
    program_plot.update_traces(textposition="inside")
    program_plot.update_layout(title="Dropouts by Program", xaxis_title="Count", yaxis_title="Program")
    st.plotly_chart(program_plot, use_container_width=True)

    # Dynamic Key Insights
    st.markdown("### 📌 Key Insights")
    top_programs = program_counts.nlargest(3, "Count")
    st.write("Top dropout programs:")
    for _, row in top_programs.iterrows():
        st.write(f"- {row['Program']} ({row['Is_Early_Dropout']}): {row['Percent']}%")

def display_financial_patterns(df):
    if df.empty:
        return
    fin_plot = px.box(
        df, x="Is_Early_Dropout", y="Pending_Overall", color="Is_Early_Dropout",
        color_discrete_sequence=px.colors.qualitative.Safe
    )
    fin_plot.update_layout(title="Financial Patterns by Dropout Timing")
    st.plotly_chart(fin_plot, use_container_width=True)

    # Key Insights
    try:
        avg_due_early = df.loc[df["Is_Early_Dropout"]=="Early", "Pending_Overall"].mean()
        avg_due_late = df.loc[df["Is_Early_Dropout"]=="Late", "Pending_Overall"].mean()
    except:
        avg_due_early = avg_due_late = 0
    st.markdown("### 📌 Key Insights")
    st.write(f"""
    - Late dropouts have **higher pending dues** on average ({avg_due_late:,.0f} PKR) compared to early dropouts ({avg_due_early:,.0f} PKR).  
    - This indicates financial stress may accumulate over time.  
    """)

import pandas as pd
import numpy as np
from scipy import stats
import plotly.express as px
import streamlit as st

def corr_with_pvalues(df):
    cols = df.columns
    n = len(cols)
    corr_matrix = np.zeros((n, n))
    p_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            r, p = stats.pearsonr(df[cols[i]].dropna(), df[cols[j]].dropna())
            corr_matrix[i, j] = r
            p_matrix[i, j] = p

    return pd.DataFrame(corr_matrix, index=cols, columns=cols), \
           pd.DataFrame(p_matrix, index=cols, columns=cols)

def significance_stars(p):
    if p < 0.001: return "***"
    elif p < 0.01: return "**"
    elif p < 0.05: return "*"
    else: return ""

def display_correlation_heatmap(df):
    if df.empty:
        st.warning("No data available for correlation heatmap.")
        return

    numeric_cols = ["CGPA", "Previous_Percentage", "Present_Percentage", 
                    "Total_Absent", "Pending_Overall", "Age_at_Join"]
    df = df[numeric_cols].dropna()

    # Correlation and significance
    corr, pvals = corr_with_pvalues(df)
    text = corr.round(2).astype(str) + pvals.applymap(significance_stars)

    # Plotly heatmap
    fig = px.imshow(
        corr.values,
        x=corr.columns,
        y=corr.index,
        color_continuous_scale="RdBu",
        aspect="auto",
        zmin=-1, zmax=1,
        text_auto=False
    )
    
def corr_with_pvalues(df):
    cols = df.columns
    n = len(cols)
    corr_matrix = np.zeros((n, n))
    p_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            # Pearson correlation (ignores NaN values)
            r, p = stats.pearsonr(df[cols[i]].dropna(), df[cols[j]].dropna())
            corr_matrix[i, j] = r
            p_matrix[i, j] = p

    return pd.DataFrame(corr_matrix, index=cols, columns=cols), \
           pd.DataFrame(p_matrix, index=cols, columns=cols)

def significance_stars(p):
    if p < 0.001: return "***"
    elif p < 0.01: return "**"
    elif p < 0.05: return "*"
    else: return ""

def interpret_corr(r):
    """Simple interpretation of correlation strength."""
    if abs(r) >= 0.8: return "very strong"
    elif abs(r) >= 0.6: return "strong"
    elif abs(r) >= 0.4: return "moderate"
    elif abs(r) >= 0.2: return "weak"
    else: return "very weak"

def display_correlation_heatmap(df):
    if df.empty:
        st.warning("No data available for correlation heatmap.")
        return

    numeric_cols = ["CGPA", "Previous_Percentage", "Present_Percentage", 
                    "Total_Absent", "Pending_Overall", "Age_at_Join"]
    df = df[numeric_cols].dropna()

    # Correlation and significance
    corr, pvals = corr_with_pvalues(df)
    text = corr.round(2).astype(str) + pvals.applymap(significance_stars)

    # Plotly heatmap
    fig = px.imshow(
        corr.values,
        x=corr.columns,
        y=corr.index,
        color_continuous_scale="RdBu",
        aspect="auto",
        zmin=-1, zmax=1,
        text_auto=False
    )

    # Add annotations with significance stars
    for i in range(len(corr)):
        for j in range(len(corr)):
            fig.add_annotation(
                text=text.iloc[i, j],
                x=corr.columns[j],
                y=corr.index[i],
                showarrow=False,
                font=dict(color="black", size=12)
            )

    fig.update_layout(title="Correlation Heatmap with Significance")
    st.plotly_chart(fig, use_container_width=True)

    # Key Insights
    corr_vals = corr.unstack()
    pval_vals = pvals.unstack()
    corr_vals = corr_vals[corr_vals < 0.9999]  # remove self-correlations

    strongest_pos = corr_vals.idxmax()
    strongest_neg = corr_vals.idxmin()

    pos_r = corr_vals[strongest_pos]
    neg_r = corr_vals[strongest_neg]
    pos_p = pval_vals[strongest_pos]
    neg_p = pval_vals[strongest_neg]

    st.markdown("### 📌 Key Insights")
    st.write(f"""
    - Strongest **positive** correlation: **{strongest_pos[0]}** and **{strongest_pos[1]}**  
      (r = {pos_r:.2f}, p = {pos_p:.3f}) → This is a **{interpret_corr(pos_r)} positive** relationship.  
      - Interpretation: As **{strongest_pos[0]}** increases, **{strongest_pos[1]}** also tends to increase.  

    - Strongest **negative** correlation: **{strongest_neg[0]}** and **{strongest_neg[1]}**  
      (r = {neg_r:.2f}, p = {neg_p:.3f}) → This is a **{interpret_corr(neg_r)} negative** relationship.  
      - Interpretation: As **{strongest_neg[0]}** increases, **{strongest_neg[1]}** tends to decrease.  

    **ℹ️ Note:**  
    - `r` (correlation coefficient) ranges from -1 to +1:  
      - +1 = perfect positive, -1 = perfect negative, 0 = no linear relation.  
    - `p` (p-value) tells if correlation is statistically significant:  
      - p < 0.05 → statistically significant (*),  
      - p < 0.01 → very significant (**),  
      - p < 0.001 → highly significant (***).  
    """)


def display_additional_charts(df):
    if df.empty:
        return
    
    # Gender Breakdown
    st.subheader("Dropout by Gender")
    gender_counts = df["Gender"].value_counts()
    gender_pie = px.pie(gender_counts, values=gender_counts.values, names=gender_counts.index, color_discrete_sequence=px.colors.qualitative.Safe)
    gender_pie.update_layout(title="Gender Distribution")
    st.plotly_chart(gender_pie, use_container_width=True)
    
    # Reasons Distribution (if available)
    if "Dropout_Reason" in df.columns and not df["Dropout_Reason"].isna().all():
        st.subheader("Dropout Reasons Distribution")
        reason_counts = df["Dropout_Reason"].value_counts()
        reason_pie = px.pie(reason_counts, values=reason_counts.values, names=reason_counts.index, color_discrete_sequence=px.colors.qualitative.Safe)
        reason_pie.update_layout(title="Reasons Distribution")
        st.plotly_chart(reason_pie, use_container_width=True)

def display_predictive_analytics(df):
    if df.empty or len(df) < 10:  # Need enough data for ML
        st.info("Insufficient data for predictive analytics.")
        return
    
    st.subheader("Predictive Analytics: Early Dropout Risk")
    
    # Features and target (predict Is_Early_Dropout as binary)
    features = ["CGPA", "Present_Percentage", "Pending_Overall"]
    target = "Is_Early_Dropout"
    df[target] = df[target].map({"Early": 1, "Late": 0})  # Binary encode
    
    X = df[features].dropna()
    y = df.loc[X.index, target]
    
    if len(X) < 2:
        st.info("Not enough valid data points for model training.")
        return
    
    # Train simple model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)
    
    # Accuracy
    acc = accuracy_score(y_test, model.predict(X_test_scaled))
    st.write(f"Model Accuracy: {acc:.2f}")
    
    # Predict risk for filtered data
    X_all_scaled = scaler.transform(df[features].fillna(0))  # Fill NaNs for demo
    df["Risk_Score"] = model.predict_proba(X_all_scaled)[:, 1]  # Probability of Early dropout
    
    # Display in table
    with st.expander("Predicted Risk Scores"):
        st.dataframe(df[["Program", "CGPA", "Present_Percentage", "Pending_Overall", "Is_Early_Dropout", "Risk_Score"]].style.format({"Risk_Score": "{:.2f}"}))
    
    # Input for hypothetical student
    st.markdown("### Predict for a Hypothetical Student")
    cgpa_input = st.number_input("CGPA", min_value=0.0, max_value=4.0, value=2.5)
    attendance_input = st.number_input("Attendance %", min_value=0.0, max_value=100.0, value=75.0)
    dues_input = st.number_input("Pending Dues (PKR)", min_value=0.0, value=10000.0)
    
    if st.button("Predict Risk"):
        input_data = scaler.transform([[cgpa_input, attendance_input, dues_input]])
        risk = model.predict_proba(input_data)[:, 1][0]
        st.write(f"Predicted Early Dropout Risk: {risk:.2f} (0-1 scale)")

def display_executive_summary():
    st.header("📊 Executive Summary for Stakeholders")
    st.markdown("""
    - **Most Dropouts are Early:** Many students leave within the first year → onboarding/adjustment support is crucial.  
    - **One Program Dominates Dropouts:** A few programs contribute the majority → targeted interventions needed.  
    - **Financial Struggles Build Up:** Late dropouts owe more dues → fee assistance or financial counseling should be prioritized.  
    - **Attendance Matters:** Strong link between attendance and CGPA → better attendance tracking and engagement programs can reduce dropout risk.  
    """)

# --- Dashboard ---
st.title("🎓 Student Dropout Dashboard")

if not filtered.empty:
    display_kpis(filtered)
    display_dropout_by_session(filtered)
    display_dropout_by_program(filtered)
    display_financial_patterns(filtered)
    display_correlation_heatmap(filtered)
    display_additional_charts(filtered)
    display_predictive_analytics(filtered)
display_executive_summary()

# Note: For Multi-Page App, save this as app.py in root, then create pages/ folder with files like overview.py (copy relevant sections), deep_dive.py (e.g., predictive). Streamlit will auto-detect.
# For Real-Time Data: Replace load_data with SQL connection as commented.
