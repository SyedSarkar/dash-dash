import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import io  # For download
import openpyxl  # For Excel loading

# Set page config for better UX
st.set_page_config(layout="wide", page_title="Student Dropout Dashboard")

# Custom CSS for aesthetics
st.markdown("""
    <style>
    .stSidebar .stRadio > label { font-size: 16px; }
    .stMetric { font-size: 18px; }
    .stTabs [data-testid="stMarkdownContainer"] { font-size: 16px; font-weight: bold; }
    .stDataFrame { border: 1px solid #ddd; border-radius: 5px; }
    </style>
""", unsafe_allow_html=True)

# --- Data Validation ---
required_cols = ["PROGRAM", "STD_GENDER", "CGPA", "PRESENT_PERCENTAGE", "PENDING_AMT", "JOINING_SEMESTER", "DROPOUT_SEMESTER", "Is_Early_Dropout"]  # Adjusted for Excel headers

# --- Load Data ---
@st.cache_data
def load_data():
    # Load CSVs (as backups/aggregates)
    dropouts = pd.read_csv("all_dropouts.csv")
    summary = pd.read_csv("summary_by_group.csv")
    financial = pd.read_csv("financial_summary.csv")
    multidim = pd.read_csv("multi_dim_summary.csv")
    
    # Load and parse full Excel (primary source)
    excel_file = "dropout dteal combine sheet.xlsx"
    wb = openpyxl.load_workbook(excel_file, data_only=True)
    all_sheets = []
    for sheet_name in wb.sheetnames:
        df_sheet = pd.read_excel(excel_file, sheet_name=sheet_name)
        df_sheet['Sheet'] = sheet_name  # Add sheet for context
        all_sheets.append(df_sheet)
    excel_df = pd.concat(all_sheets, ignore_index=True)
    excel_df = excel_df.fillna("N/A")
    # Derive Is_Early_Dropout if missing (proxy: Dropout_Delay_Semesters <=1)
    if "Is_Early_Dropout" not in excel_df.columns:
        excel_df["Is_Early_Dropout"] = np.where(excel_df.get("Dropout_Delay_Semesters", 0) <= 1, "Early", "Late")
    
    # Validate
    missing_cols = [col for col in required_cols if col not in excel_df.columns]
    if missing_cols:
        st.error(f"Missing columns in Excel: {', '.join(missing_cols)}")
        st.stop()
    
    return dropouts, summary, financial, multidim, excel_df

all_dropouts, summary_by_group, financial_summary, multi_dim_summary, excel_df = load_data()

# --- Sidebar Filters (use Excel as primary) ---
st.sidebar.title("Filters")
st.sidebar.info("Early: Dropout in first year; Late: After first year.")

#Basic Filters
top_filter = st.sidebar.radio("Show Top Programs", ["All", "Top 3", "Top 5", "Top 10"])
program_filter = st.sidebar.multiselect("Select Program", excel_df["PROGRAM"].dropna().unique())
gender_filter = st.sidebar.multiselect("Select Gender", excel_df["STD_GENDER"].dropna().unique())
dropout_filter = st.sidebar.radio("Dropout Timing", ["All", "Early", "Late"])

#session filters
joining_sem_filter = st.sidebar.multiselect("Joining Semester", excel_df["JOINING_SEMESTER"].dropna().unique())
dropout_sem_filter = st.sidebar.multiselect("Dropout Semester", excel_df["DROPOUT_SEMESTER"].dropna().unique())

#Advanced Filterss
st.sidebar.markdown("### Advanced Filters")
gpa_min, gpa_max = st.sidebar.slider(
    "CGPA Range",
    min_value=float(excel_df["CGPA"].min()),
    max_value=float(excel_df["CGPA"].max()),
    value=(float(excel_df["CGPA"].min()), float(excel_df["CGPA"].max())),
    step=0.1
)

attendance_min, attendance_max = st.sidebar.slider(
    "Attendance % Range",
    min_value=0.0,
    max_value=100.0,
    value=(0.0, 100.0),
    step=1.0
)

#optional reason filters
reason_filter = []
if "Dropout_Reason" in excel_df.columns:
    reason_filter = st.sidebar.multiselect("Dropout Reason", excel_df["Dropout_Reason"].dropna().unique())

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

# --- Functions (updated with international integrations) ---
def display_kpis(df):
    total_dropouts = len(df)
    avg_cgpa = f"{df['CGPA'].mean():.2f}" if total_dropouts > 0 else "N/A"
    avg_dues = f"{df.get('PENDING_AMT', df['Pending_Overall']).mean():,.0f} PKR" if total_dropouts > 0 else "N/A"
    
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

    # Aggregate total per session for sorting
    session_totals = session_counts.groupby("Joining_Semester")["Count"].sum().reset_index(name="Total")
    session_totals = session_totals.sort_values("Total", ascending=False)
    session_counts = session_counts.merge(session_totals, on="Joining_Semester")
    session_counts = session_counts.sort_values(["Total", "Joining_Semester"], ascending=[False, True])

    session_plot = px.bar(
        session_counts,
        x="Joining_Semester", y="Count",
        color="Is_Early_Dropout", barmode="stack",
        text=session_counts["Percent"].astype(str) + "%",
        hover_data={"Count": True, "Percent": True},
        color_discrete_sequence=px.colors.qualitative.Safe,  # Colorblind-friendly
        category_orders={"Joining_Semester": session_totals["Joining_Semester"].tolist()}
    )
    session_plot.update_traces(
        hovertemplate="<b>%{x}</b><br>Count: %{y}<br>Percent: %{customdata[0]}%<extra></extra>",
        customdata=session_counts[["Percent"]].values
    )
    session_plot.update_layout(
        title="Dropouts by Joining Session (Descending Order)", 
        xaxis_title="Session", 
        yaxis_title="Count",
        hovermode="x unified"  # Unified hover for better interaction
    )
    session_plot.update_xaxes(type="category")
    st.plotly_chart(session_plot, use_container_width=True, config={'responsive': True, 'scrollZoom': True})

    # Dynamic Key Insights
    st.markdown("### 📌 Key Insights")
    top_sessions = session_counts.nlargest(3, "Count")
    st.write("Top dropout sessions:")
    for _, row in top_sessions.iterrows():
        st.write(f"- {row['Joining_Semester']} ({row['Is_Early_Dropout']}): {row['Percent']}%")
    st.write("Global Note: Early dropouts ~82% align with worldwide rates of 20-30% in first year (Gallup/MDPI).")

def display_dropout_by_program(df, title_suffix=""):
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

    # Aggregate total per program for sorting
    program_totals = program_counts.groupby("Program")["Count"].sum().reset_index(name="Total")
    program_totals = program_totals.sort_values("Total", ascending=False)
    program_counts = program_counts.merge(program_totals, on="Program")
    program_counts = program_counts.sort_values(["Total", "Program"], ascending=[False, True])

    # Custom colors per program (colorblind-friendly)
    unique_programs = program_counts["Program"].unique()
    color_map = {prog: color for prog, color in zip(unique_programs, px.colors.qualitative.Safe[:len(unique_programs)])}

    program_plot = px.bar(
        program_counts,
        x="Count", y="Program", color="Program",
        orientation="h", barmode="stack", text="Label",
        hover_data={"Count": True, "Percent": True},
        color_discrete_map=color_map,
        category_orders={"Program": program_totals["Program"].tolist()}
    )
    program_plot.update_traces(
        textposition="inside",
        hovertemplate="<b>%{y}</b><br>Count: %{x}<br>Percent: %{customdata[0]}%<extra></extra>",
        customdata=program_counts[["Percent"]].values
    )
    program_plot.update_layout(
        title=f"Dropouts by Program (Descending Order){title_suffix}", 
        xaxis_title="Count", 
        yaxis_title="Program",
        hovermode="y unified"
    )
    st.plotly_chart(program_plot, use_container_width=True, config={'responsive': True, 'scrollZoom': True})

    # Dynamic Key Insights
    st.markdown("### 📌 Key Insights")
    top_programs = program_counts.nlargest(3, "Count")
    st.write("Top dropout programs:")
    for _, row in top_programs.iterrows():
        st.write(f"- {row['Program']} ({row['Is_Early_Dropout']}): {row['Percent']}%")
        
def display_cgpa_by_umc_status(df):
    if df.empty or "UMC_DC_Status" not in df.columns or "CGPA" not in df.columns:
        return
    agg_df = df.groupby("UMC_DC_Status")["CGPA"].mean().reset_index(name="Avg_CGPA")
    agg_df = agg_df.sort_values("Avg_CGPA", ascending=False)

    plot = px.bar(
        agg_df,
        x="UMC_DC_Status", y="Avg_CGPA",
        color="UMC_DC_Status",
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
    if df.empty or "UMC_DC_Points" not in df.columns or "Pending_Overall" not in df.columns:
        return
    # Bin UMC_DC_Points if continuous, but assuming it's categorical or low range
    agg_df = df.groupby("UMC_DC_Points")["Pending_Overall"].mean().reset_index(name="Avg_Pending_Overall")
    agg_df = agg_df.sort_values("Avg_Pending_Overall", ascending=False)

    plot = px.bar(
        agg_df,
        x="UMC_DC_Points", y="Avg_Pending_Overall",
        color="UMC_DC_Points",
        color_discrete_sequence=px.colors.qualitative.Safe,
        text=agg_df["Avg_Pending_Overall"].round(0).astype(str) + " PKR",
        hover_data={"Avg_Pending_Overall": ":.0f"}
    )
    plot.update_traces(
        hovertemplate="<b>%{x}</b><br>Avg Pending: %{y:,.0f} PKR<extra></extra>"
    )
    plot.update_layout(
        title="Average Pending Dues by UMC/DC Points (Descending Order)", 
        xaxis_title="UMC/DC Points", 
        yaxis_title="Avg Pending Overall (PKR)",
        hovermode="x unified"
    )
    st.plotly_chart(plot, use_container_width=True, config={'responsive': True, 'scrollZoom': True})

    # Key Insights
    st.markdown("### 📌 Key Insights")
    top = agg_df.nlargest(3, "Avg_Pending_Overall")
    st.write("Top UMC/DC points with highest average pending dues:")
    for _, row in top.iterrows():
        st.write(f"- {row['UMC_DC_Points']}: {row['Avg_Pending_Overall']:,.0f} PKR")
        
def display_additional_charts(df):
    if df.empty:
        return
    
    # Gender Breakdown
    st.subheader("Dropout by Gender")
    gender_counts = df["Gender"].value_counts()
    gender_pie = px.pie(gender_counts, values=gender_counts.values, names=gender_counts.index, color_discrete_sequence=px.colors.qualitative.Safe)
    gender_pie.update_traces(
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percent: %{percent}<extra></extra>"
    )
    gender_pie.update_layout(title="Gender Distribution")
    st.plotly_chart(gender_pie, use_container_width=True, config={'responsive': True})

    
    # Reasons Distribution (if available)
    if "Dropout_Reason" in df.columns and not df["Dropout_Reason"].isna().all():
        st.subheader("Dropout Reasons Distribution")
        reason_counts = df["Dropout_Reason"].value_counts()
        reason_pie = px.pie(reason_counts, values=reason_counts.values, names=reason_counts.index, color_discrete_sequence=px.colors.qualitative.Safe)
        reason_pie.update_traces(
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percent: %{percent}<extra></extra>"
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

def display_financial_patterns(df):
    if df.empty:
        return
    fin_plot = px.box(
        df, x="Is_Early_Dropout", y=df.get("PENDING_AMT", "Pending_Overall"), color="Is_Early_Dropout",
        color_discrete_sequence=px.colors.qualitative.Safe,
        points="all",
        notched=True
    )
    fin_plot.update_traces(
        hovertemplate="<b>%{x}</b><br>Pending: %{y:,.0f} PKR<br>Global: Top driver (MDPI/Emmason 2024; Gallup 54% stress-linked; Pakistan: World Bank floods/economic)<extra></extra>"
    )
    fin_plot.update_layout(title="Financial Patterns (Local + Global)")
    st.plotly_chart(fin_plot, use_container_width=True)

    avg_due_early = df.loc[df["Is_Early_Dropout"]=="Early", df.get("PENDING_AMT", "Pending_Overall")].mean()
    avg_due_late = df.loc[df["Is_Early_Dropout"]=="Late", df.get("PENDING_AMT", "Pending_Overall")].mean()
    st.markdown("### 📌 Key Insights")
    st.write(f"- Late higher dues ({avg_due_late:,.0f} vs. {avg_due_early:,.0f} PKR). Matches Pakistan economic/marriage (Nepjol/journals.irapa) and global hardship (PMC/arXiv).")

def display_dropout_reasons(df):
    if "Dropout_Reason" in df.columns and not df["Dropout_Reason"].isna().all():
        st.subheader("Dropout Reasons (Excel + Global Alignments)")
        reason_counts = df["Dropout_Reason"].value_counts()
        reason_pie = px.pie(reason_counts, values=reason_counts.values, names=reason_counts.index, color_discrete_sequence=px.colors.qualitative.Safe)
        reason_pie.update_traces(
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percent: %{percent}<br>Global: Unpreparedness (ERIC 2024); Mental health (BMC/Gallup 43%); Pakistan: Marriage/economic (World Bank/Nepjol)<extra></extra>"
        )
        st.plotly_chart(reason_pie, use_container_width=True)

def display_mixed_methods_analysis(df, summary, financial, multidim, excel):
    st.subheader("Mixed-Methods: Local Quantitative + Global Studies")
    st.info("Our data (e.g., 63% financial, 73% academic) echoes global trends (20-30% rates per MDPI/Gallup 2024; Pakistan: 50%+ from World Bank/Colombiaone analogs) with local nuances (emigration Nepjol, marriage journals.irapa).")

    st.markdown("### Convergence Matrix (w/ Global Alignments)")
    matrix_data = {
        "Theme": ["Financial Issues", "Marriage/Family", "Migration/Abroad", "Academic Struggle", "Lack of Motivation", "Wrong Program", "Transport", "Health", "Communication Gap", "UMC/DC"],
        "Local Quantitative %": [63.64, 54.55, "N/A", 72.73, 54.55, "N/A", "Limited", "N/A", "N/A", 9.09],
        "Global Alignment & Citation": [
            "Top global (MDPI/Emmason 2024 ~50%; Gallup 54% stress; Pakistan: World Bank economic/floods)",
            "Regional (Nepjol marriage; journals.irapa cultural norms)",
            "Intl/emigration (ERIC/arXiv aspirations; Nepal Nepjol; Reddit visa-waiting)",
            "Unpreparedness (PMC/web:5 demands ~30%; South Korea journals.sagepub)",
            "Disengagement (Gallup false expectations; ijip.in lack discipline)",
            "Poor fit (PMC/Polo; labmanager.com diverse reasons; tandfonline inequality)",
            "Challenges (web:1 Pakistan transport/harassment; Nepal Nepjol)",
            "Mental/physical (BMC/Gallup 43%; frontiersin front stress)",
            "Support gaps (Higher Ed alerts; che.de early measures)",
            "Violations (Aligned academic; limited, iated.org factors)"
        ],
        "Local-Global Notes": ["Strong match; Extend micro-grants (Georgia State 30% reduction pyrrhicpress)", "Gender-specific; Family engagement (doc)", "Anecdotal visa; Track (Nepjol)", "High alignment; FYE (Wikipedia/UT Austin)", "Proxy attendance; Mentoring (Brown/Albion 20% boost)", "Extension; Guidance (DePaul)", "Partial; Flexible (EDMO)", "Infer; Screening (doc/Ready Ed)", "Contradiction; AI chatbots (NYU)", "Low; Alerts (Higher Ed)"]
    }
    matrix_df = pd.DataFrame(matrix_data)
    def color_rows(val):
        if "Strong" in val or "High" in val: return 'background-color: #d4edda'
        if "Extension" in val: return 'background-color: #fff3cd'
        if "Contradiction" in val: return 'background-color: #f8d7da'
        return ''
    st.dataframe(matrix_df.style.applymap(color_rows, subset=['Local-Global Notes']))

    # Joint Displays (enhanced with citations)

    # Enhanced Recommendations with Success Stories
    st.markdown("### Recommendations (Local Qualitative + Global Successes)")
    recs = [
        "Financial: Scholarships/awareness; Micro-grants success (Georgia State 30% reduction Higher Ed/pyrrhicpress; Pakistan World Bank aid)",
        "Academic: Senior faculty 1st sem; FYE programs (UT Austin/Wikipedia; Albion ASP 72% retention studentclearinghouse)",
        "Motivation: Mentorship/honorship; Peer networks (Brown/Albion 20% boost arXiv; expandinglearning expanded opportunities)",
        "Migration/Fit: Career guidance; Purpose networks (doc; DePaul Future Forward Higher Ed)",
        "Family/Health: Engagement/screening; AI chatbots (NYU/EDMO; BMC mental health interventions)",
        "Communication: Alerts/mobile; CRM check-ins (Higher Ed; dropoutprevention 15 strategies)"
    ]
    for rec in recs:
        st.write(f"- {rec}")

# --- Dashboard ---
st.title("🎓 Student Dropout Dashboard")

if not filtered.empty:
    tab1, tab2, tab3 = st.tabs(["Core Metrics & Charts", "Program & Financial Details", "Mixed-Methods Insights"])
    
    with tab1:
        display_kpis(filtered)
        display_dropout_by_session(filtered)
        display_dropout_reasons(filtered)
        display_additional_charts(filtered)
        display_executive_summary()
    
    with tab2:
        display_dropout_by_program(filtered)
        
        # Category-specific view under the program plot
        st.subheader("Category-Specific Program Dropouts")
        categories = ["BS Programs", "BS after 14 years", "Girls Block Programs", "Only for Female", "AD 2 years", "MPhil and MSc"]
        selected_category = st.selectbox("Select Program Category", ["None"] + categories)
    
    if selected_category != "None":
        if selected_category == "BS Programs":
            category_filtered = filtered[filtered["Program"].str.contains("BS", na=False)]
        elif selected_category == "BS after 14 years":
            category_filtered = filtered[filtered["Program"].str.contains(r"\(After 14 years edu\)", na=False, regex=True)]
        elif selected_category == "Girls Block Programs":
            category_filtered = filtered[filtered["Program"].str.contains("Girls Block", na=False)]
        elif selected_category == "Only for Female":
            category_filtered = filtered[filtered["Program"].str.contains("(Only for Females)", na=False)]
        elif selected_category == "AD 2 years":
            category_filtered = filtered[filtered["Program"].str.startswith("AD", na=False)]
        elif selected_category == "MPhil and MSc":
            category_filtered = filtered[(filtered["Program"].str.startswith("MPhil", na=False)) | (filtered["Program"].str.startswith("MSc", na=False))]
        
        if not category_filtered.empty:
            display_dropout_by_program(category_filtered, f" - {selected_category}")
        else:
            st.warning("No data available for the selected category.")
          
        display_financial_patterns(filtered)
        #display_cgpa_by_umc_status(filtered)
        display_pending_by_umc_points(filtered)
        
    with tab3:
        display_mixed_methods_analysis(filtered, summary_by_group, financial_summary, multi_dim_summary, excel_df)
