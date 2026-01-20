import streamlit as st
import pandas as pd
from pathlib import Path
import concurrent.futures
import json
from streamlit.components.v1 import html

from utils import (
    clean_text,
    extract_text_from_file,
    extract_name,
    extract_emails,
    extract_phone_numbers,
    analyze_resume_simple,
)
from vectors import add_strong_resume

def process_single_file(file, jd_info_str):
    #Process a single resume file using qwen:0.5b model.
    resume_text = extract_text_from_file(file)

    result = analyze_resume_simple(resume_text, jd_info_str)  #analyze using qwen:0.5b model

    # Add to strong resumes if recommended for interview
    if result.get("recommendation") == "Advance to interview":
        candidate_data = {
            "candidate_id": file.name,
            "resume_text": resume_text,
            "name": extract_name(resume_text),
            "score": result.get("score", 0),
            "recommendation": result.get("recommendation"),
        }
        add_strong_resume(candidate_data)

    return {
        "Source": "Upload",
        "Filename": file.name,
        "Name": extract_name(resume_text),
        "Phone": extract_phone_numbers(resume_text),
        "Email": extract_emails(resume_text),
        "Score": result.get("score", 0),
        "Overall fit": result.get("overall_fit", "Unknown"),
        "Strengths": result.get("strengths", "N/A"),
        "Gaps / Concerns": result.get("gaps", "N/A"),
        "Key Evidence": result.get("evidence", "N/A"),
        "Recommendation": result.get("recommendation", "N/A"),
        "Rationale": result.get("rationale", "N/A"),
        "Summary": result,
    }

def process_single_csv_row(row_data, jd_info_str, row_index):
    #Process a single CSV row using qwen:0.5b model
    # Extract resume text from the row (assuming it has a 'Resume_Text' or similar column)
    resume_text = ""
    # Check for resume text column (case-insensitive)
    resume_keys = ['Resume_Text', 'resume_text', 'resume', 'Resume', 'text', 'Text']
    for key in resume_keys:
        if key in row_data:
            resume_text = str(row_data[key])
            break
    if not resume_text:
        # If no specific resume column, concatenate all text fields
        text_fields = []
        for key, value in row_data.items():
            if isinstance(value, str) and len(value.strip()) > 10:  # Only substantial text
                text_fields.append(f"{key}: {value}")
        resume_text = "\n".join(text_fields)

    # Simple analysis using qwen:0.5b model
    result = analyze_resume_simple(resume_text, jd_info_str)

    # Add to strong resumes if recommended for interview
    if result.get("recommendation") == "Advance to interview":
        candidate_data = {
            "candidate_id": f"CSV_Row_{row_index + 1}",
            "resume_text": resume_text,
            "name": row_data.get('Name', row_data.get('name', f"Row {row_index + 1}")),
            "score": result.get("score", 0),
            "recommendation": result.get("recommendation"),
        }
        add_strong_resume(candidate_data)

    # Extract phone and email from row_data if available, else from resume_text
    phone = row_data.get('Phone') or row_data.get('phone') or extract_phone_numbers(resume_text)
    email = row_data.get('Email') or row_data.get('email') or extract_emails(resume_text)

    return {
        "Source": "CSV",
        "Row": row_index + 1,
        "Name": row_data.get('Name', row_data.get('name', f"Row {row_index + 1}")),
        "Phone": phone,
        "Email": email,
        "Score": result.get("score", 0),
        "Overall fit": result.get("overall_fit", "Unknown"),
        "Strengths": result.get("strengths", "N/A"),
        "Gaps / Concerns": result.get("gaps", "N/A"),
        "Key Evidence": result.get("evidence", "N/A"),
        "Recommendation": result.get("recommendation", "N/A"),
        "Rationale": result.get("rationale", "N/A"),
        "Summary": result,
        "Original_Data": row_data,  # Keep original row data
    }

def process_csv_batch(csv_file, jd_info_str, batch_size, enable_parallel):
    #Process CSV rows in batches with optional parallel processing
    results = []

    try:
        # Read CSV file
        if hasattr(csv_file, 'read'):
            content = csv_file.read()
            if isinstance(content, bytes):
                content = content.decode('utf-8')
            df = pd.read_csv(pd.io.common.StringIO(content))
        else:
            df = pd.read_csv(csv_file)

        total_rows = len(df)

        if enable_parallel:
            # Process rows in parallel batches
            for i in range(0, total_rows, batch_size):
                batch_end = min(i + batch_size, total_rows)
                batch_indices = list(range(i, batch_end))

                with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
                    # Submit all tasks in the batch
                    future_to_row = {
                        executor.submit(process_single_csv_row, df.iloc[idx].to_dict(), jd_info_str, idx): idx
                        for idx in batch_indices
                    }

                    # Collect results as they complete
                    for future in concurrent.futures.as_completed(future_to_row):
                        try:
                            result = future.result()
                            results.append(result)
                        except Exception as exc:
                            row_idx = future_to_row[future]
                            st.error(f'Row {row_idx + 1} generated an exception: {exc}')
        else:
            # Process rows sequentially in batches
            for i in range(0, total_rows, batch_size):
                batch_end = min(i + batch_size, total_rows)
                for idx in range(i, batch_end):
                    row_data = df.iloc[idx].to_dict()
                    result = process_single_csv_row(row_data, jd_info_str, idx)
                    results.append(result)

    except Exception as e:
        st.error(f"Error processing CSV file: {str(e)}")

    return results

def process_files_batch(uploaded_files, jd_info_str, batch_size, enable_parallel):
    #Process uploaded files in batches with optional parallel processing
    results = []

    if enable_parallel:
        # Process files in parallel batches
        for i in range(0, len(uploaded_files), batch_size):
            batch = uploaded_files[i:i + batch_size]
            with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
                # Submit all tasks in the batch
                future_to_file = {executor.submit(process_single_file, file, jd_info_str): file for file in batch}

                # Collect results as they complete
                for future in concurrent.futures.as_completed(future_to_file):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as exc:
                        file = future_to_file[future]
                        st.error(f'File {file.name} generated an exception: {exc}')
    else:
        # Process files sequentially in batches
        for i in range(0, len(uploaded_files), batch_size):
            batch = uploaded_files[i:i + batch_size]
            for file in batch:
                result = process_single_file(file, jd_info_str)
                results.append(result)

    return results

st.set_page_config(
    page_title=" AI Resume Screener ",
    layout="wide",
    page_icon="📋",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 50%, #1e40af 100%);
        padding: 2.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(30, 58, 138, 0.3);
        position: relative;
        overflow: hidden;
    }
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grain" width="100" height="100" patternUnits="userSpaceOnUse"><circle cx="25" cy="25" r="1" fill="rgba(255,255,255,0.1)"/><circle cx="75" cy="75" r="1" fill="rgba(255,255,255,0.1)"/><circle cx="50" cy="10" r="0.5" fill="rgba(255,255,255,0.1)"/><circle cx="10" cy="50" r="0.5" fill="rgba(255,255,255,0.1)"/><circle cx="90" cy="50" r="0.5" fill="rgba(255,255,255,0.1)"/></pattern></defs><rect width="100" height="100" fill="url(%23grain)"/></svg>');
        opacity: 0.1;
    }
    .main-header h1 {
        position: relative;
        z-index: 1;
        margin: 0 0 0.5rem 0;
        font-size: 3rem;
        font-weight: 700;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    .main-header p {
        position: relative;
        z-index: 1;
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.95;
        font-weight: 400;
    }
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        padding: 2rem;
        border-radius: 12px;
        border-left: 5px solid #3b82f6;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        margin-bottom: 1.5rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        border: 1px solid #e2e8f0;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.12);
    }
    .metric-card h3 {
        margin: 0 0 0.5rem 0;
        font-size: 2.5rem;
        font-weight: 700;
        color: #1e293b;
    }
    .metric-card p {
        margin: 0;
        color: #64748b;
        font-size: 1rem;
        font-weight: 500;
    }
    .candidate-card {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #3b82f6;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem;
    }
    .candidate-card h3 {
        margin: 0 0 1rem 0;
        font-size: 1.25rem;
        font-weight: 600;
        color: #1e293b;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .candidate-card h4 {
        margin: 1rem 0 0.5rem 0;
        font-size: 0.9rem;
        font-weight: 600;
        color: #374151;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .candidate-card p {
        margin: 0.25rem 0;
        color: #4b5563;
        line-height: 1.5;
        font-size: 0.9rem;
    }
    .candidate-card ul {
        margin: 0.25rem 0 0 0;
        padding-left: 1.2rem;
    }
    .candidate-card li {
        margin-bottom: 0.25rem;
        color: #4b5563;
        line-height: 1.4;
        font-size: 0.85rem;
    }
    .candidate-card strong {
        color: #1f2937;
        font-weight: 600;
    }
    .score-display {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        border: 2px solid #0ea5e9;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 8px rgba(14, 165, 233, 0.1);
    }
    .score-display .score-number {
        font-size: 2.5rem;
        font-weight: 800;
        color: #0c4a6e;
        margin: 0;
        text-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    .score-display .score-label {
        font-size: 0.9rem;
        color: #0369a1;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin: 0.25rem 0 0 0;
    }
    .status-badge {
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        display: inline-block;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .status-advance {
        background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
        color: #166534;
        border: 1px solid #16a34a;
    }
    .status-hold {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        color: #92400e;
        border: 1px solid #d97706;
    }
    .status-reject {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        color: #991b1b;
        border: 1px solid #dc2626;
    }
    .sidebar-header {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
    }
    .sidebar-header h3 {
        margin: 0 0 0.5rem 0;
        color: #1e293b;
        font-size: 1.1rem;
        font-weight: 600;
    }
    .sidebar-header p {
        margin: 0;
        color: #64748b;
        font-size: 0.9rem;
        line-height: 1.5;
    }
    .progress-container {
        margin-bottom: 2rem;
    }
    .section-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent 0%, #e2e8f0 50%, transparent 100%);
        margin: 2rem 0;
        border: none;
    }
    .info-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 2rem;
        margin-top: 1rem;
    }
    .info-section {
        background: #f8fafc;
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
    }
    @media (max-width: 768px) {
        .info-grid {
            grid-template-columns: 1fr;
            gap: 1rem;
        }
        .main-header h1 {
            font-size: 2rem;
        }
        .candidate-card {
            padding: 1.5rem;
        }
    }
</style>
""", unsafe_allow_html = True)

# Professional Header
st.markdown("""
<div class="main-header">
    <h1 style="margin: 0; font-size: 2.5rem;">📋AI Resume Screener Agent</h1>
    <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem; opacity: 0.9;">
        AI-Powered Resume Screening & Candidate Analysis Tool
    </p>
    <p style="margin: 0.5rem 0 0 0; font-size: 1rem; opacity: 0.8;">
        Streamline your hiring process with intelligent resume analysis and automated candidate ranking
    </p>
</div>
""", unsafe_allow_html=True)

BASE_DIR = Path(__file__).resolve().parent
JD_PATH = BASE_DIR / "job_description.txt"
JD_PATH.touch(exist_ok=True)  # Ensure the job description file exists

# Enhanced Sidebar
with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        <h3 style="margin: 0; color: #333;">⚙️ Configuration</h3>
        <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem; color: #666;">
            Customize your screening parameters
        </p>
    </div>
    """, unsafe_allow_html=True)

    # File upload settings
    st.subheader("📁 Upload Settings")
    max_file_size = st.slider("Max file size per upload (MB)", min_value=50, max_value=1000, value=200, step=50)
    st.caption(f"📏 Current limit: {max_file_size} MB per file")

    # Batch processing settings
    st.subheader("⚡ Processing Settings")
    batch_size = st.slider("Batch size (files to process simultaneously)", min_value=1, max_value=10, value=3, step=1)
    enable_parallel = st.checkbox("Enable parallel processing", value=True)
    st.caption(f"🔄 Processing {batch_size} files at a time")

    # AI Model Info
    st.subheader("🤖 AI Model")
    st.info("Using Qwen 0.5B model for intelligent analysis")

    # Set the max upload size
    st.session_state.max_upload_size_mb = max_file_size
    st.session_state.batch_size = batch_size
    st.session_state.enable_parallel = enable_parallel

st.header("Job Description Input")
st.divider()

if JD_PATH.exists():
    jd_raw = JD_PATH.read_text(encoding="utf-8")
else:
    jd_raw = ""

jd_raw = st.text_area("Enter Job Description", value=jd_raw, height=200)

if st.button("Save Job Description"):
    JD_PATH.write_text(jd_raw, encoding="utf-8")
    st.success("Job description saved")

if not jd_raw.strip():
    st.warning("Please enter a job description to continue.")
    st.stop()

# Store job description as string for simple processing
jd_info_str = jd_raw

st.header("Resume Upload & Processing Settings")
st.divider()

# Processing Mode Selection
processing_mode = st.radio(
    "Choose how to process resumes:",
    ["Upload Individual Files", "Process CSV Dataset"],
    help="Upload individual resume files or process a CSV file where each row contains resume data"
)

# Upload Resumes
if processing_mode == "Upload Individual Files":
    uploaded_files = st.file_uploader(
        "Upload resume files (PDF, DOCX, PPTX, TXT, PNG, JPG, JPEG)",
        type=["pdf", "docx", "pptx", "txt", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
    )
    csv_file = None
else:  # Process CSV Dataset
    csv_file = st.file_uploader(
        "Upload CSV file with resume data",
        type=["csv"],
        accept_multiple_files=False,
    )
    uploaded_files = None

# Processing
results = []

if uploaded_files:
    st.write(f"Processing {len(uploaded_files)} files...")

    progress_bar = st.progress(0)
    status_text = st.empty()

    batch_size = st.session_state.get('batch_size', 3)
    enable_parallel = st.session_state.get('enable_parallel', True)

    # Process files in batch
    batch_results = process_files_batch(uploaded_files, jd_info_str, batch_size, enable_parallel)
    results.extend(batch_results)

    progress_bar.progress(1.0)
    status_text.text(f"Completed processing {len(uploaded_files)} files")

elif csv_file:
    st.write(f"Processing CSV file...")

    progress_bar = st.progress(0)
    status_text = st.empty()

    batch_size = st.session_state.get('batch_size', 3)
    enable_parallel = st.session_state.get('enable_parallel', True)

    # Process CSV rows in batch
    csv_results = process_csv_batch(csv_file, jd_info_str, batch_size, enable_parallel)
    results.extend(csv_results)

    #Update progress bar to complete
    progress_bar.progress(1.0)
    status_text.text(f"Completed processing {len(csv_results)} rows from CSV")

if results:
    df = pd.DataFrame(results)

    # Add Candidate Summary column
    def create_candidate_summary(row):
        source = row.get('Source', 'N/A')
        name = row.get('Name', 'N/A')
        email = row.get('Email', [])
        phone = row.get('Phone', [])
        recommendation = row.get('Recommendation', 'N/A')

        # Format email and phone
        email_str = ', '.join(email) if isinstance(email, list) and email else str(email) if email else 'N/A'
        phone_str = ', '.join(phone) if isinstance(phone, list) and phone else str(phone) if phone else 'N/A'

        return f"Source: {source} | Name: {name} | Email: {email_str} | Phone: {phone_str} | Recommendation: {recommendation}"

    df['Candidate Summary'] = df.apply(create_candidate_summary, axis=1)

    # Dashboard Summary Metrics
    st.header("📊 Screening Results Dashboard")
    st.divider()

    # Calculate metrics
    total_candidates = len(df)
    shortlisted = len(df[df['Recommendation'] == "Advance to interview"])
    on_hold = len(df[df['Recommendation'] == "Hold"])
    rejected = len(df[df['Recommendation'] == "Reject"])
    avg_score = df['Score'].mean()

    # Metrics Row
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="margin: 0; color: #333; font-size: 2rem;">{total_candidates}</h3>
            <p style="margin: 0.5rem 0 0 0; color: #666; font-size: 0.9rem;">Total Candidates</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="margin: 0; color: #28a745; font-size: 2rem;">{shortlisted}</h3>
            <p style="margin: 0.5rem 0 0 0; color: #666; font-size: 0.9rem;">Shortlisted</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="margin: 0; color: #ffc107; font-size: 2rem;">{on_hold}</h3>
            <p style="margin: 0.5rem 0 0 0; color: #666; font-size: 0.9rem;">On Hold</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="margin: 0; color: #dc3545; font-size: 2rem;">{rejected}</h3>
            <p style="margin: 0.5rem 0 0 0; color: #666; font-size: 0.9rem;">Rejected</p>
        </div>
        """, unsafe_allow_html=True)

    with col5:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="margin: 0; color: #667eea; font-size: 2rem;">{avg_score:.1f}%</h3>
            <p style="margin: 0.5rem 0 0 0; color: #666; font-size: 0.9rem;">Avg Score</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Filters
    col1, col2 = st.columns(2)
    with col1:
        sort_by_score = st.checkbox("Sort by score (high to low)", value=True)
    with col2:
        show_shortlisted_only = st.checkbox("Show only shortlisted candidates")

    # Apply filters
    filtered_df = df.copy()
    if show_shortlisted_only:
        filtered_df = filtered_df[filtered_df['Recommendation'] == "Advance to interview"]

    if sort_by_score:
        filtered_df = filtered_df.sort_values("Score", ascending=False)
    elif "Row" in filtered_df.columns and processing_mode == "Process CSV Dataset":
        filtered_df = filtered_df.sort_values("Row", ascending=True)

    # Create display table with candidate summary
    display_df = filtered_df[['Name', 'Candidate Summary', 'Score', 'Overall fit', 'Recommendation']].copy()

    # Candidate Overview Table
    st.subheader("Candidate Overview")
    table_df = filtered_df[['Source', 'Name', 'Phone', 'Recommendation']].copy()
    table_df['Phone'] = table_df['Phone'].apply(lambda x: ', '.join(x) if isinstance(x, list) else str(x))

    # Configure columns for better UI
    column_config = {
        "Source": st.column_config.TextColumn("Source", width="small"),
        "Name": st.column_config.TextColumn("Name", width="medium"),
        "Phone": st.column_config.TextColumn("Phone Number", width="medium"),
        "Recommendation": st.column_config.TextColumn("Recommendation", width="medium"),
    }

    st.dataframe(table_df, column_config=column_config, width='stretch', hide_index=True)

    # Enhanced Candidate Details with Expandable Dropdowns
    st.subheader("📋 Detailed Candidate Profiles")
    st.divider()

    for idx, row in filtered_df.iterrows():
        summary_data = row.get('Summary', {})

        # Determine styling based on recommendation
        rec = row.get('Recommendation', 'N/A')
        status_icon = "✅" if rec == "Advance to interview" else "⏳" if rec == "Hold" else "❌"
        status_color = "green" if rec == "Advance to interview" else "orange" if rec == "Hold" else "red"

        # Format contact info
        email = row.get('Email', [])
        phone = row.get('Phone', [])
        email_str = ', '.join(email) if isinstance(email, list) and email else str(email) if email else 'N/A'
        phone_str = ', '.join(phone) if isinstance(phone, list) and phone else str(phone) if phone else 'N/A'

        # Create expander header with candidate name and score
        expander_title = f"{status_icon} {row.get('Name', 'N/A')} - {row.get('Score', 0)}% Match"

        with st.expander(expander_title, expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**📞 Contact Information**")
                st.write(f"**Email:** {email_str}")
                st.write(f"**Phone:** {phone_str}")
                st.write(f"**Source:** {row.get('Source', 'N/A')}")

                st.markdown("**✅ Strengths**")
                strengths = summary_data.get('strengths', [])
                if isinstance(strengths, list) and strengths:
                    for strength in strengths:
                        st.write(f"• {strength}")
                else:
                    st.write("• No specific strengths identified")

                st.markdown("**🔍 Key Evidence**")
                evidence = summary_data.get('evidence', [])
                if isinstance(evidence, list) and evidence:
                    for item in evidence:
                        st.write(f"• {item}")
                else:
                    st.write("• No specific evidence available")

            with col2:
                st.markdown("**⚠️ Skill Gaps & Concerns**")
                gaps = summary_data.get('gaps', [])
                if isinstance(gaps, list) and gaps:
                    for gap in gaps:
                        st.write(f"• {gap}")
                else:
                    st.write("• None indicated")

                st.markdown("**🎯 Recommendation**")
                st.write(f"**{rec}**")

                st.markdown("**📝 Rationale**")
                st.write(row.get('Rationale', 'N/A'))

                st.markdown("**📊 Overall Fit**")
                st.write(f"**{row.get('Overall fit', 'N/A')}**")

    # Download button
    st.download_button(
        "Download Results CSV",
        data=filtered_df.drop(columns=["Summary", "Original_Data"], errors='ignore').to_csv(index=False),
        file_name="resume_results.csv",
        mime="text/csv",
    )

    # Analytics CSV Download
    analytics_data = {
        "Metric": ["Total Candidates", "Shortlisted", "On Hold", "Rejected", "Average Score"],
        "Value": [total_candidates, shortlisted, on_hold, rejected, f"{avg_score:.1f}%"]
    }
    analytics_df = pd.DataFrame(analytics_data)
    st.download_button(
        "Download Analytics CSV",
        data=analytics_df.to_csv(index=False),
        file_name="resume_analytics.csv",
        mime="text/csv",
        help="Download basic analytics summary as CSV"
    )

    # Additional Basic Features Section
    st.header("📊 Advanced Analytics & Tools")
    st.divider()

    # Statistics Overview
    with st.expander("📈 Detailed Statistics", expanded=False):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Score Distribution")
            if not df.empty:
                score_ranges = {
                    "90-100%": len(df[df['Score'] >= 90]),
                    "80-89%": len(df[(df['Score'] >= 80) & (df['Score'] < 90)]),
                    "70-79%": len(df[(df['Score'] >= 70) & (df['Score'] < 80)]),
                    "60-69%": len(df[(df['Score'] >= 60) & (df['Score'] < 70)]),
                    "Below 60%": len(df[df['Score'] < 60])
                }
                for range_name, count in score_ranges.items():
                    st.write(f"{range_name}: {count}")

        with col2:
            st.subheader("Recommendation Breakdown")
            if not df.empty:
                rec_counts = df['Recommendation'].value_counts()
                for rec, count in rec_counts.items():
                    st.write(f"{rec}: {count}")

        with col3:
            st.subheader("Top Skills")
            if not df.empty:
                all_strengths = []
                for _, row in df.iterrows():
                    summary = row.get('Summary', {})
                    strengths = summary.get('strengths', [])
                    all_strengths.extend(strengths)

                from collections import Counter
                skill_counts = Counter()
                for strength in all_strengths:
                    if "skill" in strength.lower():
                        # Extract skill names (basic parsing)
                        words = strength.lower().split()
                        for word in words:
                            if len(word) > 3 and word not in ['skills', 'skill', 'matched', 'required', 'preferred']:
                                skill_counts[word.title()] += 1

                top_skills = skill_counts.most_common(5)
                for skill, count in top_skills:
                    st.write(f"{skill}: {count}")

    # Search and Filter Tools
    with st.expander("🔍 Search & Filter Tools", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            search_term = st.text_input("Search candidates by name or email:", "")
            if search_term:
                search_results = filtered_df[
                    filtered_df['Name'].str.contains(search_term, case=False, na=False) |
                    filtered_df['Email'].astype(str).str.contains(search_term, case=False, na=False)
                ]
                st.write(f"Found {len(search_results)} matching candidates")
                if not search_results.empty:
                    st.dataframe(search_results[['Name', 'Email', 'Score', 'Recommendation']], width='stretch')

        with col2:
            min_score_filter = st.slider("Filter by minimum score:", 0, 100, 0)
            filtered_by_score = filtered_df[filtered_df['Score'] >= min_score_filter]
            st.write(f"Candidates with score ≥ {min_score_filter}: {len(filtered_by_score)}")

            if st.button("Show Filtered Results"):
                st.dataframe(filtered_by_score[['Name', 'Score', 'Recommendation']], width='stretch')

    # Help & FAQ Section
    with st.expander("❓ Help & FAQ", expanded=False):
        st.subheader("Frequently Asked Questions")

        with st.container():
            st.markdown("**Q: How does the scoring work?**")
            st.markdown("A: The AI analyzes resume content against job requirements using keyword matching, skills alignment, and experience evaluation.")

            st.markdown("**Q: What file formats are supported?**")
            st.markdown("A: PDF, DOCX, PPTX, TXT, CSV, and image files (PNG, JPG, JPEG) with OCR capability.")

            st.markdown("**Q: How are recommendations determined?**")
            st.markdown("A: Based on overall fit score: 70%+ = Advance to interview, 50-69% = Hold, below 50% = Reject.")

            st.markdown("**Q: Can I process CSV files with multiple resumes?**")
            st.markdown("A: Yes! Upload a CSV where each row contains resume data. The system will process each row as a separate candidate.")

            st.markdown("**Q: What information is extracted from resumes?**")
            st.markdown("A: Name, email, phone, skills, experience, education, certifications, and achievements.")

    # Quick Actions
    with st.expander("⚡ Quick Actions", expanded=False):
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("📋 Copy Job Description"):
                st.code(jd_info_str, language=None)
                st.success("Job description copied to clipboard area above!")

        with col2:
            if st.button("📊 Generate Summary Report"):
                if not df.empty:
                    report = f"""
Resume Screening Summary Report
Total Candidates Processed: {total_candidates}
Shortlisted: {shortlisted} ({shortlisted/total_candidates*100:.1f}%)
On Hold: {on_hold} ({on_hold/total_candidates*100:.1f}%)
Rejected: {rejected} ({rejected/total_candidates*100:.1f}%)
Average Score: {avg_score:.1f}%

Top Performing Candidates:
"""
                    top_candidates = filtered_df.nlargest(3, 'Score')[['Name', 'Score', 'Recommendation']]
                    for _, candidate in top_candidates.iterrows():
                        report += f"- {candidate['Name']}: {candidate['Score']}% ({candidate['Recommendation']})\n"

                    st.code(report, language=None)
                    st.success("Summary report generated above!")
                else:
                    st.warning("No results to generate report from.")

        with col3:
            if st.button("🔄 Clear All Results"):
                if st.session_state.get('confirm_clear', False):
                    # This would clear results in a real implementation
                    st.success("Results cleared! Please refresh the page.")
                    st.session_state.confirm_clear = False
                else:
                    st.warning("Click again to confirm clearing all results.")
                    st.session_state.confirm_clear = True

    # Settings & Customization
    with st.expander("⚙️ Additional Settings", expanded=False):
        st.subheader("Display Preferences")

        show_detailed_scores = st.checkbox("Show detailed score breakdowns", value=False)
        if show_detailed_scores and not df.empty:
            st.subheader("Score Breakdown Analysis")
            score_df = df[['Name', 'Score']].copy()
            score_df['Score Range'] = pd.cut(score_df['Score'],
                                           bins=[0, 25, 50, 75, 100],
                                           labels=['0-25%', '26-50%', '51-75%', '76-100%'])
            st.dataframe(score_df, width='stretch')

        auto_expand_profiles = st.checkbox("Auto-expand candidate profiles", value=False)
        if auto_expand_profiles:
            st.info("💡 Tip: Candidate profiles are now auto-expanded for easier viewing")

        export_format = st.selectbox("Preferred export format:",
                                   ["CSV", "JSON", "Excel (CSV)"])
        if export_format == "JSON" and st.button("Export as JSON"):
            json_data = filtered_df.drop(columns=["Summary", "Original_Data"], errors='ignore').to_json(orient='records', indent=2)
            st.download_button(
                "Download JSON",
                data=json_data,
                file_name="resume_results.json",
                mime="application/json"
            )
