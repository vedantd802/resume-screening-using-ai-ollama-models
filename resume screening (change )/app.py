import streamlit as st
import pandas as pd
from pathlib import Path
import concurrent.futures
import json

from utils import (
    clean_text,
    extract_text_from_file,
    extract_name,
    extract_emails,
    extract_phone_numbers,
    analyze_resume_simple,
)

def process_single_file(file, jd_info_str):
    #Process a single resume file using qwen:0.5b model.
    resume_text = extract_text_from_file(file)

    result = analyze_resume_simple(resume_text, jd_info_str)  #analyze using qwen:0.5b model

    return {
        "Source": "Upload",
        "Filename": file.name,
        "Name": extract_name(resume_text),
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
    if 'Resume_Text' in row_data:
        resume_text = str(row_data['Resume_Text'])
    elif 'resume' in row_data:
        resume_text = str(row_data['resume'])
    elif 'resume_text' in row_data:
        resume_text = str(row_data['resume_text'])
    else:
        # If no specific resume column, concatenate all text fields
        text_fields = []
        for key, value in row_data.items():
            if isinstance(value, str) and len(value.strip()) > 10:  # Only substantial text
                text_fields.append(f"{key}: {value}")
        resume_text = "\n".join(text_fields)

    # Simple analysis using qwen:0.5b model
    result = analyze_resume_simple(resume_text, jd_info_str)

    return {
        "Source": "CSV",
        "Row": row_index + 1,
        "Name": row_data.get('Name', row_data.get('name', f"Row {row_index + 1}")),
        "Email": row_data.get('Email', row_data.get('email', "Not provided")),
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

st.set_page_config(page_title="Resume Screener", layout="wide")
st.title("📄 AI Agent Resume Screener")

BASE_DIR = Path(__file__).resolve().parent
JD_PATH = BASE_DIR / "job_description.txt"

with st.sidebar:
    st.header("⚙️ Settings")

    # File upload settings
    st.subheader("📁 Upload Settings")
    max_file_size = st.slider("Max file size per upload (MB)", min_value=50, max_value=1000, value=2000, step=50)
    st.caption(f"Current limit: {max_file_size} MB per file")

    # Batch processing settings
    st.subheader("⚡ Batch Processing")
    batch_size = st.slider("Batch size (files to process simultaneously)", min_value=1, max_value=10, value=3, step=1)
    enable_parallel = st.checkbox("Enable parallel processing", value=True)
    st.caption(f"Processing {batch_size} files at a time")

    # Set the max upload size
    st.session_state.max_upload_size_mb = max_file_size
    st.session_state.batch_size = batch_size
    st.session_state.enable_parallel = enable_parallel

st.subheader("📌 Job Description")

if JD_PATH.exists():
    jd_raw = JD_PATH.read_text(encoding="utf-8")
else:
    jd_raw = ""

jd_raw = st.text_area("Edit Job Description", value=jd_raw, height=200)

if st.button("💾 Save Job Description"):
    JD_PATH.write_text(jd_raw, encoding="utf-8")
    st.success("Job description saved")

if not jd_raw.strip():
    st.warning("Please enter a job description to continue.")
    st.stop()

# Store job description as string for simple processing
jd_info_str = jd_raw

# Processing Mode Selection
st.subheader("🔄 Processing Mode")
processing_mode = st.radio(
    "Choose how to process resumes:",
    ["Upload Individual Files", "Process CSV Dataset"],
    help="Upload individual resume files or process a CSV file where each row contains resume data"
)

# Upload Resumes
st.subheader("📤 Upload Resumes")

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

results = []

if uploaded_files:
    st.write(f"📋 Processing {len(uploaded_files)} files using **qwen:0.5b** model...")

    progress_bar = st.progress(0)
    status_text = st.empty()

    batch_size = st.session_state.get('batch_size', 3)
    enable_parallel = st.session_state.get('enable_parallel', True)

    # Process files in batch
    batch_results = process_files_batch(uploaded_files, jd_info_str, batch_size, enable_parallel)
    results.extend(batch_results)

    progress_bar.progress(1.0)
    status_text.text(f"✅ Completed processing {len(uploaded_files)} files")

elif csv_file:
    st.write(f"📋 Processing CSV file using **qwen:0.5b** model...")

    progress_bar = st.progress(0)
    status_text = st.empty()

    batch_size = st.session_state.get('batch_size', 3)
    enable_parallel = st.session_state.get('enable_parallel', True)

    # Process CSV rows in batch
    csv_results = process_csv_batch(csv_file, jd_info_str, batch_size, enable_parallel)
    results.extend(csv_results)
     
    #Update progress bar to complete
    progress_bar.progress(1.0)
    status_text.text(f"✅ Completed processing {len(csv_results)} rows from CSV")

if results:
    df = pd.DataFrame(results)

    # Sort by score for individual files, by row number for CSV data
    if "Row" in df.columns and processing_mode == "Process CSV Dataset":
        df = df.sort_values("Row", ascending=True)
        st.subheader("📊 CSV Rows Processed Sequentially")
    else:
        df = df.sort_values("Score", ascending=False)
        st.subheader("📊 Candidate Ranking & Summaries")

    display_cols = ["Name", "Score", "Overall fit", "Recommendation"]
    if "Filename" in df.columns:
        display_cols.insert(0, "Filename")
    elif "Row" in df.columns:
        display_cols.insert(0, "Row")
    if "Email" in df.columns:
        display_cols.insert(1, "Email")
    st.dataframe(df[display_cols], width='stretch')

    st.markdown("### 📋 Detailed Candidate Summaries")
    for i, row in df.iterrows():
        name = row['Name'] if pd.notna(row['Name']) else "Unknown"
        source = ""
        if "Filename" in row and pd.notna(row['Filename']):
            source = f" ({row['Filename']})"
        elif "Row" in row and pd.notna(row['Row']):
            source = f" (CSV Row {row['Row']})"
        st.markdown(f"---\n#### {name}{source}")
        st.markdown(f"**Overall fit:** {row['Overall fit']} &nbsp;&nbsp; **Score:** {row['Score']} / 100")
        st.markdown(f"**Recommendation:** {row['Recommendation']}")

        # Handle strengths (could be list or string)
        strengths = row['Strengths']
        if isinstance(strengths, list):
            st.markdown(f"**Strengths:**\n- " + "\n- ".join(strengths))
        else:
            st.markdown(f"**Strengths:**\n- {strengths}")

        # Handle gaps (could be list or string)
        gaps = row['Gaps / Concerns']
        if isinstance(gaps, list):
            st.markdown(f"**Gaps / Concerns:**\n- " + "\n- ".join(gaps))
        else:
            st.markdown(f"**Gaps / Concerns:**\n- {gaps}")

        # Handle evidence (could be list or string)
        evidence = row['Key Evidence']
        if isinstance(evidence, list):
            st.markdown(f"**Key Evidence:**\n- " + "\n- ".join(evidence))
        else:
            st.markdown(f"**Key Evidence:**\n- {evidence}")

        st.markdown(f"**Rationale:** {row['Rationale']}")

    st.download_button(
        "⬇️ Download Results CSV",
        data=df.drop(columns=["Summary"], errors='ignore').to_csv(index=False),
        file_name="resume_results.csv",
        mime="text/csv",
    )
