from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pathlib import Path
import pandas as pd
from typing import List, Optional
import json
import concurrent.futures
import os
from passlib.context import CryptContext
from datetime import datetime, timedelta
import jwt
from pydantic import BaseModel
import io

from utils import (
    extract_text_from_file,
    extract_name,
    extract_emails,
    extract_phone_numbers,
    analyze_resume_simple,
    extract_name_and_email_with_model,
)

# Security
SECRET_KEY = "your-secret-key-here"  # In production, use environment variable
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

app = FastAPI(title="AI Resume Screener API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage (replace with database in production)
processed_results = []
job_descriptions = {}

# JSON file for persistent storage
RESUMES_JSON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resumes.json")

def load_resumes_from_json():
    """Load resumes from JSON file"""
    if os.path.exists(RESUMES_JSON_PATH):
        try:
            with open(RESUMES_JSON_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading resumes from JSON: {e}")
            return []
    return []

def save_resumes_to_json(resumes):
    """Save resumes to JSON file"""
    try:
        with open(RESUMES_JSON_PATH, 'w', encoding='utf-8') as f:
            json.dump(resumes, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error saving resumes to JSON: {e}")

def clear_resumes_json():
    """Clear the resumes JSON file"""
    try:
        if os.path.exists(RESUMES_JSON_PATH):
            os.remove(RESUMES_JSON_PATH)
    except Exception as e:
        print(f"Error clearing resumes JSON: {e}")

# Pydantic models
class LoginRequest(BaseModel):
    email: str
    password: str

class RegisterRequest(BaseModel):
    email: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class JobDescription(BaseModel):
    jd_text: str

# Auth functions
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

users_db = {"admin@example.com": get_password_hash("password")}  # Default user for development

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def authenticate_user(email: str, password: str):
    # Allow any email and password for demo purposes
    return {"email": email}

def register_user(email: str, password: str):
    # Check if user already exists
    if email in users_db:
        return False  # User already exists
    # Hash password and store user
    hashed_password = get_password_hash(password)
    users_db[email] = hashed_password
    return True

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    return username

# Resume processing functions (adapted from original)
def process_single_file(file, jd_info_str):
    # Process a single resume file using qwen:0.5b model.
    resume_text = extract_text_from_file(file)

    # Check if text extraction failed
    if isinstance(resume_text, str) and resume_text.startswith("Error extracting text"):
        # Return error result for failed extraction
        return {
            "Source": "Upload",
            "Filename": file.name,
            "Name": "Not found",
            "Phone": [],
            "Email": [],
            "Score": 0,
            "Overall fit": "Unknown",
            "Strengths": ["N/A"],
            "Gaps / Concerns": [resume_text],  # Include the error message
            "Key Evidence": ["N/A"],
            "Recommendation": "Reject",
            "Rationale": resume_text,
            "Summary": {"error": resume_text},
        }

    # Use LLM for better name and email extraction
    name_email_info = extract_name_and_email_with_model(resume_text)
    name = name_email_info.get("name", extract_name(resume_text))  # Fallback to rule-based

    result = analyze_resume_simple(resume_text, jd_info_str, use_llm=False)  # Use fast keyword analysis

    return {
        "Source": "Upload",
        "Filename": file.name,
        "Name": name,
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
    # Process a single CSV row using qwen:0.5b model
    # Extract resume text from the row (assuming it has a 'Resume_Text' or similar column)
    resume_text = ""
    # Check for resume text column (case-insensitive)
    resume_keys = ['Resume_Text', 'resume_text', 'resume', 'Resume', 'text', 'Text', 'ResumeText', 'resumeText']
    for key in resume_keys:
        if key in row_data and row_data[key] and str(row_data[key]).strip():
            resume_text = str(row_data[key]).strip()
            break

    if not resume_text:
        # If no specific resume column, try to find any column with substantial text content
        text_fields = []
        for key, value in row_data.items():
            if isinstance(value, str) and len(value.strip()) > 20:  # Substantial text content
                text_fields.append(f"{key}: {value.strip()}")
        if text_fields:
            resume_text = "\n".join(text_fields)
        else:
            # Last resort: concatenate all non-empty string values
            all_text = []
            for value in row_data.values():
                if isinstance(value, str) and value.strip():
                    all_text.append(value.strip())
            resume_text = " ".join(all_text)

    # Ensure we have some text to analyze
    if not resume_text or len(resume_text.strip()) < 10:
        resume_text = f"No substantial resume text found in row {row_index + 1}. Available data: {', '.join([f'{k}: {str(v)[:50]}...' for k, v in row_data.items() if v])}"

    # Simple analysis using qwen:0.5b model (with timeout to prevent hanging)
    try:
        result = analyze_resume_simple(resume_text, jd_info_str, use_llm=True)
    except Exception as e:
        # Fallback to simple analysis if LLM fails or hangs
        result = analyze_resume_simple(resume_text, jd_info_str, use_llm=False)

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

def process_csv_cell_by_cell(csv_file, jd_info_str):
    # Process CSV file cell by cell (line by line)
    results = []
    import csv as csv_module

    print(f"Starting CSV processing with jd_info_str length: {len(jd_info_str)}")

    try:
        # Handle different input types
        if hasattr(csv_file, 'read'):
            # It's a file-like object (StringIO from upload)
            print("Processing as file-like object (StringIO)")
            csv_reader = csv_module.reader(csv_file)
        else:
            # It's a file path
            print(f"Processing as file path: {csv_file}")
            csv_reader = csv_module.reader(open(csv_file, 'r', encoding='utf-8'))

        # Read header row
        headers = next(csv_reader, None)
        print(f"Headers found: {headers}")
        if not headers:
            raise ValueError("CSV file has no headers")

        # Clean headers
        headers = [h.strip() for h in headers]
        print(f"Cleaned headers: {headers}")

        row_index = 0
        for row in csv_reader:
            row_index += 1
            print(f"Processing row {row_index}: {row}")

            # Process each cell in the row
            row_data = {}
            for col_idx, cell_value in enumerate(row):
                if col_idx < len(headers):
                    header = headers[col_idx]
                    cell_value = cell_value.strip() if cell_value else ""
                    row_data[header] = cell_value

            print(f"Row data: {row_data}")

            # Skip empty rows
            if not any(row_data.values()):
                print(f"Skipping empty row {row_index}")
                continue

            # Process this row using the existing function
            try:
                result = process_single_csv_row(row_data, jd_info_str, row_index - 1)
                results.append(result)
                print(f"Successfully processed row {row_index}")
            except Exception as exc:
                print(f'Row {row_index} generated an exception: {exc}')
                # Return error result for failed row
                results.append({
                    "Source": "CSV",
                    "Row": row_index,
                    "Name": f"Row {row_index}",
                    "Phone": [],
                    "Email": [],
                    "Score": 0,
                    "Overall fit": "Unknown",
                    "Strengths": ["N/A"],
                    "Gaps / Concerns": [f"Error processing row: {str(exc)}"],
                    "Key Evidence": ["N/A"],
                    "Recommendation": "Reject",
                    "Rationale": f"Row processing error: {str(exc)}",
                    "Summary": {"error": f"Row processing error: {str(exc)}"},
                })

        print(f"Total rows processed: {row_index}")
        print(f"Total results generated: {len(results)}")

    except Exception as e:
        print(f"Error processing CSV file: {str(e)}")
        import traceback
        traceback.print_exc()
        # Return error result for CSV reading failure
        results.append({
            "Source": "CSV",
            "Row": 1,
            "Name": "CSV File Error",
            "Phone": [],
            "Email": [],
            "Score": 0,
            "Overall fit": "Unknown",
            "Strengths": ["N/A"],
            "Gaps / Concerns": [f"Error reading CSV file: {str(e)}"],
            "Key Evidence": ["N/A"],
            "Recommendation": "Reject",
            "Rationale": f"CSV reading error: {str(e)}",
            "Summary": {"error": f"CSV reading error: {str(e)}"},
        })

    return results

def process_csv_batch(csv_file, jd_info_str, batch_size, enable_parallel):
    # Process CSV rows in batches with optional parallel processing
    results = []

    try:
        # Read CSV file
        if hasattr(csv_file, 'read'):
            content = csv_file.read()
            if isinstance(content, bytes):
                content = content.decode('utf-8')
            df = pd.read_csv(io.StringIO(content))
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
                            print(f'Row {row_idx + 1} generated an exception: {exc}')
                            # Return error result for failed row
                            results.append({
                                "Source": "CSV",
                                "Row": row_idx + 1,
                                "Name": f"Row {row_idx + 1}",
                                "Phone": [],
                                "Email": [],
                                "Score": 0,
                                "Overall fit": "Unknown",
                                "Strengths": ["N/A"],
                                "Gaps / Concerns": [f"Error processing row: {str(exc)}"],
                                "Key Evidence": ["N/A"],
                                "Recommendation": "Reject",
                                "Rationale": f"Row processing error: {str(exc)}",
                                "Summary": {"error": f"Row processing error: {str(exc)}"},
                            })
        else:
            # Process rows sequentially in batches
            for i in range(0, total_rows, batch_size):
                batch_end = min(i + batch_size, total_rows)
                for idx in range(i, batch_end):
                    try:
                        row_data = df.iloc[idx].to_dict()
                        result = process_single_csv_row(row_data, jd_info_str, idx)
                        results.append(result)
                    except Exception as exc:
                        print(f'Row {idx + 1} generated an exception: {exc}')
                        # Return error result for failed row
                        results.append({
                            "Source": "CSV",
                            "Row": idx + 1,
                            "Name": f"Row {idx + 1}",
                            "Phone": [],
                            "Email": [],
                            "Score": 0,
                            "Overall fit": "Unknown",
                            "Strengths": ["N/A"],
                            "Gaps / Concerns": [f"Error processing row: {str(exc)}"],
                            "Key Evidence": ["N/A"],
                            "Recommendation": "Reject",
                            "Rationale": f"Row processing error: {str(exc)}",
                            "Summary": {"error": f"Row processing error: {str(exc)}"},
                        })

    except Exception as e:
        print(f"Error processing CSV file: {str(e)}")
        # Return error result for CSV reading failure
        results.append({
            "Source": "CSV",
            "Row": 1,
            "Name": "CSV File Error",
            "Phone": [],
            "Email": [],
            "Score": 0,
            "Overall fit": "Unknown",
            "Strengths": ["N/A"],
            "Gaps / Concerns": [f"Error reading CSV file: {str(e)}"],
            "Key Evidence": ["N/A"],
            "Recommendation": "Reject",
            "Rationale": f"CSV reading error: {str(e)}",
            "Summary": {"error": f"CSV reading error: {str(e)}"},
        })

    return results

def process_files_batch(uploaded_files, jd_info_str, batch_size, enable_parallel):
    # Process uploaded files in batches with optional parallel processing
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
                        print(f'File {file.name} generated an exception: {exc}')
                        # Return error result for failed file
                        results.append({
                            "Source": "Upload",
                            "Filename": file.name,
                            "Name": "Not found",
                            "Phone": [],
                            "Email": [],
                            "Score": 0,
                            "Overall fit": "Unknown",
                            "Strengths": ["N/A"],
                            "Gaps / Concerns": [f"Error processing file: {str(exc)}"],
                            "Key Evidence": ["N/A"],
                            "Recommendation": "Reject",
                            "Rationale": f"File processing error: {str(exc)}",
                            "Summary": {"error": f"File processing error: {str(exc)}"},
                        })
    else:
        # Process files sequentially in batches
        for i in range(0, len(uploaded_files), batch_size):
            batch = uploaded_files[i:i + batch_size]
            for file in batch:
                try:
                    result = process_single_file(file, jd_info_str)
                    results.append(result)
                except Exception as exc:
                    print(f'File {file.name} generated an exception: {exc}')
                    # Return error result for failed file
                    results.append({
                        "Source": "Upload",
                        "Filename": file.name,
                        "Name": "Not found",
                        "Phone": [],
                        "Email": [],
                        "Score": 0,
                        "Overall fit": "Unknown",
                        "Strengths": ["N/A"],
                        "Gaps / Concerns": [f"Error processing file: {str(exc)}"],
                        "Key Evidence": ["N/A"],
                        "Recommendation": "Reject",
                        "Rationale": f"File processing error: {str(exc)}",
                        "Summary": {"error": f"File processing error: {str(exc)}"},
                    })

    return results

# API Endpoints
@app.post("/register")
async def register(request: RegisterRequest):
    if register_user(request.email, request.password):
        return {"message": "User registered successfully"}
    else:
        raise HTTPException(status_code=400, detail="User already exists")

@app.post("/signup")
async def signup(request: RegisterRequest):
    if register_user(request.email, request.password):
        return {"message": "User registered successfully"}
    else:
        raise HTTPException(status_code=400, detail="User already exists")

@app.post("/login", response_model=Token)
async def login(request: LoginRequest):
    user = authenticate_user(request.email, request.password)
    if not user:
        raise HTTPException(status_code=401, detail="Incorrect email or password")
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["email"]}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/upload")
async def upload_resumes(
    files: List[UploadFile] = File(None),
    csv_file: UploadFile = File(None),
    jd_text: str = Form(...),
    batch_size: int = Form(3),
    enable_parallel: bool = Form(True)
):
    global processed_results

    if not files and not csv_file:
        raise HTTPException(status_code=400, detail="No files provided")

    results = []

    if files:
        # Process individual files
        file_objects = []
        for file in files:
            content = await file.read()
            # Create a file-like object
            from io import BytesIO
            file_obj = BytesIO(content)
            file_obj.name = file.filename
            file_objects.append(file_obj)

        results = process_files_batch(file_objects, jd_text, batch_size, enable_parallel)

    elif csv_file:
        # Process CSV file cell by cell (line by line)
        content = await csv_file.read()
        csv_content = io.StringIO(content.decode('utf-8'))
        print(f"CSV content length: {len(content)}")  # Debug
        print(f"CSV content preview: {content.decode('utf-8')[:200]}...")  # Debug
        results = process_csv_cell_by_cell(csv_content, jd_text)  # Process cell by cell
        print(f"Processed {len(results)} results from CSV")  # Debug

    # Load existing resumes and append new ones
    existing_resumes = load_resumes_from_json()
    existing_resumes.extend(results)
    save_resumes_to_json(existing_resumes)

    # Also store in memory for immediate access
    processed_results.extend(results)

    return {"results": results, "count": len(results)}

@app.get("/resumes")
async def get_resumes():
    resumes = load_resumes_from_json()
    return {"resumes": resumes}

@app.get("/analysis")
async def get_analysis():
    if not processed_results:
        return {"message": "No analysis results available"}

    df = pd.DataFrame(processed_results)

    # Calculate metrics
    total_candidates = len(df)
    shortlisted = len(df[df['Recommendation'] == "Advance to interview"])
    on_hold = len(df[df['Recommendation'] == "Hold"])
    rejected = len(df[df['Recommendation'] == "Reject"])
    avg_score = df['Score'].mean() if not df.empty else 0

    return {
        "total_candidates": total_candidates,
        "shortlisted": shortlisted,
        "on_hold": on_hold,
        "rejected": rejected,
        "avg_score": avg_score,
        "results": processed_results
    }

@app.post("/job-description")
async def save_job_description(jd: JobDescription):
    BASE_DIR = Path(__file__).resolve().parent
    JD_PATH = BASE_DIR / "job_description.txt"
    JD_PATH.write_text(jd.jd_text, encoding="utf-8")
    return {"message": "Job description saved"}

@app.get("/job-description")
async def get_job_description():
    BASE_DIR = Path(__file__).resolve().parent
    JD_PATH = BASE_DIR / "job_description.txt"
    if JD_PATH.exists():
        jd_text = JD_PATH.read_text(encoding="utf-8")
    else:
        jd_text = ""
    return {"jd_text": jd_text}

@app.delete("/resumes/{index}")
async def delete_resume(index: str):
    global processed_results
    if index == "all":
        processed_results.clear()
        clear_resumes_json()
        return {"message": "All resumes cleared"}
    else:
        try:
            index_int = int(index)
        except ValueError:
            raise HTTPException(status_code=404, detail="Resume index not found")

        resumes = load_resumes_from_json()
        if 0 <= index_int < len(resumes):
            deleted_resume = resumes.pop(index_int)
            save_resumes_to_json(resumes)
            # Also update in-memory list
            processed_results.clear()
            processed_results.extend(resumes)
            return {"message": f"Resume for {deleted_resume.get('Name', 'Unknown')} deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Resume index not found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
