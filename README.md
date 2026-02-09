AI Resume Screening Tool

An AI-powered Resume Screening system that analyzes candidate resumes against a Job Description (JD) and calculates a relevance score using NLP, embeddings, and similarity techniques. The system is designed to be modular, scalable, and suitable for real-world hiring workflows.

🎯 Problem Statement

Manual resume screening is time-consuming, inconsistent, and error-prone. Recruiters often struggle to quickly identify the most relevant candidates from a large pool of resumes.

This project automates resume screening by:

Extracting structured information from resumes

Comparing resumes against a job description using semantic similarity

Generating relevance scores to assist shortlisting decisions

🚀 Features

📄 Supports multiple resume formats

PDF, DOCX, TXT, Images (OCR), CSV

🔍 Information extraction

Name, Email, Phone number, Skills, Experience (basic)

📊 Resume–Job Description matching

Embedding-based semantic similarity

TF-IDF + cosine similarity fallback

Keyword overlap scoring

🤖 Optional LLM-based keyword extraction (Ollama)

🧠 Robust fallback logic if AI models are unavailable

🧩 Modular architecture for maintainability

🔑 JWT-based authentication

📤 Single & batch resume uploads

🎯 Drag & drop file upload

📈 Resume scoring & ranking

✅ Candidate shortlisting & filtering

⚡ Asynchronous FastAPI backend

🖥 Responsive React frontend

🏗️ System Architecture
Frontend (React)
   ↓
FastAPI Backend
   ↓
Resume Parser → Text Cleaning
   ↓
Embedding / NLP Scoring
   ↓
Similarity Calculation
   ↓
Final Resume Score

🧠 How the Scoring Works

Resume Parsing

Text is extracted from resumes using format-specific parsers.

Text Preprocessing

Cleaning, normalization, and tokenization.

Embedding Generation

Resumes and Job Descriptions are converted into vector embeddings.

Similarity Calculation

Cosine similarity is used to measure semantic relevance.

TF-IDF and keyword overlap act as fallback mechanisms.

Final Score

A weighted score is generated and used to rank candidates.

❓ Why RAG Is Not Used

This project does not use RAG (Retrieval-Augmented Generation) because:

Resume screening is a direct comparison problem, not a knowledge retrieval problem.

There is no external document corpus to retrieve from.

Embedding-based similarity is more efficient, interpretable, and cost-effective.

RAG would be suitable only if resumes needed to be compared against a large external knowledge base.

🧰 Tech Stack

Frontend: React, Axios, Tailwind CSS

Backend: Python, FastAPI, Uvicorn

Authentication: JWT

AI/NLP: Embeddings, TF-IDF, similarity scoring

Storage: File system / optional database

Caching (optional): Redis

📦 Installation
Backend (FastAPI)
git clone https://github.com/vedantd802/resume-screening-using-ai-ollama-models.git
cd resume-screening-using-ai-ollama-models
pip install -r requirements.txt
uvicorn main:app --reload

Frontend (React)
cd frontend
npm install
npm start

📊 Sample Output
Candidate	Similarity Score	Status
John Doe	0.82	Shortlisted
Jane Smith	0.67	Review
Alex Ray	0.45	Rejected
🔮 Future Improvements

Resume explanation using LLMs

Skill-level extraction (beginner/intermediate/expert)

Recruiter dashboard analytics

Cloud deployment (AWS / Azure)

Multi-job comparison support

👨‍💻 Author

Vedant Deshmukh
B.Tech Computer Science Engineering
Intern – Artificial Intelligence & Data Science

⚠️ Disclaimer

This project is built for learning and internship purposes.
All logic is transparent, modular, and can be adapted for enterprise use.
