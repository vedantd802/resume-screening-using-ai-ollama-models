# AI Resume Screening Tool

An AI-powered Resume Screening system that analyzes resumes against a Job Description (JD) and calculates a relevance score using NLP and similarity techniques.  

---

## 🚀 Features

- 📄 **Supports multiple resume formats**:
  - PDF, DOCX, TXT, Images (OCR), CSV.
- 🔍 **Extracts key information**:
  - Candidate name, Email ID, Phone number, Skills, Experience (basic)
- 📊 **Resume–Job Description matching**:
  - TF-IDF + Cosine Similarity (fallback if Ollama fails)
  - Keyword overlap scoring
- 🤖 **Optional LLM-based keyword extraction** (Ollama)
- 🧠 **Fallback logic** if AI/ML libraries are unavailable
- 🧩 **Modular architecture** for easy refactoring and scaling
- 🔑 **User authentication** with JWT tokens
- 📤 Upload resumes individually or in batch (PDF / DOCX)
- 🎯 Drag & drop file upload support
- 📈 View candidate analysis results with scores
- ✅ Filter shortlisted candidates only
- 🔢 Sort results by score
- ⚙️ Configurable batch size and parallel processing
- ⚡ FastAPI backend with asynchronous processing
- 🖥 Responsive React frontend UI

---

## 🧰 Tech Stack

- **Frontend:** React, Axios, Tailwind CSS (optional)  
- **Backend:** Python, FastAPI, Uvicorn  
- **Authentication:** JWT  
- **Database / Storage:** Optional (file system or database)  
- **AI/ML:** Resume scoring / NLP-based analysis  
- **Other:** npm for frontend package management, Redis (optional) for caching  

---

## 📦 Installation

### Backend (FastAPI)

1. Clone the repository:
```bash
git clone <https://github.com/vedantd802/resume-screening-using-ai-ollama-models.git>
cd resume-screener-backend

---

## 👨‍💻 Author

**Vedant Deshmukh**  
B.Tech Computer Science Engineering  
Intern – Artificial Intelligence & Data Science  

---

## ⚠️ Disclaimer

This project is built for learning and internship purposes.  
All logic is implemented with a clear understanding and can be modified as per organizational requirements
