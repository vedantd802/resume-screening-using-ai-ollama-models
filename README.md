# AI Resume Screening Tool

An AI-powered Resume Screening system that analyzes resumes against a Job Description (JD) and calculates a relevance score using NLP and similarity techniques.  

## 🚀 Features


- 📄 Supports multiple resume formats:
  - PDF
  - DOCX
  - TXT
  - Images (OCR)
  - CSV
  - PPTX
- 🔍 Extracts key information:
  - Candidate name
  - Email ID
  - Phone number
  - Skills
  - Experience (basic)
- 📊 Resume–Job Description matching:
  - TF-IDF + Cosine Similarity(For Hybrid Matching /If Ollama Fails Model Still Runs)
  - Keyword overlap scoring
- 🤖 Optional LLM-based keyword extraction (Ollama)
- 🧠 Fallback logic if AI/ML libraries are unavailable
- 🧩 Modular architecture for easy refactoring and scaling

---
## 🚀 Features

- User authentication with **JWT tokens**  
- Upload resumes individually or in batch (`PDF` / `DOCX`)  
- Drag & drop file upload support  
- View candidate analysis results with scores  
- Filter shortlisted candidates only  
- Sort results by score  
- Configurable batch size and parallel processing  
- FastAPI backend with **asynchronous processing**  
- React frontend with responsive UI  

---

## 🧰 Tech Stack

- **Frontend:** React, Axios, Tailwind CSS (optional)  
- **Backend:** Python, FastAPI, Uvicorn  
- **Authentication:** JWT  
- **Database / Storage:** (Optional – can use file system or database)  
- **AI/ML:** Resume scoring / NLP-based analysis  
- **Other:** npm for frontend package management, Redis (optional) for caching  

---

## 📦 Installation

### Backend (FastAPI)

1. Clone the repo:
```bash
git clone <repo-url>
cd resume-screener-backend


## 📁 Project Structure

resume-screener/
│
├─ backend/
│  ├─ main.py
│  ├─ api/
│  ├─ models/
│  └─ requirements.txt
│
├─ frontend/
│  ├─ src/
│  │  ├─ components/
│  │  ├─ pages/
│  │  └─ App.js
│  └─ package.json
│
└─ README.md

---

## ⚙️ How It Works

1. Resume text is extracted based on file type (PDF, DOCX, Image, etc.)
2. Text is cleaned and normalized
3. Important keywords and skills are extracted
4. Resume content is compared with the Job Description
5. A final relevance score is generated based on similarity

---

## 🧪 Similarity Logic

- **Ollama/TF-IDF(Optional) + Cosine Similarity **
- ** Keyword match percentage **

---

## 👨‍💻 Author

**Vedant Deshmukh**  
B.Tech Computer Science Engineering  
Intern – Artificial Intelligence & Data Science  

---

## ⚠️ Disclaimer

This project is built for learning and internship purposes.  
All logic is implemented with a clear understanding and can be modified as per organizational requirements
