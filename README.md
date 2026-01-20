# AI Resume Screening Tool

An AI-powered Resume Screening system that analyzes resumes against a Job Description (JD) and calculates a relevance score using NLP and similarity techniques.  
The project is designed with a modular backend-first approach and can be extended into a full production-grade API.

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

## 🛠️ Tech Stack

- **Language:** Python  
- **Core Libraries:**
  - scikit-learn
  - numpy
  - pandas
  - pdfminer / PyPDF2
  - pytesseract (OCR)
- **AI / LLM **
  - Ollama (Qwen:0.5b)(local LLM)// According To Parameter
- **Architecture:**
  - Modular Python backend
  - API-ready (FastAPI compatible)

---

## 📁 Project Structure


---

## ⚙️ How It Works

1. Resume text is extracted based on file type (PDF, DOCX, Image, etc.)
2. Text is cleaned and normalized
3. Important keywords and skills are extracted
4. Resume content is compared with the Job Description
5. A final relevance score is generated based on similarity

---

## 🧪 Similarity Logic

- **60% weight:** Ollama/TF-IDF(Optional) + Cosine Similarity  
- **40% weight:** Keyword match percentage  

This hybrid approach improves accuracy compared to keyword-only matching.

---

## 🔮 Future Enhancements

- Convert to FastAPI-based REST service
- Database integration for resume storage
- Improved experience extraction
- Model-based classification (Selected / Rejected)
- Admin dashboard for recruiters
- Authentication & role-based access

---

## 👨‍💻 Author

**Vedant Deshmukh**  
B.Tech Computer Science Engineering  
Intern – Artificial Intelligence & Data Science  

---

## ⚠️ Disclaimer

This project is built for learning and internship purposes.  
AI-assisted development tools were used, and the logic is fully understood and modifiable.
