import os
import io
import re
from collections import Counter
import importlib

def _try_import(module_name, attr=None):
    try:
        module = importlib.import_module(module_name)
        return getattr(module, attr) if attr else module
    except Exception:
        return None

def extract_emails(text):
    #Extracts email addresses from the given text.
    matches = re.findall(r'\b[\w.-]+?@\w+?\.\w+?\b',text)
    return matches

def extract_phone_numbers(text):
     #Extracts Indian phone numbers from the given text.
     matches = re.findall(r'(\+91[\s-]?)?\d{10}', text)
     return matches

# PDF extraction
pdf_extract_text = _try_import('pdfminer.high_level', 'extract_text')
PDFMINER_AVAILABLE = pdf_extract_text is not None

# OCR (PIL + pytesseract)
pytesseract = _try_import('pytesseract')
Image = _try_import('PIL.Image') or _try_import('PIL')
OCR_AVAILABLE = pytesseract is not None and Image is not None

# Word / PPTX
Document = _try_import('docx', 'Document')
DOCX_AVAILABLE = Document is not None

Presentation = _try_import('pptx', 'Presentation')
PPTX_AVAILABLE = Presentation is not None

#CSV
CSV_AVAILABLE = _try_import('csv') is not None

# pandas
pd = _try_import('pandas')

# Ollama embeddings client (optional)
ollama = _try_import('ollama')
Ollama = _try_import('ollama', 'Ollama') if ollama else None
ollama_client = None
if Ollama is not None:
    try:
        # Specify the model version explicitly
        ollama_client = Ollama(model="qwen:0.5b")
    except Exception:
        ollama_client = None
OLLAMA_AVAILABLE = ollama is not None

# numpy
np = _try_import('numpy')

RESUME_DB = []
_add_resume_to_db = _try_import('resume_db', 'add_resume_to_db')
_extract_skills = _try_import('skill_extractor', 'extract_skills')
extract_experience_years = _try_import('experience_extractor', 'extract_experience_years')
analyze_job_description = _try_import('job_description_analyzer', 'analyze_job_description')
extract_name_and_email_with_model = _try_import('name_email_extractor', 'extract_name_and_email_with_model')
analyze_resume_simple = _try_import('simple_resume_analyzer', 'analyze_resume_simple')


# JSON for LLM parsing
import json as _json

# sklearn (TF-IDF + cosine)
TfidfVectorizer = _try_import('sklearn.feature_extraction.text', 'TfidfVectorizer')
_cosine_module = _try_import('sklearn.metrics.pairwise')
cosine_similarity = getattr(_cosine_module, 'cosine_similarity', None) if _cosine_module else None
SKLEARN_AVAILABLE = TfidfVectorizer is not None and cosine_similarity is not None

# PyPDF2 (fallback PDF reader)
PyPDF2 = _try_import('PyPDF2')

# Stopwords and normalization helpers for keyword extraction
STOPWORDS_EXT = set([
    'the','and','is','in','to','with','a','an','of','for','on','by','or','as','are','be','this','that','it','we','you','your',
    'candidate','candidates','experience','preferred','basic','knowledge','seeking','welcome','freshers','years','skills',
    'technologies','technology','technical','solutions','solution','based','level','levels','must','should','required','requirements','qualification','qualifications','degree','degrees','bachelor','master','phd','postgraduate','graduate','highschool','college','university','etc','various','including','like','within','role','roles','understand','concepts'
])

def _normalize_keywords(keywords, top_n=20, source_text=None):
    if not keywords:
        return []
    source_lower = source_text.lower() if source_text else None
    normalized = []
    seen = set()
    for k in keywords:
        try:
            tok = str(k).lower().strip()
        except Exception:
            continue
        tok = re.sub(r'[\n\r]+', ' ', tok)
        tok = tok.strip(' "\'')
        tok = re.sub(r'^[^a-z0-9\+#\.\-]+|[^a-z0-9\+#\.\-]+$', '', tok)
        tok = re.sub(r'\s{2,}', ' ', tok).strip()
        if not tok:
            continue
        if len(tok) < 2 or re.fullmatch(r'\d+', tok):
            continue
        if len(tok.split()) > 3:
            continue
        if any(w in STOPWORDS_EXT for w in tok.split()):
            continue
        if ' ' in tok and source_lower and tok not in source_lower:
            continue
        if tok in STOPWORDS_EXT:
            continue
        if tok in seen:
            continue
        normalized.append(tok)
        seen.add(tok)
        if len(normalized) >= top_n:
            break
    return normalized

def get_embeddings(text, model='qwen:0.5b'):
    """
    Get embeddings for the text using Ollama if available.
    Returns a list or None on failure.
    """
    if ollama is not None and hasattr(ollama, 'embeddings'):
        response = ollama.embeddings(model=model, prompt=text)
        if response is None:
            return None
        if isinstance(response, dict):
            return response.get('embedding')
        if hasattr(response, 'embedding'):
            return list(response.embedding)
        return response

    if ollama_client is not None:
        if hasattr(ollama_client, 'embed'):
            response = ollama_client.embed(text)
        elif hasattr(ollama_client, 'embeddings'):
            response = ollama_client.embeddings(model=model, prompt=text)
        else:
            return None
        if response is None:
            return None
        if isinstance(response, dict):
            return response.get('embedding')
        if hasattr(response, 'embedding'):
            return list(response.embedding)
        return response

    return None

def calculate_similarity(resume_text, jd_text):
    """
    Calculates similarity between resume and job description using a hybrid method:
    - 60% weight to cosine similarity (TF-IDF when available)
    - 40% weight to keyword match percentage
    """

    # Step 1: Cosine similarity (use sklearn TF-IDF if available, else fallback to token overlap)
    try:
        if SKLEARN_AVAILABLE and TfidfVectorizer is not None and cosine_similarity is not None:
            vectorizer = TfidfVectorizer()
            vectors = vectorizer.fit_transform([resume_text, jd_text])
            cosine_score = float(cosine_similarity(vectors[0:1], vectors[1:2])[0][0]) * 100
        else:
            resume_set = set(resume_text.split())
            jd_set = set(jd_text.split())
            overlap = len(resume_set.intersection(jd_set))
            cosine_score = (overlap / max(len(jd_set), 1)) * 100
    except Exception:
        cosine_score = 0.0

    # Step 2: Keyword match percentage
    jd_keywords = set(jd_text.split())
    resume_words = set(resume_text.split())
    matched_keywords = jd_keywords.intersection(resume_words)
    keyword_match_percentage = (len(matched_keywords) / len(jd_keywords)) * 100 if jd_keywords else 0

    # Step 3: Weighted final score
    final_score = (0.6 * cosine_score) + (0.4 * keyword_match_percentage)
    return round(final_score, 2)

def calculate_similarity_embeddings(resume_text, jd_text, model='qwen:1.8b'):
    try:
        resume_emb = get_embeddings(resume_text, model)
        jd_emb = get_embeddings(jd_text, model)
        if not resume_emb or not jd_emb or np is None:
            return calculate_similarity(resume_text, jd_text)
        dot_product = np.dot(resume_emb, jd_emb)
        norm_resume = np.linalg.norm(resume_emb)
        norm_jd = np.linalg.norm(jd_emb)
        if norm_resume == 0 or norm_jd == 0:
            return calculate_similarity(resume_text, jd_text)
        similarity = dot_product / (norm_resume * norm_jd)
        return round(similarity * 100, 2)
    except Exception:
        return calculate_similarity(resume_text, jd_text)

def extract_keywords_with_model(text, top_n=20, model_hint=None):
    #Try an LLM first to extract keywords
    if not text:
        return []

    try:
        status = ollama_status()
        models = status.get('models', [])
        gen_model = None
        for m in models:
            if 'embedding' in m.lower():
                continue
            gen_model = m
            break

        if model_hint and model_hint in models:
            gen_model = model_hint

        def _parse_json_from_text(t):
            if not t:
                return None
            try:
                data = _json.loads(t)
                if isinstance(data, list):
                    return data
            except Exception:
                pass
            # try to find first JSON array
            m = re.search(r'\[.*?\]', t, re.S)
            if m:
                try:
                    data = _json.loads(m.group(0))
                    if isinstance(data, list):
                        return data
                except Exception:
                    return None
            return None

        if gen_model and ollama is not None and hasattr(ollama, 'generate'):
            prompt = (
                f"Extract the top {top_n} concise keywords or skills from the following job description.\n"
                "Return EXACTLY a JSON array of short strings and nothing else. Example: [\"python\",\"sql\",\"pandas\"].\n\n"
                "Job Description:\n\"\"\"\n" + text + "\n\"\"\""
            )

            resp_text = None
            parsed = None
            # Try up to 2 generation attempts; tighten prompt on second attempt
            for attempt in range(2):
                try:
                    resp = None
                    try:
                        resp = ollama.generate(model=gen_model, prompt=prompt)
                    except Exception:
                        try:
                            resp = ollama.generate(model=gen_model, messages=[{"role": "user", "content": prompt}])
                        except Exception:
                            resp = None

                    if resp is None:
                        break

                    if isinstance(resp, str):
                        resp_text = resp
                    elif hasattr(resp, 'content'):
                        resp_text = getattr(resp, 'content')
                    elif hasattr(resp, 'text'):
                        resp_text = getattr(resp, 'text')
                    else:
                        resp_text = str(resp)

                    parsed = _parse_json_from_text(resp_text)
                    if parsed and isinstance(parsed, list):
                        return _normalize_keywords(parsed, top_n=top_n, source_text=text)
                except Exception:
                    parsed = None
                # tighten prompt for second attempt
                prompt = (
                    "Again: RETURN EXACTLY a JSON array of short keyword strings and nothing else. "
                    "If you cannot, respond with an empty array `[]`. Example: [\"python\",\"sql\",\"pandas\"].\n\n"
                    "Job Description:\n\"\"\"\n" + text + "\n\"\"\""
                )

        # Fallbacks: vocabulary match + n-gram frequency
        skills = extract_skills(text)

        def ngram_freqs(s, max_n=3):
            words = [w for w in re.findall(r"\b[a-zA-Z0-9+#.+-]+\b", s.lower())]
            freqs = Counter()
            for n in range(1, max_n + 1):
                for i in range(len(words) - n + 1):
                    grams = words[i:i + n]
                    if any(g in STOPWORDS_EXT for g in grams):
                        continue
                    token = ' '.join(grams)
                    if len(token) < 2:
                        continue
                    freqs[token] += 1
            return freqs

        ng = ngram_freqs(text)
        ng_sorted = sorted(ng.items(), key=lambda x: (-x[1], -len(x[0])))
        tf_kws = [tok for tok, _ in ng_sorted]

        combined = []
        for s in skills + tf_kws:
            if s is None:
                continue
            # normalization happens in _normalize_keywords - collect raw candidates
            combined.append(s)
            if len(combined) >= top_n * 3:
                break

        return _normalize_keywords(combined, top_n=top_n, source_text=text)
    except Exception:
        skills = extract_skills(text)
        return _normalize_keywords(skills, top_n=top_n, source_text=text)

def ollama_status(model='qwen:0.5b'):
    """Return status about the Python package / client and whether a model is available."""
    status = {
        'available': ollama is not None,
        'client': ollama_client is not None,
        'model_loaded': False,
        'models': []
    }

    if ollama is None:
        status['message'] = 'ollama package is not installed'
        return status

    try:
        client = ollama_client
        models = None
        if client is not None:
            if hasattr(client, 'models'):
                models = client.models()
            elif hasattr(client, 'list_models'):
                models = client.list_models()
        if models is None and ollama is not None:
            if hasattr(ollama, 'list'):
                try:
                    models = ollama.list().models if hasattr(ollama.list(), 'models') else ollama.list()
                except Exception:
                    models = ollama.list()
            elif hasattr(ollama, 'models'):
                models = ollama.models()
            elif hasattr(ollama, 'list_models'):
                models = ollama.list_models()

        names = []
        if models:
            for m in models:
                if isinstance(m, dict):
                    names.append(m.get('name') or m.get('model') or str(m))
                elif hasattr(m, 'model'):
                    names.append(getattr(m, 'model'))
                else:
                    names.append(str(m))
            status['models'] = names
            status['model_loaded'] = model in names
    except Exception as exc:
        status['error'] = str(exc)

    return status

def score_resume(text, keywords=None):
    if not keywords:
        keywords = ["python", "machine learning", "data", "analysis", "developer", "engineer", "ai", "nlp", "deep learning"]
    text_lower = text.lower()
    matches = sum(1 for kw in keywords if kw.lower() in text_lower)
    score = int((matches / len(keywords)) * 100)
    return score

def extract_text_from_pdf(path):
    if hasattr(path, 'read'):
        import tempfile
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmpf:
                tmpf.write(path.read())
                tmp_path = tmpf.name
            if PDFMINER_AVAILABLE and pdf_extract_text is not None:
                text = pdf_extract_text(tmp_path)
            else:
                if PyPDF2 is None:
                    text = ""
                else:
                    with open(tmp_path, "rb") as fpdf:
                        reader = PyPDF2.PdfReader(fpdf)
                        text = "\n".join([p.extract_text() or "" for p in reader.pages])
        except Exception:
            text = ""
        finally:
            try:
                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
        return text
    else:
        if PDFMINER_AVAILABLE and pdf_extract_text is not None:
            return pdf_extract_text(path)
        else:
            try:
                if PyPDF2 is None:
                    raise RuntimeError("PyPDF2 is not available to extract PDF pages")
                with open(path, "rb") as fpdf:
                    reader = PyPDF2.PdfReader(fpdf)
                    return "\n".join([p.extract_text() or "" for p in reader.pages])
            except Exception:
                raise RuntimeError("PDF extraction requires 'pdfminer' or 'PyPDF2'.")

def clean_text(text):
    #Basic text cleaning and normalization like lowercasing and removinng specails characters
    text = text.lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

def extract_name(text):
 #Extracts name from the resume text
    lines = text.strip().split('\n')[:10]
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if re.search(r'@|\d|curriculum|resume|emails|phone|contact|address|summary|profile|page|cv|curriculum vitae|linkedin|github|website|mobile|tel|fax', line, re.I):
            continue
        words = line.split()
        if 1 <= len(words) <= 4:
            alpha_words = [w for w in words if re.match(r'^[A-Za-z\-]+$', w)]
            if len(alpha_words) == len(words):
                if all(w.istitle() for w in words) or all(w.isupper() for w in words):
                    if not any(word.lower() in ['mr', 'mrs', 'ms', 'dr', 'prof', 'sir', 'madam'] for word in words):
                        return line
    return "Not found"

def extract_text_from_file(uploaded_file):
    #Extrats text based on file type
    filename = uploaded_file.name.lower()
    if filename.endswith('.txt'):
        # Plain text
        return uploaded_file.read().decode('utf-8')
    elif filename.endswith('.pdf'):
        # PDF
        try:
            return extract_text_from_pdf(uploaded_file)
        except Exception as e:
            # Handle invalid PDF files
            return f"Error extracting text from PDF: {str(e)}. The file may not be a valid PDF or may be corrupted."
    elif filename.endswith(('.jpeg', '.jpg', '.png')):
        # Image - OCR
        try:
            image = Image.open(uploaded_file)
            text = pytesseract.image_to_string(image)
            return text
        except Exception as e:
            return f"Error extracting text from image: {str(e)}"
    elif filename.endswith('.docx'):
        # Word document
        try:
            doc = Document(uploaded_file)
            text = ""
            for para in doc.paragraphs:
                text += para.text + "\n"
            return text
        except Exception as e:
            return f"Error extracting text from DOCX: {str(e)}"
    elif filename.endswith('.pptx'):
        # PowerPoint
        try:
            prs = Presentation(uploaded_file)
            text = ""
            for slide in prs.slides:
                for shape in slide.shapes:
                    if hasattr(shape, "text"):
                        text += shape.text + "\n"
            return text
        except Exception as e:
            return f"Error extracting text from PPTX: {str(e)}"
    elif filename.endswith('.csv'):
        # CSV file
        try:
            if hasattr(uploaded_file, 'read'):
                content = uploaded_file.read()
                if isinstance(content, bytes):
                    content = content.decode('utf-8')
                if pd is not None:
                    df = pd.read_csv(io.StringIO(content))
                    text = ""
                    for column in df.columns:
                        text += f"{column}: {', '.join(df[column].astype(str).tolist())}\n"
                    return text
                elif CSV_AVAILABLE:
                    import csv
                    reader = csv.reader(io.StringIO(content))
                    rows = list(reader)
                    if not rows:
                        return ""
                    text = ""
                    headers = rows[0]
                    for i, header in enumerate(headers):
                        col_vals = [row[i] for row in rows[1:] if len(row) > i]
                        text += f"{header}: {', '.join(col_vals)}\n"
                    return text
                else:
                    return "CSV extraction requires pandas or csv module."
            else:
                if pd is not None:
                    df = pd.read_csv(uploaded_file)
                    text = ""
                    for column in df.columns:
                        text += f"{column}: {', '.join(df[column].astype(str).tolist())}\n"
                    return text
                elif CSV_AVAILABLE:
                    import csv
                    uploaded_file.seek(0)
                    reader = csv.reader(uploaded_file)
                    rows = list(reader)
                    if not rows:
                        return ""
                    text = ""
                    headers = rows[0]
                    for i, header in enumerate(headers):
                        col_vals = [row[i] for row in rows[1:] if len(row) > i]
                        text += f"{header}: {', '.join(col_vals)}\n"
                    return text
                else:
                    return "CSV extraction requires pandas or csv module."
        except Exception as e:
            return f"Error extracting text from CSV: {str(e)}"
    else:
        raise ValueError(f"Unsupported file type: {filename}")

def extract_certificates(text):
   #Extract certificates from resume text and matches with known keywords
    cert_keywords = ['certification', 'certificate', 'certified', 'certs', 'aws', 'azure', 'gcp', 'cisco', 'microsoft', 'google', 'oracle', 'ibm', 'comp tia', 'pmp', 'csm', 'csd', 'cissp', 'ceh', 'chfi']
    text_lower = text.lower()
    found_certs = []
    for cert in cert_keywords:
        if cert in text_lower:
            found_certs.append(cert.title())
    return list(set(found_certs))  # Remove duplicates

def extract_experience_level(text):
   #Extract experience level from job description text based on keywords
    text_lower = text.lower()
    if any(word in text_lower for word in ['junior', 'entry level', 'fresher', '0-2 years', '1-3 years']):
        return 'Junior'
    elif any(word in text_lower for word in ['mid', 'intermediate', '3-5 years', '4-6 years']):
        return 'Mid'
    elif any(word in text_lower for word in ['senior', 'lead', 'principal', '5+ years', '7+ years', '10+ years']):
        return 'Senior'
    else:
        return 'Unknown'
    

def extract_responsibilities(text):
    #Extracts core responsibilities from job description.
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    resp_section = False
    responsibilities = []
    for line in lines:
        if any(h in line.lower() for h in ["responsibilities", "duties", "what you will do", "role and responsibilities"]):
            resp_section = True
            continue
        if resp_section:
            if re.match(r'^[\-\*\u2022]', line) or len(line.split()) > 3:
                responsibilities.append(line.lstrip('-*•').strip())
            else:
                break
    return responsibilities

def extract_education_requirements(text):
    #Extracts education requirements from job description.
    edu_keywords = ['bachelor', 'master', 'phd', 'degree', 'graduate', 'postgraduate', 'diploma']
    found = []
    for kw in edu_keywords:
        if re.search(rf'\b{kw}\b', text.lower()):
            found.append(kw.title())
    return list(set(found))

def extract_industry_domain(text):
    #Attempts to extract industry or domain relevance from job description.
    industry_keywords = ['finance', 'healthcare', 'education', 'retail', 'manufacturing', 'technology', 'consulting', 'government', 'automotive', 'energy', 'telecom', 'insurance', 'media', 'logistics', 'pharma', 'biotech']
    found = []
    for kw in industry_keywords:
        if re.search(rf'\b{kw}\b', text.lower()):
            found.append(kw.title())
    return list(set(found))

def extract_achievements(text):
 #Extract Achievment from resume text based on keywords
    achievement_keywords = ['achieved', 'improved', 'increased', 'reduced', 'led', 'managed', 'delivered', 'completed', 'awarded', 'recognized', 'exceeded', 'implemented', 'optimized', 'won', 'successfully']
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    achievements = []
    for line in lines:
        if any(kw in line.lower() for kw in achievement_keywords):
            achievements.append(line)
    return achievements

def extract_education_from_resume(text):
  #Extract Education from resume text based on keywords
    edu_keywords = ['bachelor', 'master', 'phd', 'degree', 'b.sc', 'm.sc', 'b.tech', 'm.tech', 'mba', 'bachelor\'s', 'master\'s', 'ph.d', 'diploma']
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    education = []
    for line in lines:
        if any(kw in line.lower() for kw in edu_keywords):
            education.append(line)
    return education



def detect_career_gaps(text):
  #Detects career gaps of candidate
    years = []
    date_pattern = r"(?:19|20)\d{2}"

    for match in re.findall(date_pattern, text):
        try:
            years.append(int(match))
        except Exception:
            continue
    years = sorted(set(years))
    for i in range(1, len(years)):
        if years[i] - years[i-1] > 2:
            return True
    return False

def analyze_job_description_agent(text):
  #Enhanced job description analysis for agent resume ai.
    if not text:
        return {
            "responsibilities": [],
            "required_skills": [],
            "preferred_skills": [],
            "experience_years": None,
            "education_requirements": [],
            "certification_requirements": [],
            "industry_domain": [],
            "keywords": [],
            "notes": "No job description text provided."
        }
    responsibilities = extract_responsibilities(text)
    skills_info = analyze_job_description(text)
    education = extract_education_requirements(text)
    industry = extract_industry_domain(text)
    certs = extract_certificates(text)
    keywords = extract_keywords(text, top_n=20)
    return {
        "responsibilities": responsibilities,
        "required_skills": skills_info.get("required_skills", []),
        "preferred_skills": skills_info.get("preferred_skills", []),
        "experience_years": skills_info.get("experience_years"),
        "education_requirements": education,
        "certification_requirements": certs,
        "industry_domain": industry,
        "keywords": keywords,
        "notes": "Extracted for agent resume ai"
    }

def analyze_resume_agent(resume_text, jd_info):
   #Enhanced resume analysis for agent resume ai.
    # Use LLM for name and email extraction
    name_email = extract_name_and_email_with_model(resume_text)
    name = name_email.get("name", "Not found")
    emails = [name_email.get("email", "Not found")] if name_email.get("email") != "Not found" else ["Not found"]

    skills = extract_skills(resume_text)
    experience_years = extract_experience_years(resume_text)
    achievements = extract_achievements(resume_text)
    education = extract_education_from_resume(resume_text)
    certifications = extract_certificates(resume_text)
    industry = extract_industry_domain(resume_text)
    # Map to Job Description
    required_skills = jd_info.get("required_skills", [])
    preferred_skills = jd_info.get("preferred_skills", [])
    education_reqs = jd_info.get("education_requirements", [])
    cert_reqs = jd_info.get("certification_requirements", [])
    industry_reqs = jd_info.get("industry_domain", [])
    # Evidence mapping
    skill_matches = [s for s in skills if s.lower() in [r.lower() for r in required_skills]]
    preferred_matches = [s for s in skills if s.lower() in [p.lower() for p in preferred_skills]]
    education_matches = [e for e in education if any(req.lower() in e.lower() for req in education_reqs)]
    cert_matches = [c for c in certifications if any(req.lower() in c.lower() for req in cert_reqs)]
    industry_matches = [i for i in industry if any(req.lower() in i.lower() for req in industry_reqs)]
    # Gaps
    missing_required = [r for r in required_skills if r.lower() not in [s.lower() for s in skills]]
    missing_education = [e for e in education_reqs if not any(e.lower() in ed.lower() for ed in education)]
    missing_certs = [c for c in cert_reqs if not any(c.lower() in cert.lower() for cert in certifications)]
    # Career gap detection
    has_gap = detect_career_gaps(resume_text)
    return {
        "name": name,
        "skills": skills,
        "experience_years": experience_years,
        "achievements": achievements,
        "education": education,
        "certifications": certifications,
        "industry": industry,
        "emails": emails,  
        "required_skill_matches": skill_matches,
        "preferred_skill_matches": preferred_matches,
        "education_matches": education_matches,
        "certification_matches": cert_matches,
        "industry_matches": industry_matches,
        "missing_required_skills": missing_required,
        "missing_education": missing_education,
        "missing_certifications": missing_certs,
        "career_gap": has_gap
    }

def score_candidate_agent(resume_info, jd_info):
    
    # #Scores candidate per agent resume ai based on email and requirements match.
    # Weights:
    #   Required skills: 40%
    #   Relevant experience: 30%
    #   Preferred skills: 15%
    #   Achievements: 10%
    #   Education/certifications: 5%
    
    # Required skills
    req_total = len(jd_info.get("required_skills", []))
    req_matched = len(resume_info.get("required_skill_matches", []))
    req_score = (req_matched / req_total) if req_total else 0

    # Experience
    jd_exp = jd_info.get("experience_years")
    cand_exp = resume_info.get("experience_years")
    if jd_exp and cand_exp:
        exp_score = min(cand_exp / jd_exp, 1.0)
    elif jd_exp:
        exp_score = 0
    else:
        exp_score = 1.0  # No experience required

    # Preferred skills
    pref_total = len(jd_info.get("preferred_skills", []))
    pref_matched = len(resume_info.get("preferred_skill_matches", []))
    pref_score = (pref_matched / pref_total) if pref_total else 1.0

    # Achievements
    ach_score = min(len(resume_info.get("achievements", [])) / 3, 1.0) # 3+ achievements = full score

    # Education/certifications
    edu_score = 1.0
    if jd_info.get("education_requirements"):
        edu_score = 1.0 if resume_info.get("education_matches") else 0
    cert_score = 1.0
    if jd_info.get("certification_requirements"):
        cert_score = 1.0 if resume_info.get("certification_matches") else 0
    edu_cert_score = (edu_score + cert_score) / 2

    # Weighted sum
    score = (
        req_score * 0.4 +
        exp_score * 0.3 +
        pref_score * 0.15 +
        ach_score * 0.10 +
        edu_cert_score * 0.05
    ) * 100
    return round(score, 1)

def candidate_fit_label(score):
    if score >= 80:
        return "Strong"
    elif score >= 60:
        return "Moderate"
    else:
        return "Weak"

def generate_candidate_summary(resume_info, jd_info, score, resume_text=None):
    
    #Produces the required output format per candidate for agent resume ai.
    #Enhanced with RAG: Retrieves similar resumes and uses LLM for better summary generation.
    
    fit = candidate_fit_label(score)

    similar_resumes = []
    if resume_text:
        try:
            from vectors import find_similar_resumes, get_embeddings
            resume_emb = get_embeddings(resume_text)
            if resume_emb:
                similar_resumes = find_similar_resumes(resume_emb, top_k=3)
        except Exception:
            similar_resumes = []

    # Build context from similar resumes
    context = ""
    if similar_resumes:
        context = "Similar candidates from database:\n"
        for i, sim in enumerate(similar_resumes, 1):
            context += f"{i}. Candidate ID: {sim['candidate_id']}, Similarity: {sim['score']}%, Metadata: {sim['metadata']}\n"

    # Use LLM for enhanced summary generation if available
    try:
        status = ollama_status()
        models = status.get('models', [])
        gen_model = None
        for m in models:
            if 'embedding' in m.lower():
                continue
            gen_model = m
            break

        if gen_model and ollama is not None and hasattr(ollama, 'generate'):
            # Build prompt for LLM
            prompt = f"""
Based on the following resume analysis, generate a concise candidate summary for resume screening.

Resume Info:
- Name: {resume_info.get("name", "Not found")}
- Email: {resume_info.get("emails", ["Not found"])[0] if resume_info.get("emails") else "Not found"}
- Phone:{resume_info.get("phone", "Not found")}
- Skills: {', '.join(resume_info.get("skills", []))}
- Experience: {resume_info.get("experience_years", "Unknown")} years
- Education: {', '.join(resume_info.get("education", []))}
- Certifications: {', '.join(resume_info.get("certifications", []))}
- Achievements: {', '.join(resume_info.get("achievements", []))}
- Required Skill Matches: {', '.join(resume_info.get("required_skill_matches", []))}
- Preferred Skill Matches: {', '.join(resume_info.get("preferred_skill_matches", []))}
- Missing Required Skills: {', '.join(resume_info.get("missing_required_skills", []))}
- Education Matches: {', '.join(resume_info.get("education_matches", []))}
- Certification Matches: {', '.join(resume_info.get("certification_matches", []))}
- Industry Matches: {', '.join(resume_info.get("industry_matches", []))}
- Career Gap: {resume_info.get("career_gap", False)}

Job Description Requirements:
- Required Skills: {', '.join(jd_info.get("required_skills", []))}
- Preferred Skills: {', '.join(jd_info.get("preferred_skills", []))}
- Experience Required: {jd_info.get("experience_years", "Unknown")} years
- Education Requirements: {', '.join(jd_info.get("education_requirements", []))}
- Certification Requirements: {', '.join(jd_info.get("certification_requirements", []))}
- Industry/Domain: {', '.join(jd_info.get("industry_domain", []))}

{context}

Overall Fit Score: {score}/100 (Fit Label: {fit})

Generate a JSON object with the following keys:
- "Candidate Summary": A brief summary string including email
- "Email": The email
- "Phone Number"
- "Name": The name
- "Overall fit": The fit label
- "Score": The score string
- "Strengths": Array of strength points
- "Gaps / Concerns": Array of gaps (or ["None indicated"])
- "Key Evidence": Array of key evidence points
- "Recommendation": "Advance to interview", "Hold", or "Reject"
- "Rationale": Brief rationale for recommendation

Return EXACTLY the JSON object and nothing else.
"""

            resp_text = None
            for attempt in range(2):
                try:
                    resp = None
                    try:
                        resp = ollama.generate(model=gen_model, prompt=prompt)
                    except Exception:
                        try:
                            resp = ollama.generate(model=gen_model, messages=[{"role": "user", "content": prompt}])
                        except Exception:
                            resp = None

                    if resp is None:
                        break

                    if isinstance(resp, str):
                        resp_text = resp
                    elif hasattr(resp, 'content'):
                        resp_text = getattr(resp, 'content')
                    elif hasattr(resp, 'text'):
                        resp_text = getattr(resp, 'text')
                    else:
                        resp_text = str(resp)

                    # parsing JSON
                    try:
                        parsed = _json.loads(resp_text)
                        if isinstance(parsed, dict) and all(k in parsed for k in ["Candidate Summary", "Email", "Name", "Overall fit", "Score", "Strengths", "Gaps / Concerns", "Key Evidence", "Recommendation", "Rationale"]):
                            return parsed
                    except Exception:
                        pass

                    # Try to find JSON object in text
                    m = re.search(r'\{.*?\}', resp_text, re.S)
                    if m:
                        try:
                            parsed = _json.loads(m.group(0))
                            if isinstance(parsed, dict) and all(k in parsed for k in ["Candidate Summary", "Email", "Name", "Overall fit", "Score", "Strengths", "Gaps / Concerns", "Key Evidence", "Recommendation", "Rationale"]):
                                return parsed
                        except Exception:
                            pass
                except Exception:
                    pass
                # Tighten prompt on second attempt
                prompt = "Again: RETURN EXACTLY a JSON object with the specified keys. Nothing else.\n\n" + prompt
    except Exception:
        pass

    # Fallback to rule-based summary
    strengths = []
    for s in resume_info.get("required_skill_matches", []):
        strengths.append(f"Matched required skill: {s}")
    for s in resume_info.get("preferred_skill_matches", []):
        strengths.append(f"Matched preferred skill: {s}")
    if resume_info.get("achievements"):
        strengths.append(f"Achievements: {', '.join(resume_info['achievements'][:2])}")
    if resume_info.get("education_matches"):
        strengths.append(f"Education: {', '.join(resume_info['education_matches'])}")
    if resume_info.get("certification_matches"):
        strengths.append(f"Certifications: {', '.join(resume_info['certification_matches'])}")
    if resume_info.get("industry_matches"):
        strengths.append(f"Industry/domain: {', '.join(resume_info['industry_matches'])}")

    gaps = []
    if resume_info.get("missing_required_skills"):
        gaps.append(f"Missing required skills: {', '.join(resume_info['missing_required_skills'])}")
    if resume_info.get("missing_education"):
        gaps.append(f"Missing education: {', '.join(resume_info['missing_education'])}")
    if resume_info.get("missing_certifications"):
        gaps.append(f"Missing certifications: {', '.join(resume_info['missing_certifications'])}")
    if resume_info.get("career_gap"):
        gaps.append("Career gap detected (flagged neutrally)")

    key_evidence = []
    for s in resume_info.get("required_skill_matches", [])[:2]:
        key_evidence.append(f"Skill: {s}")
    for a in resume_info.get("achievements", [])[:1]:
        key_evidence.append(f"Achievement: {a}")
    for e in resume_info.get("education_matches", [])[:1]:
        key_evidence.append(f"Education: {e}")

    # Recommendation logic
    if fit == "Strong" and not gaps:
        recommendation = "Advance to interview"
        rationale = "Candidate meets or exceeds all key requirements based on resume evidence."
    elif fit == "Moderate":
        recommendation = "Hold"
        rationale = "Candidate meets most requirements but has some gaps or weaker areas."
    else:
        recommendation = "Reject"
        rationale = "Candidate lacks several required qualifications or has weak alignment."

    # Use name and emails from resume_info (extracted by LLM)
    name = resume_info.get("name", "Not found")
    emails = resume_info.get("emails", ["Not found"])
    email_str = ", ".join(emails) if emails else "Not found"
    summary = {
        "Candidate Summary": f"Emails: {email_str}",  # <-- Show emails in summary string
        "Emails": emails,
        "Name": name,
        "Overall fit": fit,
        "Score": f"{score} / 100",
        "Strengths": strengths,
        "Gaps / Concerns": gaps if gaps else ["None indicated"],
        "Key Evidence": key_evidence,
        "Recommendation": recommendation,
        "Rationale": rationale
    }
    return summary

def analyze_job_description(text):
    """
    Basic job description analysis for required/preferred skills and experience.
    Used internally by agent resume ai.
    """
    required_skills = extract_skills(text)
    preferred_skills = []  # Could be improved by section parsing or methods
    experience_years = extract_experience_years(text)
    return {
        "required_skills": required_skills,
        "preferred_skills": preferred_skills,
        "experience_years": experience_years,
    }
def extract_skills(text):
    """
    Extracts skills from text using a simple keyword list.
    """
    skill_keywords = [
        "python", "java", "c++", "sql", "excel", "machine learning", "deep learning", "nlp",
        "data analysis", "data science", "pandas", "numpy", "tensorflow", "keras", "pytorch",
        "scikit-learn", "aws", "azure", "gcp", "docker", "kubernetes", "linux", "git",
        "javascript", "html", "css", "react", "node.js", "angular", "tableau", "power bi",
        "project management", "communication", "leadership", "problem solving"
    ]
    text_lower = text.lower()
    found_skills = []
    for skill in skill_keywords:
        if skill in text_lower:
            found_skills.append(skill.title())
    return list(set(found_skills))

def extract_experience_years(text):
    """
    Extracts years of experience from text using regex.
    Returns the maximum number found, or None.
    """
    matches = re.findall(r'(\d+)\s*\+?\s*(?:years|yrs|year)', text.lower())
    years = [int(m) for m in matches if m.isdigit()]
    return max(years) if years else None

def extract_keywords(text, top_n=20):
    #Extracts top N keywords from text by frequency, ignoring stopwords.
    stopwords = set([
        "the", "and", "to", "of", "in", "a", "for", "with", "on", "is", "as", "by", "an", "be",
        "are", "at", "from", "or", "that", "this", "will", "can", "have", "has", "it", "if", "not"
    ])
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    words = [w for w in words if w not in stopwords]
    freq = Counter(words)
    return [w for w, _ in freq.most_common(top_n)]

def extract_texts_from_directory(directory_path):
    """
    Extracts text from all supported resume files in the given directory.
    Returns a list of dicts: [{'filename': ..., 'text': ...}, ...]
    """
    supported_exts = ('.txt', '.pdf', '.jpeg', '.jpg', '.png', '.docx', '.pptx', '.csv')
    results = []
    for filename in os.listdir(directory_path):
        if filename.lower().endswith(supported_exts):
            file_path = os.path.join(directory_path, filename)
            try:
                ext = filename.lower().split('.')[-1]
                if ext in ['pdf', 'docx', 'pptx', 'csv', 'txt']:
                    mode = 'rb' if ext != 'txt' else 'r'
                    with open(file_path, mode) as f:
                        if ext == 'txt':
                            class TxtFileWrapper:
                                def __init__(self, f, name):
                                    self.f = f
                                    self.name = name
                                def read(self):
                                    return self.f.read().encode('utf-8')
                            uploaded_file = TxtFileWrapper(f, filename)
                        else:
                            f.name = filename
                            uploaded_file = f
                        text = extract_text_from_file(uploaded_file)
                elif ext in ['jpeg', 'jpg', 'png']:
                    with open(file_path, 'rb') as f:
                        f.name = filename
                        text = extract_text_from_file(f)
                else:
                    continue
                results.append({'filename': filename, 'text': text})
            except Exception as e:
                results.append({'filename': filename, 'text': f"Error: {str(e)}"})
    return results

def extract_name_and_email_with_model(text, model_hint=None):
   
    if not text:
        return {"name": "Not found", "email": "Not found"}

    try:
        status = ollama_status()
        models = status.get('models', [])
        gen_model = None
        for m in models:
            if 'embedding' in m.lower():
                continue
            gen_model = m
            break

        if model_hint and model_hint in models:
            gen_model = model_hint

        if gen_model and ollama is not None and hasattr(ollama, 'generate'):
            prompt = (
                "Extract the candidate's full name and email address from the following resume text.\n"
                "Return EXACTLY a JSON object with keys 'name' and 'email'. If not found, use 'Not found'.\n"
                "Example: {\"name\": \"John Doe\", \"email\": \"john.doe@example.com\"}\n\n"
                "Resume Text:\n\"\"\"\n" + text[:2000] + "\n\"\"\""  # Limit text to avoid token limits
            )

            resp_text = None
            parsed = None
            for attempt in range(2):
                try:
                    resp = None
                    try:
                        resp = ollama.generate(model=gen_model, prompt=prompt)
                    except Exception:
                        try:
                            resp = ollama.generate(model=gen_model, messages=[{"role": "user", "content": prompt}])
                        except Exception:
                            resp = None

                    if resp is None:
                        break

                    if isinstance(resp, str):
                        resp_text = resp
                    elif hasattr(resp, 'content'):
                        resp_text = getattr(resp, 'content')
                    elif hasattr(resp, 'text'):
                        resp_text = getattr(resp, 'text')
                    else:
                        resp_text = str(resp)

                    # Try to parse JSON
                    try:
                        parsed = _json.loads(resp_text)
                        if isinstance(parsed, dict) and 'name' in parsed and 'email' in parsed:
                            return parsed
                    except Exception:
                        pass

                    # Try to find JSON object in text
                    m = re.search(r'\{.*?\}', resp_text, re.S)
                    if m:
                        try:
                            parsed = _json.loads(m.group(0))
                            if isinstance(parsed, dict) and 'name' in parsed and 'email' in parsed:
                                return parsed
                        except Exception:
                            pass
                except Exception:
                    parsed = None
                # Tighten prompt on second attempt
                prompt = (
                    "Again: RETURN EXACTLY a JSON object with 'name' and 'email' keys. Nothing else.\n"
                    "Example: {\"name\": \"John Doe\", \"email\": \"john.doe@example.com\"}\n\n"
                    "Resume Text:\n\"\"\"\n" + text[:2000] + "\n\"\"\""
                )

        # Fallback to rule-based extraction
        name = extract_name(text)
        emails = extract_emails(text)
        email = emails[0] if emails else "Not found"
        return {"name": name, "email": email}
    except Exception:
        # Final fallback
        name = extract_name(text)
        emails = extract_emails(text)
        email = emails[0] if emails else "Not found"
        return {"name": name, "email": email}

def analyze_resume_simple(resume_text, jd_info_str):
    """
    Resume screening using qwen:0.5b model.
    Scans resume to extract skills, then matches against job description requirements.
    Returns a dictionary with analysis results.
    """
    try:
        if ollama_client is not None:
            # Step 1: Scan resume to extract skills and experience
            resume_scan_prompt = f"""
First, carefully scan this resume and extract the candidate's skills, experience, and qualifications.

Resume:
{resume_text[:2000]}

Extract and list:
- Technical skills and technologies mentioned
- Years of experience (if stated)
- Key qualifications and certifications
- Education background
- Notable achievements or projects

Return your extraction as a structured summary.
"""

            resume_response = ollama_client(resume_scan_prompt)
            if hasattr(resume_response, 'content'):
                resume_analysis = resume_response.content
            elif isinstance(resume_response, str):
                resume_analysis = resume_response
            else:
                resume_analysis = str(resume_response)

            # Step 2: Match extracted skills against job requirements
            matching_prompt = f"""
Now, compare the candidate's skills and qualifications against this job description and provide a JSON response with the following keys:
- score: A number from 0-100 indicating fit based on skill matching
- overall_fit: "Strong", "Moderate", or "Weak"
- strengths: Array of 2-3 key strengths from the resume
- gaps: Array of 1-2 main gaps or missing skills
- evidence: Array of 2-3 key pieces of evidence from resume
- recommendation: "Advance to interview", "Hold", or "Reject"
- rationale: Brief explanation for the recommendation

Candidate's Skills & Experience:
{resume_analysis}

Job Description Requirements:
{jd_info_str}

Return only JSON, no other text.
"""

            response = ollama_client(matching_prompt)
            if hasattr(response, 'content'):
                result_text = response.content
            elif isinstance(response, str):
                result_text = response
            else:
                result_text = str(response)

            # Try to parse JSON
            try:
                result = _json.loads(result_text)
                return result
            except:
                # Try to extract JSON from text
                import re
                json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
                if json_match:
                    try:
                        result = _json.loads(json_match.group(0))
                        return result
                    except:
                        pass

        # Fallback to basic keyword matching
        jd_lower = jd_info_str.lower()
        resume_lower = resume_text.lower()

        # Extract skills from resume using simple keyword matching
        skill_keywords = [
            "python", "java", "c++", "sql", "excel", "machine learning", "deep learning", "nlp",
            "data analysis", "data science", "pandas", "numpy", "tensorflow", "keras", "pytorch",
            "scikit-learn", "aws", "azure", "gcp", "docker", "kubernetes", "linux", "git",
            "javascript", "html", "css", "react", "node.js", "angular", "tableau", "power bi",
            "project management", "communication", "leadership", "problem solving"
        ]

        resume_skills = [skill for skill in skill_keywords if skill in resume_lower]
        jd_skills = [skill for skill in skill_keywords if skill in jd_lower]

        matched_skills = set(resume_skills).intersection(set(jd_skills))
        match_ratio = len(matched_skills) / len(jd_skills) if jd_skills else 0
        score = min(100, int(match_ratio * 100))

        if score >= 70:
            overall_fit = "Strong"
            recommendation = "Advance to interview"
        elif score >= 50:
            overall_fit = "Moderate"
            recommendation = "Hold"
        else:
            overall_fit = "Weak"
            recommendation = "Reject"

        return {
            "score": score,
            "overall_fit": overall_fit,
            "strengths": [f"Matched skills: {', '.join(list(matched_skills)[:3])}"] if matched_skills else ["Some relevant experience"],
            "gaps": [f"Missing key skills: {', '.join(list(set(jd_skills) - matched_skills)[:2])}"] if jd_skills else ["Limited skill analysis"],
            "evidence": [f"Found {len(matched_skills)} matching skills out of {len(jd_skills)} required"] if jd_skills else ["Basic skill matching performed"],
            "recommendation": recommendation,
            "rationale": f"Score based on {len(matched_skills)}/{len(jd_skills)} skill matches with job requirements"
        }

    except Exception as e:
        # Final fallback
        return {
            "score": 50,
            "overall_fit": "Unknown",
            "strengths": ["Analysis completed"],
            "gaps": ["Error in analysis"],
            "evidence": ["Basic processing"],
            "recommendation": "Hold",
            "rationale": f"Analysis error: {str(e)}"
        }

def extract_resumes_from_csv(csv_path, resume_column='resume'):
    """
    Reads a CSV file where each row is a resume (or contains resume text in a column).
    Returns a list of dicts: [{...row..., 'resume_text': ...}, ...]
    If resume_column is not found, tries to use the first column.
    """
    resumes = []
    if pd is not None:
        df = pd.read_csv(csv_path)
        if resume_column not in df.columns:
            resume_column = df.columns[0]
        for _, row in df.iterrows():
            row_dict = row.to_dict()
            row_dict['resume_text'] = str(row[resume_column])
            resumes.append(row_dict)
    elif CSV_AVAILABLE:
        import csv
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            columns = reader.fieldnames
            if resume_column not in columns:
                resume_column = columns[0]
            for row in reader:
                row['resume_text'] = row.get(resume_column, '')
                resumes.append(row)
    else:
        raise RuntimeError("CSV extraction requires pandas or csv module.")
    return resumes




