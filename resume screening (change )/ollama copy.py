import os
import io
import re
from datetime import datetime
from collections import Counter

# Optional dependencies: import when available, otherwise provide fallbacks or flags
import importlib

def _try_import(module_name, attr=None):
    try:
        module = importlib.import_module(module_name)
        return getattr(module, attr) if attr else module
    except Exception:
        return None

# sklearn (TF-IDF + cosine)
TfidfVectorizer = _try_import('sklearn.feature_extraction.text', 'TfidfVectorizer')
_cosine_module = _try_import('sklearn.metrics.pairwise')
cosine_similarity = getattr(_cosine_module, 'cosine_similarity', None) if _cosine_module else None
SKLEARN_AVAILABLE = TfidfVectorizer is not None and cosine_similarity is not None

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

# pandas
pd = _try_import('pandas')

# Ollama embeddings client (optional)
ollama = _try_import('ollama')
Ollama = _try_import('ollama', 'Ollama') if ollama else None
ollama_client = None
if Ollama is not None:
    try:
        ollama_client = Ollama()
    except Exception:
        ollama_client = None
OLLAMA_AVAILABLE = ollama is not None

# numpy
np = _try_import('numpy')

# PyPDF2 (fallback PDF reader)
PyPDF2 = _try_import('PyPDF2')


def extract_text_from_pdf(path):
    if hasattr(path, 'read'):  # For Streamlit uploaded file-like objects
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
                    # Couldn't extract PDF pages without PyPDF2
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
    text = text.lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

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

def extract_email(text):
    pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', re.IGNORECASE)
    match = pattern.search(text or "")
    return match.group(0) if match else "Not found"

def extract_name(text):
    lines = text.strip().split('\n')
    for line in lines[:10]:  # Check first 10 lines only
        line = line.strip()
        words = line.split()
        if 1 < len(words) <= 5 and all(w.istitle() or w.isupper() for w in words):
            return line
    return "Not found"

def extract_text_from_file(uploaded_file):
    """
    Extracts text from various file types: txt, pdf, jpeg, png, docx, pptx, csv.
    """
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
            # For Streamlit uploaded file, we need to read the content first
            if hasattr(uploaded_file, 'read'):
                content = uploaded_file.read()
                if isinstance(content, bytes):
                    content = content.decode('utf-8')
                df = pd.read_csv(io.StringIO(content))
            else:
                df = pd.read_csv(uploaded_file)
            # Convert DataFrame to text representation
            text = ""
            for column in df.columns:
                text += f"{column}: {', '.join(df[column].astype(str).tolist())}\n"
            return text
        except Exception as e:
            return f"Error extracting text from CSV: {str(e)}"
    else:
        raise ValueError(f"Unsupported file type: {filename}")

def extract_certificates(text):
    """
    Extract certificates from resume text.
    Looks for common certificate keywords.
    """
    cert_keywords = ['certification', 'certificate', 'certified', 'certs', 'aws', 'azure', 'gcp', 'cisco', 'microsoft', 'google', 'oracle', 'ibm', 'comp tia', 'pmp', 'csm', 'csd', 'cissp', 'ceh', 'chfi']
    text_lower = text.lower()
    found_certs = []
    for cert in cert_keywords:
        if cert in text_lower:
            found_certs.append(cert.title())
    return list(set(found_certs))  # Remove duplicates

def extract_experience_level(text):
    """
    Extract experience level from resume text.
    Categorizes as Junior, Mid, Senior based on keywords.
    """
    text_lower = text.lower()
    if any(word in text_lower for word in ['junior', 'entry level', 'fresher', '0-2 years', '1-3 years']):
        return 'Junior'
    elif any(word in text_lower for word in ['mid', 'intermediate', '3-5 years', '4-6 years']):
        return 'Mid'
    elif any(word in text_lower for word in ['senior', 'lead', 'principal', '5+ years', '7+ years', '10+ years']):
        return 'Senior'
    else:
        return 'Unknown'

def extract_keywords(text, top_n=20):
    
    try:
        
        kws = []
        if SKLEARN_AVAILABLE and TfidfVectorizer is not None:
            # prefer 1-2 grams and use english stopwords; limit features to twice the requested keywords to allow filtering
            vectorizer = TfidfVectorizer(stop_words='english', max_features=top_n * 2, ngram_range=(1, 2))
            tfidf_matrix = vectorizer.fit_transform([text])
            feature_names = vectorizer.get_feature_names_out()
            scores = tfidf_matrix.toarray()[0]
            keywords = sorted(zip(feature_names, scores), key=lambda x: x[1], reverse=True)
            kws = [kw for kw, score in keywords]
        else:
            # fallback: simple frequency over tokens and bi-grams
            words = [w for w in re.findall(r"\b[a-zA-Z0-9+#\.\-]{2,}\b", text.lower()) if len(w) > 1]
            common = Counter(words).most_common(top_n * 2)
            kws = [w for w, _ in common]
        return _normalize_keywords(kws, top_n=top_n, source_text=text)
    except Exception:
        return []

# Default skills list used for keyword matching. Extend as needed.
DEFAULT_SKILLS = [
    "python","java","c++","c#","javascript","typescript","sql","nosql","postgresql","mysql","mongodb",
    "docker","kubernetes","aws","azure","gcp","linux","git","spark","hadoop","pandas","numpy","scipy",
    "scikit-learn","sklearn","tensorflow","pytorch","keras","react","angular","vue","node.js","express",
    "django","flask","rest","graphql","html","css","bootstrap","tailwind","excel","powerpoint","word",
    "tableau","power bi","powerbi","matlab","r","sas","ansible","terraform","jenkins","ci/cd","api",
    "microservices","nlp","natural language processing","machine learning","deep learning","data science",
    "project management","sql server","cassandra","redis","elasticsearch","jira","confleunce","scrum"
]

# Stopwords and normalization helpers for keyword extraction
STOPWORDS_EXT = set([
    'the','and','is','in','to','with','a','an','of','for','on','by','or','as','are','be','this','that','it','we','you','your',
    'candidate','candidates','experience','preferred','basic','knowledge','seeking','welcome','freshers','years','skills',
    'technologies','technology','technical','solutions','solution','based','level','levels','must','should','required','requirements','qualification','qualifications','degree','degrees','bachelor','master','phd','postgraduate','graduate','highschool','college','university','etc','various','including','like','within','role','roles','understand','concepts'
])

import json as _json  # localized JSON helper for parsing LLM output

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
        # strip surrounding punctuation but allow +, #, ., - inside tokens
        tok = re.sub(r'^[^a-z0-9\+#\.\-]+|[^a-z0-9\+#\.\-]+$', '', tok)
        tok = re.sub(r'\s{2,}', ' ', tok).strip()
        if not tok:
            continue
        # drop tokens that are too short or purely numeric
        if len(tok) < 2 or re.fullmatch(r'\d+', tok):
            continue
        # drop long 'sentence-like' tokens (max 3 words)
        if len(tok.split()) > 3:
            continue
        # drop tokens containing any stopword as a part word (e.g., 'understand data cleaning')
        if any(w in STOPWORDS_EXT for w in tok.split()):
            continue
        # if multi-word token, require verbatim occurrence in source text (if provided)
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


def extract_skills(text, skills_list=None, top_n=20):
    
    if not text:
        return []
    text_norm = re.sub(r"[\W_]+", " ", text.lower())
    if skills_list is None:
        skills_list = DEFAULT_SKILLS
    found = []
    for skill in skills_list:
        skill_norm = re.sub(r"[\W_]+", " ", skill.lower()).strip()
        m = re.search(r"\b" + re.escape(skill_norm) + r"\b", text_norm)
        if m:
            found.append((skill, m.start()))
    found_sorted = sorted(found, key=lambda x: x[1])
    skills = []
    seen = set()
    for s, _ in found_sorted:
        key = s.lower()
        if key not in seen:
            skills.append(s)
            seen.add(key)
        if len(skills) >= top_n:
            break
    return skills

def extract_career_start(text):
   
    # Use non-capturing groups for full-year matches and capture group for Month YYYY to extract year
    date_patterns = [
        r"\b(?:19|20)\d{2}\b",  # YYYY
        r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+((?:19|20)\d{2})\b",  # Month YYYY (capture year)
    ]
    text_lower = text.lower()
    years = []
    for pattern in date_patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            # match will be a string (the year) for these patterns or a captured group
            try:
                year = int(match)
            except Exception:
                # Fallback: try to extract a 4-digit year from the match
                m = re.search(r"(19|20)\d{2}", str(match))
                if not m:
                    continue
                year = int(m.group(0))
            if 1950 <= year <= datetime.now().year:
                years.append(year)
    return min(years) if years else None

def get_embeddings(text, model='qwen:0.5b'):
    """
    Get embeddings for the text using Ollama if available.
    Returns a list or None on failure.

    Uses the module-level `ollama.embeddings` API where available (this has worked reliably in this environment).
    """
    # Prefer module-level API (works with the current installed ollama)
    if ollama is not None and hasattr(ollama, 'embeddings'):
        response = ollama.embeddings(model=model, prompt=text)
        if response is None:
            return None
        if isinstance(response, dict):
            return response.get('embedding')
        if hasattr(response, 'embedding'):
            return list(response.embedding)
        return response

    # As a final fallback, try client instance if present
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


def calculate_similarity_embeddings(resume_text, jd_text, model='qwen:0.5b'):
    
   
    try:
        resume_emb = get_embeddings(resume_text, model)
        jd_emb = get_embeddings(jd_text, model)
        if not resume_emb or not jd_emb or np is None:
            # Fallback
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
        # Try client methods first
        if client is not None:
            if hasattr(client, 'models'):
                models = client.models()
            elif hasattr(client, 'list_models'):
                models = client.list_models()
        # Fallback to module-level functions
        if models is None and ollama is not None:
         # Newer module exposes list() and embeddings/embeds
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


def extract_keywords_with_model(text, top_n=20, model_hint=None):
    """Try an LLM first (ask for strict JSON array), with one retry. If that fails use vocabulary + n-gram frequency fallback.
    Results are normalized and deduplicated by `_normalize_keywords`.
    """
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

        # Helper to attempt parse from model output
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