import streamlit as st
import pandas as pd
import pymupdf
import re
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import io
import base64

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="ResumeIQ - AI Resume Analyzer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- CUSTOM CSS ----------
st.markdown("""
<style>
    /* Main background and text */
    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0;
    }
    
    .sub-header {
        color: #a0a0a0;
        text-align: center;
        font-size: 1.1rem;
        margin-top: -10px;
    }
    
    /* Card styling */
    .metric-card {
        background: linear-gradient(145deg, #1e1e30 0%, #2a2a40 100%);
        border-radius: 16px;
        padding: 24px;
        border: 1px solid rgba(102, 126, 234, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(102, 126, 234, 0.2);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-label {
        color: #888;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 16px 32px;
        font-size: 1.1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* File uploader */
    .stFileUploader {
        border: 2px dashed rgba(102, 126, 234, 0.4);
        border-radius: 16px;
        padding: 20px;
        background: rgba(30, 30, 48, 0.5);
    }
    
    /* Text area */
    .stTextArea textarea {
        background: rgba(30, 30, 48, 0.8);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 12px;
        color: white;
    }
    
    /* Dataframe styling */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }
    
    /* Progress bars */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2);
        border-radius: 10px;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(30, 30, 48, 0.8);
        border-radius: 8px;
        color: white;
        border: 1px solid rgba(102, 126, 234, 0.2);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #667eea, #764ba2);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(30, 30, 48, 0.8);
        border-radius: 8px;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1a1a2e;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #667eea, #764ba2);
        border-radius: 4px;
    }
    
    /* Badge styling */
    .skill-badge {
        display: inline-block;
        padding: 4px 12px;
        margin: 2px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
    }
    
    .skill-matched {
        background: rgba(46, 204, 113, 0.2);
        color: #2ecc71;
        border: 1px solid #2ecc71;
    }
    
    .skill-missing {
        background: rgba(231, 76, 60, 0.2);
        color: #e74c3c;
        border: 1px solid #e74c3c;
    }
    
    /* Rank badges */
    .rank-gold {
        background: linear-gradient(135deg, #f7971e, #ffd200);
        color: #000;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 700;
    }
    
    .rank-silver {
        background: linear-gradient(135deg, #bdc3c7, #ecf0f1);
        color: #000;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 700;
    }
    
    .rank-bronze {
        background: linear-gradient(135deg, #cd7f32, #e5a969);
        color: #000;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 700;
    }
    
    /* Animation */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .animate-fade {
        animation: fadeIn 0.6s ease-out;
    }
</style>
""", unsafe_allow_html=True)

# ---------- SESSION STATE ----------
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'results' not in st.session_state:
    st.session_state.results = None
if 'comparison_mode' not in st.session_state:
    st.session_state.comparison_mode = False

# ---------- EXPANDED SKILLS DATABASE ----------
SKILLS_DATABASE = {
    "Programming Languages": [
        "python", "java", "javascript", "typescript", "c++", "c#", "ruby", "go", "rust", 
        "scala", "kotlin", "swift", "php", "r", "matlab", "perl", "shell", "bash"
    ],
    "Data Science & ML": [
        "machine learning", "deep learning", "neural networks", "tensorflow", "pytorch",
        "keras", "scikit-learn", "pandas", "numpy", "scipy", "data analysis", "data science",
        "nlp", "natural language processing", "computer vision", "reinforcement learning",
        "feature engineering", "model deployment", "mlops", "hugging face", "transformers"
    ],
    "Databases": [
        "sql", "mysql", "postgresql", "mongodb", "redis", "elasticsearch", "cassandra",
        "oracle", "sqlite", "dynamodb", "firebase", "neo4j", "graphql"
    ],
    "Cloud & DevOps": [
        "aws", "azure", "gcp", "google cloud", "docker", "kubernetes", "terraform",
        "jenkins", "ci/cd", "devops", "linux", "ansible", "cloudformation", "serverless"
    ],
    "Data Visualization": [
        "tableau", "power bi", "looker", "matplotlib", "seaborn", "plotly", "d3.js",
        "grafana", "metabase", "qlik"
    ],
    "Web Development": [
        "react", "angular", "vue", "node.js", "django", "flask", "fastapi", "spring",
        "html", "css", "rest api", "microservices", "graphql"
    ],
    "Big Data": [
        "spark", "hadoop", "hive", "kafka", "airflow", "databricks", "snowflake",
        "redshift", "bigquery", "data warehouse", "etl", "data pipeline"
    ],
    "Soft Skills": [
        "leadership", "communication", "teamwork", "problem solving", "analytical",
        "project management", "agile", "scrum", "presentation", "collaboration"
    ],
    "Tools & Platforms": [
        "git", "github", "gitlab", "jira", "confluence", "slack", "excel", "jupyter",
        "vscode", "postman", "figma", "notion"
    ]
}

# Flatten skills for extraction
ALL_SKILLS = []
for category, skills in SKILLS_DATABASE.items():
    ALL_SKILLS.extend(skills)

# ---------- FUNCTIONS ----------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s\+\#]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_text_from_pdf(uploaded_file):
    text = ""
    doc = pymupdf.open(stream=uploaded_file.read(), filetype="pdf")
    for page in doc:
        text += page.get_text()
    uploaded_file.seek(0)  # Reset file pointer
    return text

def extract_skills(text, skills_list):
    text_lower = text.lower()
    found_skills = []
    for skill in skills_list:
        # Use word boundaries for more accurate matching
        pattern = r'\b' + re.escape(skill.lower()) + r'\b'
        if re.search(pattern, text_lower):
            found_skills.append(skill)
    return list(set(found_skills))

def extract_experience(text):
    patterns = [
        r'(\d+)\+?\s*(?:years?|yrs?)\s*(?:of)?\s*(?:experience|exp)?',
        r'experience[:\s]*(\d+)\+?\s*(?:years?|yrs?)',
        r'(\d+)\+?\s*(?:years?|yrs?)\s*in'
    ]
    years_found = []
    for pattern in patterns:
        matches = re.findall(pattern, text.lower())
        years_found.extend([int(m) for m in matches if int(m) <= 50])
    return max(years_found) if years_found else 0

def extract_education(text):
    text_lower = text.lower()
    degrees = {
        "phd": "Ph.D.",
        "doctorate": "Ph.D.",
        "master": "Master's",
        "mba": "MBA",
        "bachelor": "Bachelor's",
        "b.tech": "B.Tech",
        "b.e.": "B.E.",
        "b.sc": "B.Sc",
        "m.tech": "M.Tech",
        "m.sc": "M.Sc"
    }
    found_degrees = []
    for key, value in degrees.items():
        if key in text_lower:
            found_degrees.append(value)
    return found_degrees[0] if found_degrees else "Not specified"

def extract_email(text):
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    matches = re.findall(email_pattern, text)
    return matches[0] if matches else "Not found"

def extract_phone(text):
    phone_patterns = [
        r'\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}',
        r'\+?[0-9]{10,12}'
    ]
    for pattern in phone_patterns:
        matches = re.findall(pattern, text)
        if matches:
            return matches[0]
    return "Not found"

def extract_certifications(text):
    cert_keywords = [
        "aws certified", "azure certified", "google certified", "pmp", "cissp",
        "cka", "ckad", "comptia", "cisco", "oracle certified", "scrum master",
        "six sigma", "itil", "data science certification", "machine learning certification"
    ]
    text_lower = text.lower()
    found_certs = []
    for cert in cert_keywords:
        if cert in text_lower:
            found_certs.append(cert.title())
    return found_certs

def calculate_ats_score(resume_text, jd_text):
    """Calculate ATS compatibility score based on various factors"""
    score = 0
    resume_lower = resume_text.lower()
    jd_lower = jd_text.lower()
    
    # Keyword density (40%)
    jd_words = set(jd_lower.split())
    resume_words = set(resume_lower.split())
    common_words = jd_words.intersection(resume_words)
    keyword_score = len(common_words) / len(jd_words) * 40 if jd_words else 0
    score += min(keyword_score, 40)
    
    # Contact info presence (10%)
    if extract_email(resume_text) != "Not found":
        score += 5
    if extract_phone(resume_text) != "Not found":
        score += 5
    
    # Section headers (20%)
    sections = ["experience", "education", "skills", "projects", "summary", "objective"]
    for section in sections:
        if section in resume_lower:
            score += 3.33
    score = min(score, 70)  # Cap section score contribution
    
    # Skills match (30%)
    jd_skills = extract_skills(jd_lower, ALL_SKILLS)
    resume_skills = extract_skills(resume_lower, ALL_SKILLS)
    if jd_skills:
        skills_match = len(set(resume_skills).intersection(set(jd_skills))) / len(jd_skills)
        score += skills_match * 30
    
    return min(round(score, 1), 100)

def get_ai_recommendation(score, exp_match, skills_match_pct):
    """Generate detailed AI recommendation"""
    if score >= 80 and exp_match and skills_match_pct >= 70:
        return {
            "status": "🌟 Highly Recommended",
            "color": "#2ecc71",
            "action": "Fast-track for interview",
            "confidence": "High"
        }
    elif score >= 60 and (exp_match or skills_match_pct >= 50):
        return {
            "status": "✅ Recommended",
            "color": "#3498db",
            "action": "Schedule initial screening",
            "confidence": "Medium-High"
        }
    elif score >= 40:
        return {
            "status": "🔄 Consider with Reservations",
            "color": "#f39c12",
            "action": "Review manually for potential",
            "confidence": "Medium"
        }
    else:
        return {
            "status": "⏸️ Not Recommended",
            "color": "#e74c3c",
            "action": "Keep in talent pool",
            "confidence": "Low"
        }

def create_radar_chart(skills_by_category, jd_skills_by_category):
    """Create a radar chart for skills comparison"""
    categories = list(SKILLS_DATABASE.keys())
    
    resume_scores = []
    jd_scores = []
    
    for cat in categories:
        cat_skills = SKILLS_DATABASE[cat]
        resume_cat_skills = skills_by_category.get(cat, [])
        jd_cat_skills = jd_skills_by_category.get(cat, [])
        
        if jd_cat_skills:
            match = len(set(resume_cat_skills).intersection(set(jd_cat_skills)))
            resume_scores.append(match / len(jd_cat_skills) * 100)
        else:
            resume_scores.append(0)
        
        jd_scores.append(100 if jd_cat_skills else 0)
    
    return categories, resume_scores, jd_scores

def categorize_skills(skills):
    """Categorize skills into their respective categories"""
    categorized = {}
    for skill in skills:
        for category, cat_skills in SKILLS_DATABASE.items():
            if skill.lower() in [s.lower() for s in cat_skills]:
                if category not in categorized:
                    categorized[category] = []
                categorized[category].append(skill)
                break
    return categorized

# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown("## ⚙️ Analysis Settings")
    st.markdown("---")
    
    st.markdown("### 🎯 Scoring Weights")
    weight_skills = st.slider("Skills Match Weight", 0.0, 1.0, 0.4, 0.05)
    weight_exp = st.slider("Experience Weight", 0.0, 1.0, 0.3, 0.05)
    weight_similarity = st.slider("Content Similarity Weight", 0.0, 1.0, 0.3, 0.05)
    
    # Normalize weights
    total_weight = weight_skills + weight_exp + weight_similarity
    if total_weight > 0:
        weight_skills /= total_weight
        weight_exp /= total_weight
        weight_similarity /= total_weight
    
    st.markdown("---")
    st.markdown("### 🔍 Filter Candidates")
    min_score = st.slider("Minimum Score (%)", 0, 100, 0, 5)
    min_exp = st.slider("Minimum Experience (Years)", 0, 15, 0)
    
    st.markdown("---")
    st.markdown("### 📊 Display Options")
    show_contact = st.checkbox("Show Contact Info", value=True)
    show_ats_score = st.checkbox("Show ATS Score", value=True)
    show_certifications = st.checkbox("Show Certifications", value=True)
    
    st.markdown("---")
    st.markdown("### 🎨 Theme")
    chart_theme = st.selectbox("Chart Color Theme", ["Purple Gradient", "Blue Ocean", "Green Forest", "Sunset"])
    
    theme_colors = {
        "Purple Gradient": ("#667eea", "#764ba2"),
        "Blue Ocean": ("#2193b0", "#6dd5ed"),
        "Green Forest": ("#11998e", "#38ef7d"),
        "Sunset": ("#f12711", "#f5af19")
    }

# ---------- MAIN HEADER ----------
st.markdown('<h1 class="main-header">🎯 ResumeIQ</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Resume Screening & Candidate Ranking System</p>', unsafe_allow_html=True)
st.markdown("")

# ---------- FEATURE HIGHLIGHTS ----------
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("""
    <div style="text-align: center; padding: 10px;">
        <span style="font-size: 2rem;">🤖</span>
        <p style="color: #667eea; font-weight: 600; margin: 5px 0;">AI Analysis</p>
        <p style="color: #888; font-size: 0.8rem;">Smart candidate ranking</p>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div style="text-align: center; padding: 10px;">
        <span style="font-size: 2rem;">📊</span>
        <p style="color: #667eea; font-weight: 600; margin: 5px 0;">ATS Scoring</p>
        <p style="color: #888; font-size: 0.8rem;">Compatibility check</p>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown("""
    <div style="text-align: center; padding: 10px;">
        <span style="font-size: 2rem;">🎯</span>
        <p style="color: #667eea; font-weight: 600; margin: 5px 0;">Skill Matching</p>
        <p style="color: #888; font-size: 0.8rem;">100+ skills tracked</p>
    </div>
    """, unsafe_allow_html=True)
with col4:
    st.markdown("""
    <div style="text-align: center; padding: 10px;">
        <span style="font-size: 2rem;">📈</span>
        <p style="color: #667eea; font-weight: 600; margin: 5px 0;">Visual Reports</p>
        <p style="color: #888; font-size: 0.8rem;">Interactive charts</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ---------- INPUT SECTION ----------
st.markdown("### 📝 Job Requirements")

col1, col2 = st.columns([3, 1])

with col1:
    job_description = st.text_area(
        "Enter Job Description",
        height=180,
        placeholder="Paste the complete job description here including required skills, experience, and qualifications...",
        help="Include all requirements for better matching accuracy"
    )

with col2:
    st.markdown("""
    <div style="background: rgba(102, 126, 234, 0.1); padding: 15px; border-radius: 12px; border: 1px solid rgba(102, 126, 234, 0.2);">
        <h4 style="color: #667eea; margin-top: 0;">💡 Pro Tips</h4>
        <ul style="color: #ccc; font-size: 0.85rem; padding-left: 20px;">
            <li>Include specific technical skills</li>
            <li>Mention required experience</li>
            <li>Add preferred certifications</li>
            <li>List soft skills needed</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("")

# Quick JD Templates
with st.expander("📋 Quick Templates (Click to use)"):
    template_cols = st.columns(3)
    
    templates = {
        "Data Scientist": """We are looking for a Data Scientist with 3+ years of experience.

Required Skills:
- Python, SQL, Machine Learning
- TensorFlow or PyTorch
- Data Analysis and Visualization
- Statistics and Mathematics

Nice to have:
- AWS or GCP experience
- Deep Learning expertise
- NLP experience""",
        
        "Software Engineer": """Software Engineer position requiring 2+ years of experience.

Requirements:
- Strong programming skills in Python or Java
- Experience with REST APIs and microservices
- Database knowledge (SQL and NoSQL)
- Git version control
- Agile/Scrum methodology

Preferred:
- Cloud platforms (AWS/Azure/GCP)
- Docker and Kubernetes
- CI/CD pipelines""",
        
        "Data Analyst": """Data Analyst role with 2+ years of experience.

Must have:
- SQL proficiency
- Excel advanced skills
- Data visualization (Tableau/Power BI)
- Python or R for analysis
- Strong analytical skills

Good to have:
- Business intelligence tools
- Statistics background
- Communication skills"""
    }
    
    with template_cols[0]:
        if st.button("🔬 Data Scientist", use_container_width=True):
            st.session_state.template = templates["Data Scientist"]
    with template_cols[1]:
        if st.button("💻 Software Engineer", use_container_width=True):
            st.session_state.template = templates["Software Engineer"]
    with template_cols[2]:
        if st.button("📊 Data Analyst", use_container_width=True):
            st.session_state.template = templates["Data Analyst"]
    
    if 'template' in st.session_state:
        st.info("Template loaded! Copy the text below:")
        st.code(st.session_state.template)

st.markdown("")

# ---------- FILE UPLOAD ----------
st.markdown("### 📂 Upload Resumes")

uploaded_files = st.file_uploader(
    "Drag and drop resume PDFs here",
    type=["pdf"],
    accept_multiple_files=True,
    help="Upload multiple PDF resumes for batch analysis"
)

if uploaded_files:
    st.success(f"✅ {len(uploaded_files)} resume(s) uploaded successfully")
    
    # Show file previews
    with st.expander("👁️ View Uploaded Files"):
        for i, file in enumerate(uploaded_files):
            st.markdown(f"**{i+1}.** {file.name} ({round(file.size/1024, 1)} KB)")

st.markdown("")

# ---------- ANALYZE BUTTON ----------
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    analyze_clicked = st.button("🚀 Analyze Resumes", use_container_width=True, type="primary")

# ---------- ANALYSIS ----------
if analyze_clicked:
    if not job_description or not uploaded_files:
        st.warning("⚠️ Please provide both a job description and at least one resume")
    else:
        with st.spinner("🔍 Analyzing resumes with AI..."):
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            resumes_data = []
            jd_clean = clean_text(job_description)
            jd_skills = extract_skills(jd_clean, ALL_SKILLS)
            jd_experience = extract_experience(job_description)
            jd_skills_categorized = categorize_skills(jd_skills)
            
            for i, file in enumerate(uploaded_files):
                status_text.text(f"Processing: {file.name}")
                progress_bar.progress((i + 1) / len(uploaded_files))
                
                text = extract_text_from_pdf(file)
                text_clean = clean_text(text)
                
                # Extract all information
                resume_skills = extract_skills(text_clean, ALL_SKILLS)
                experience = extract_experience(text)
                education = extract_education(text)
                email = extract_email(text)
                phone = extract_phone(text)
                certifications = extract_certifications(text)
                ats_score = calculate_ats_score(text, job_description)
                
                # Calculate skill match
                matched_skills = list(set(resume_skills) & set(jd_skills))
                missing_skills = list(set(jd_skills) - set(resume_skills))
                additional_skills = list(set(resume_skills) - set(jd_skills))
                skills_match_pct = len(matched_skills) / len(jd_skills) * 100 if jd_skills else 0
                
                # Experience match
                exp_match = experience >= jd_experience if jd_experience > 0 else True
                
                resumes_data.append({
                    "file_name": file.name,
                    "text": text,
                    "text_clean": text_clean,
                    "skills": resume_skills,
                    "matched_skills": matched_skills,
                    "missing_skills": missing_skills,
                    "additional_skills": additional_skills,
                    "skills_match_pct": skills_match_pct,
                    "experience": experience,
                    "exp_match": exp_match,
                    "education": education,
                    "email": email,
                    "phone": phone,
                    "certifications": certifications,
                    "ats_score": ats_score
                })
            
            # TF-IDF Similarity
            status_text.text("Calculating content similarity...")
            all_texts = [jd_clean] + [r["text_clean"] for r in resumes_data]
            vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
            vectors = vectorizer.fit_transform(all_texts)
            
            for i, resume in enumerate(resumes_data):
                similarity = cosine_similarity(vectors[0], vectors[i+1])[0][0]
                resume["similarity"] = similarity * 100
                
                # Calculate weighted final score
                exp_score = min(resume["experience"] / max(jd_experience, 1) * 100, 100) if jd_experience > 0 else 50
                final_score = (
                    weight_skills * resume["skills_match_pct"] +
                    weight_exp * exp_score +
                    weight_similarity * resume["similarity"]
                )
                resume["final_score"] = round(final_score, 1)
                
                # AI Recommendation
                resume["recommendation"] = get_ai_recommendation(
                    resume["final_score"],
                    resume["exp_match"],
                    resume["skills_match_pct"]
                )
            
            # Sort by final score
            resumes_data.sort(key=lambda x: x["final_score"], reverse=True)
            
            # Assign ranks
            for i, resume in enumerate(resumes_data):
                resume["rank"] = i + 1
            
            progress_bar.progress(100)
            status_text.text("Analysis complete!")
            
            # Store results
            st.session_state.results = resumes_data
            st.session_state.jd_skills = jd_skills
            st.session_state.jd_experience = jd_experience
            st.session_state.analysis_complete = True
            st.session_state.jd_skills_categorized = jd_skills_categorized

# ---------- RESULTS DISPLAY ----------
if st.session_state.analysis_complete and st.session_state.results:
    results = st.session_state.results
    jd_skills = st.session_state.jd_skills
    jd_experience = st.session_state.jd_experience
    
    # Filter results
    filtered_results = [
        r for r in results 
        if r["final_score"] >= min_score and r["experience"] >= min_exp
    ]
    
    if not filtered_results:
        st.warning("No candidates match the current filter criteria. Try adjusting the filters.")
        st.stop()
    
    st.markdown("---")
    st.markdown("## 📊 Analysis Results")
    
    # ---------- KPI METRICS ----------
    kpi_cols = st.columns(5)
    
    with kpi_cols[0]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Analyzed</div>
            <div class="metric-value">{len(results)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with kpi_cols[1]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Qualified</div>
            <div class="metric-value">{len(filtered_results)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with kpi_cols[2]:
        avg_score = round(sum(r["final_score"] for r in filtered_results) / len(filtered_results), 1)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Avg Score</div>
            <div class="metric-value">{avg_score}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with kpi_cols[3]:
        top_score = filtered_results[0]["final_score"]
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Top Score</div>
            <div class="metric-value">{top_score}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with kpi_cols[4]:
        highly_rec = len([r for r in filtered_results if r["recommendation"]["status"].startswith("🌟")])
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Highly Recommended</div>
            <div class="metric-value">{highly_rec}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("")
    
    # ---------- TABS ----------
    tab1, tab2, tab3, tab4 = st.tabs(["🏆 Rankings", "📋 Detailed View", "📈 Analytics", "🔄 Compare"])
    
    # ---------- TAB 1: RANKINGS ----------
    with tab1:
        st.markdown("### Candidate Rankings")
        
        for i, candidate in enumerate(filtered_results[:10]):  # Top 10
            rank_class = "rank-gold" if i == 0 else ("rank-silver" if i == 1 else ("rank-bronze" if i == 2 else ""))
            rank_emoji = "🥇" if i == 0 else ("🥈" if i == 1 else ("🥉" if i == 2 else f"#{i+1}"))
            
            with st.container():
                col1, col2, col3, col4 = st.columns([0.5, 2, 1, 1])
                
                with col1:
                    st.markdown(f"<h2 style='text-align: center; margin: 0;'>{rank_emoji}</h2>", unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"**{candidate['file_name']}**")
                    st.markdown(f"<span style='color: {candidate['recommendation']['color']};'>{candidate['recommendation']['status']}</span>", unsafe_allow_html=True)
                
                with col3:
                    st.metric("Score", f"{candidate['final_score']}%")
                
                with col4:
                    st.metric("Experience", f"{candidate['experience']} yrs")
                
                # Progress bar
                st.progress(int(candidate['final_score']))
                
                # Expandable details
                with st.expander("View Details"):
                    detail_cols = st.columns(3)
                    
                    with detail_cols[0]:
                        st.markdown("**✅ Matched Skills:**")
                        if candidate['matched_skills']:
                            for skill in candidate['matched_skills'][:8]:
                                st.markdown(f"<span class='skill-badge skill-matched'>{skill}</span>", unsafe_allow_html=True)
                        else:
                            st.write("None")
                    
                    with detail_cols[1]:
                        st.markdown("**❌ Missing Skills:**")
                        if candidate['missing_skills']:
                            for skill in candidate['missing_skills'][:8]:
                                st.markdown(f"<span class='skill-badge skill-missing'>{skill}</span>", unsafe_allow_html=True)
                        else:
                            st.write("None")
                    
                    with detail_cols[2]:
                        st.markdown("**ℹ️ Additional Info:**")
                        st.write(f"📧 {candidate['email']}")
                        st.write(f"📱 {candidate['phone']}")
                        st.write(f"🎓 {candidate['education']}")
                        if show_ats_score:
                            st.write(f"🤖 ATS Score: {candidate['ats_score']}%")
                
                st.markdown("---")
    
    # ---------- TAB 2: DETAILED VIEW ----------
    with tab2:
        st.markdown("### Detailed Candidate Analysis")
        
        # Create detailed dataframe
        df_data = []
        for r in filtered_results:
            row = {
                "Rank": r["rank"],
                "Resume": r["file_name"],
                "Score (%)": r["final_score"],
                "Skills Match (%)": round(r["skills_match_pct"], 1),
                "Experience": r["experience"],
                "Education": r["education"],
                "Recommendation": r["recommendation"]["status"]
            }
            if show_contact:
                row["Email"] = r["email"]
            if show_ats_score:
                row["ATS Score"] = r["ats_score"]
            if show_certifications:
                row["Certifications"] = ", ".join(r["certifications"][:3]) if r["certifications"] else "None"
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        
        # Style the dataframe
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Score (%)": st.column_config.ProgressColumn(
                    "Score (%)",
                    min_value=0,
                    max_value=100,
                    format="%d%%"
                ),
                "Skills Match (%)": st.column_config.ProgressColumn(
                    "Skills Match (%)",
                    min_value=0,
                    max_value=100,
                    format="%d%%"
                )
            }
        )
    
    # ---------- TAB 3: ANALYTICS ----------
    with tab3:
        st.markdown("### Visual Analytics")
        
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            # Score Distribution
            st.markdown("#### Score Distribution")
            fig, ax = plt.subplots(figsize=(8, 5))
            fig.patch.set_facecolor('#0E1117')
            ax.set_facecolor('#0E1117')
            
            scores = [r["final_score"] for r in filtered_results]
            names = [r["file_name"][:20] for r in filtered_results]
            colors = [theme_colors[chart_theme][0] if s >= 60 else theme_colors[chart_theme][1] for s in scores]
            
            bars = ax.barh(names[:10], scores[:10], color=colors)
            ax.set_xlabel('Score (%)', color='white')
            ax.tick_params(colors='white')
            ax.spines['bottom'].set_color('white')
            ax.spines['left'].set_color('white')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            for bar, score in zip(bars, scores[:10]):
                ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                       f'{score}%', va='center', color='white', fontsize=9)
            
            plt.tight_layout()
            st.pyplot(fig)
        
        with chart_col2:
            # Skills Gap Analysis
            st.markdown("#### Skills Gap Analysis (Top Candidate)")
            
            top_candidate = filtered_results[0]
            matched = len(top_candidate['matched_skills'])
            missing = len(top_candidate['missing_skills'])
            
            fig, ax = plt.subplots(figsize=(6, 6))
            fig.patch.set_facecolor('#0E1117')
            ax.set_facecolor('#0E1117')
            
            sizes = [matched, missing]
            labels = [f'Matched\n({matched})', f'Missing\n({missing})']
            colors_pie = ['#2ecc71', '#e74c3c']
            explode = (0.05, 0)
            
            wedges, texts, autotexts = ax.pie(
                sizes, labels=labels, autopct='%1.1f%%',
                colors=colors_pie, explode=explode,
                wedgeprops=dict(width=0.5, edgecolor='#0E1117'),
                textprops={'color': 'white'}
            )
            
            ax.set_title(top_candidate['file_name'], color='white', fontsize=12)
            plt.tight_layout()
            st.pyplot(fig)
        
        # Experience vs Score scatter
        st.markdown("#### Experience vs Score Analysis")
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor('#0E1117')
        ax.set_facecolor('#0E1117')
        
        exp = [r["experience"] for r in filtered_results]
        scores = [r["final_score"] for r in filtered_results]
        names = [r["file_name"] for r in filtered_results]
        
        scatter = ax.scatter(exp, scores, c=scores, cmap='RdYlGn', s=100, alpha=0.8, edgecolors='white')
        
        for i, name in enumerate(names):
            ax.annotate(name[:15], (exp[i], scores[i]), fontsize=8, color='white', 
                       xytext=(5, 5), textcoords='offset points')
        
        ax.set_xlabel('Experience (Years)', color='white')
        ax.set_ylabel('Score (%)', color='white')
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        if jd_experience > 0:
            ax.axvline(x=jd_experience, color='#e74c3c', linestyle='--', label=f'Required: {jd_experience} yrs')
            ax.legend(facecolor='#1a1a2e', labelcolor='white')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # ---------- TAB 4: COMPARE ----------
    with tab4:
        st.markdown("### Compare Candidates Side by Side")
        
        candidate_names = [r["file_name"] for r in filtered_results]
        
        col1, col2 = st.columns(2)
        with col1:
            candidate1 = st.selectbox("Select First Candidate", candidate_names, key="cand1")
        with col2:
            candidate2 = st.selectbox("Select Second Candidate", candidate_names, index=min(1, len(candidate_names)-1), key="cand2")
        
        if candidate1 and candidate2:
            c1 = next(r for r in filtered_results if r["file_name"] == candidate1)
            c2 = next(r for r in filtered_results if r["file_name"] == candidate2)
            
            compare_cols = st.columns(2)
            
            for col, candidate, label in zip(compare_cols, [c1, c2], [candidate1, candidate2]):
                with col:
                    st.markdown(f"""
                    <div style="background: rgba(30, 30, 48, 0.8); padding: 20px; border-radius: 12px; border: 1px solid rgba(102, 126, 234, 0.3);">
                        <h3 style="color: #667eea; margin-top: 0;">{label}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.metric("Overall Score", f"{candidate['final_score']}%")
                    st.metric("Skills Match", f"{round(candidate['skills_match_pct'], 1)}%")
                    st.metric("Experience", f"{candidate['experience']} years")
                    st.metric("ATS Score", f"{candidate['ats_score']}%")
                    
                    st.markdown("**Matched Skills:**")
                    st.write(", ".join(candidate['matched_skills'][:10]) or "None")
                    
                    st.markdown("**Missing Skills:**")
                    st.write(", ".join(candidate['missing_skills'][:10]) or "None")
                    
                    st.markdown(f"**Recommendation:** {candidate['recommendation']['status']}")
    
    # ---------- DOWNLOAD SECTION ----------
    st.markdown("---")
    st.markdown("### 📥 Export Results")
    
    export_cols = st.columns(3)
    
    with export_cols[0]:
        # CSV Export
        csv_data = []
        for r in filtered_results:
            csv_data.append({
                "Rank": r["rank"],
                "Resume": r["file_name"],
                "Final Score (%)": r["final_score"],
                "Skills Match (%)": round(r["skills_match_pct"], 1),
                "Experience (Years)": r["experience"],
                "Education": r["education"],
                "Email": r["email"],
                "Phone": r["phone"],
                "ATS Score": r["ats_score"],
                "Matched Skills": ", ".join(r["matched_skills"]),
                "Missing Skills": ", ".join(r["missing_skills"]),
                "Certifications": ", ".join(r["certifications"]),
                "AI Recommendation": r["recommendation"]["status"],
                "Recommended Action": r["recommendation"]["action"]
            })
        
        csv_df = pd.DataFrame(csv_data)
        csv = csv_df.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            "📊 Download CSV",
            csv,
            f"resume_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            "text/csv",
            use_container_width=True
        )
    
    with export_cols[1]:
        # JSON Export
        import json
        json_data = json.dumps(filtered_results, indent=2, default=str)
        
        st.download_button(
            "📋 Download JSON",
            json_data,
            f"resume_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
            "application/json",
            use_container_width=True
        )
    
    with export_cols[2]:
        # Summary Report
        summary = f"""
RESUME ANALYSIS SUMMARY REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

OVERVIEW
========
Total Resumes Analyzed: {len(results)}
Candidates Meeting Criteria: {len(filtered_results)}
Average Score: {avg_score}%
Top Score: {top_score}%

JOB REQUIREMENTS
================
Required Experience: {jd_experience} years
Required Skills: {', '.join(jd_skills[:15])}

TOP 5 CANDIDATES
================
"""
        for i, c in enumerate(filtered_results[:5]):
            summary += f"""
{i+1}. {c['file_name']}
   Score: {c['final_score']}%
   Experience: {c['experience']} years
   Skills Match: {round(c['skills_match_pct'], 1)}%
   Recommendation: {c['recommendation']['status']}
"""
        
        st.download_button(
            "📄 Download Summary",
            summary,
            f"resume_summary_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            "text/plain",
            use_container_width=True
        )

# ---------- FOOTER ----------
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p>🎯 <strong>ResumeIQ</strong> - AI-Powered Resume Screening System</p>
    <p style="font-size: 0.8rem;">Built with Streamlit • Powered by Machine Learning</p>
</div>
""", unsafe_allow_html=True)
