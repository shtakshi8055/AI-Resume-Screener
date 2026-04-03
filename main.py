import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
import re
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide"
)

# ---------- DARK THEME CSS ----------
st.markdown("""
    <style>
    body {
        background-color: #0E1117;
        color: white;
    }
    .stMetric {
        background-color: #1c1f26;
        padding: 15px;
        border-radius: 10px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        height: 3em;
        width: 100%;
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)

# ---------- FUNCTIONS ----------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

def extract_text_from_pdf(uploaded_file):
    text = ""
    reader = PdfReader(uploaded_file)
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def extract_skills(text, skills_list):
    return [skill for skill in skills_list if skill in text]

def extract_experience(text):
    matches = re.findall(r'(\d+)\+?\s*(years|yrs)', text)
    if matches:
        years = [int(match[0]) for match in matches]
        return max(years)
    return 0

def extract_required_experience(text):
    match = re.search(r'(\d+)\+?\s*(years|yrs)', text)
    if match:
        return int(match.group(1))
    return 0

# ---------- SKILLS ----------
skills_list = [
    "python", "sql", "machine learning", "data analysis",
    "power bi", "excel", "tableau", "statistics"
]

# ---------- SIDEBAR ----------
st.sidebar.title("⚙️ Filters")

min_score = st.sidebar.slider("Minimum Score (%)", 0, 100, 0)
min_exp = st.sidebar.slider("Minimum Experience (Years)", 0, 10, 0)

# ---------- HEADER ----------
st.title("📄 AI Resume Shortlisting System")
st.caption("Smart Hiring Assistant using AI & NLP 🚀")

# ---------- INPUT ----------
col1, col2 = st.columns([2, 1])

with col1:
    job_description = st.text_area("📌 Enter Job Description", height=150)

with col2:
    st.info("💡 Tips:\n- Add skills\n- Add tools\n- Mention experience")

uploaded_files = st.file_uploader(
    "📂 Upload Resume PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

# ---------- BUTTON ----------
if st.button("🚀 Analyze Resumes"):

    if not job_description or not uploaded_files:
        st.warning("⚠️ Please provide both Job Description and Resumes")
    else:
        with st.spinner("🔍 Analyzing resumes..."):

            resumes = []
            file_names = []

            for file in uploaded_files:
                text = extract_text_from_pdf(file)
                resumes.append(text)
                file_names.append(file.name)

            job_clean = clean_text(job_description)
            resumes_clean = [clean_text(r) for r in resumes]

            all_texts = [job_clean] + resumes_clean

            vectorizer = TfidfVectorizer()
            vectors = vectorizer.fit_transform(all_texts)

            jd_vector = vectors[0]

            scores = []
            matched_skills = []
            missing_skills = []
            experience_list = []

            jd_skills = extract_skills(job_clean, skills_list)
            jd_experience = extract_required_experience(job_clean)

            for i in range(1, vectors.shape[0]):
                score = cosine_similarity(jd_vector, vectors[i])[0][0]
                scores.append(score)

                res_text = resumes_clean[i-1]

                res_skills = extract_skills(res_text, skills_list)
                matched = list(set(res_skills) & set(jd_skills))
                missing = list(set(jd_skills) - set(res_skills))

                matched_skills.append(", ".join(matched))
                missing_skills.append(", ".join(missing))

                exp = extract_experience(res_text)
                experience_list.append(exp)

            # ---------- RESULTS ----------
            results = pd.DataFrame({
                "Resume": file_names,
                "Score (%)": [round(s*100, 2) for s in scores],
                "Experience (Years)": experience_list,
                "Matched Skills": matched_skills,
                "Missing Skills": missing_skills
            })

            results = results.sort_values(by="Score (%)", ascending=False)
            results.reset_index(drop=True, inplace=True)

            # ---------- FILTERS ----------
            results = results[
                (results["Score (%)"] >= min_score) &
                (results["Experience (Years)"] >= min_exp)
            ]

        if results.empty:
            st.warning("No candidates match the filter criteria")
            st.stop()

        top = results.iloc[0]

        st.success("✅ Analysis Complete")

        # ---------- KPI ----------
        col1, col2, col3 = st.columns(3)
        col1.metric("🏆 Top Score", f"{top['Score (%)']}%")
        col2.metric("📄 Total Resumes", len(results))
        col3.metric("📊 Avg Score", f"{round(results['Score (%)'].mean(),2)}%")

        # ---------- TABLE ----------
        st.subheader("📋 Candidate Analysis")
        st.dataframe(results, use_container_width=True)

        # ---------- PROGRESS ----------
        st.subheader("📈 Score Visualization")
        for i, row in results.iterrows():
            st.write(f"{row['Resume']} ({row['Score (%)']}%)")
            st.progress(int(row["Score (%)"]))

        # ---------- DONUT ----------
        st.subheader("🎯 Skill Match Overview")

        matched = len(top["Matched Skills"].split(", ")) if top["Matched Skills"] else 0
        missing = len(top["Missing Skills"].split(", ")) if top["Missing Skills"] else 0

        fig, ax = plt.subplots(figsize=(2.5,2.5))
        ax.pie(
            [matched, missing],
            labels=["Matched", "Missing"],
            autopct="%1.1f%%",
            wedgeprops=dict(width=0.4)
        )
        ax.text(0, 0, "Skills", ha='center', va='center', fontsize=9)

        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.pyplot(fig, use_container_width=False)

        # ---------- TOP CANDIDATE ----------
        st.subheader("🏆 Top Candidate")
        st.success(f"""
        **{top['Resume']}**

        Score: {top['Score (%)']}%  
        Experience: {top['Experience (Years)']} years  
        """)

        # ---------- EXPERIENCE ----------
        st.subheader("🧠 Experience Analysis")
        st.write(f"Required Experience: {jd_experience} years")

        for i, row in results.iterrows():
            status = "✅" if row["Experience (Years)"] >= jd_experience else "❌"
            st.write(f"{row['Resume']} → {row['Experience (Years)']} years {status}")

        # ---------- AI FEEDBACK ----------
        def get_feedback(score):
            if score >= 75:
                return "Excellent"
            elif score >= 50:
                return "Good"
            else:
                return "Needs Improvement"

        results["AI Feedback"] = results["Score (%)"].apply(get_feedback)

        st.subheader("🤖 AI Feedback")
        st.dataframe(results[["Resume","Score (%)","AI Feedback"]], use_container_width=True)

        # ---------- DOWNLOAD ----------
        csv = results.to_csv(index=False).encode('utf-8')

        st.download_button(
            "📥 Download Results",
            csv,
            "resume_analysis.csv",
            "text/csv"
        )