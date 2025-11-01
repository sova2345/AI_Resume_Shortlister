import streamlit as st
import re
import PyPDF2
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------------
# 🎯 APP HEADER
# --------------------------------------------------
st.set_page_config(page_title="AI Resume Shortlister", page_icon="🤖", layout="wide")
st.title("🤖 AI Resume Shortlister")
st.write("""
Upload your **resume** and paste any **job description** below.  
The app will calculate how well your resume matches that job’s requirements and give smart suggestions to improve it.
""")
st.markdown("---")

# --------------------------------------------------
# 📋 JOB DESCRIPTION INPUT
# --------------------------------------------------
st.subheader("📄 Step 1: Paste Job Description")
job_description = st.text_area(
    "Paste the job description here (copy from LinkedIn, Naukri, or company careers page)",
    height=220,
    placeholder="Example: We are hiring a Software Engineer skilled in Python, Java, DSA, and SQL..."
)

st.markdown("---")

# --------------------------------------------------
# 📤 RESUME UPLOAD
# --------------------------------------------------
st.subheader("📎 Step 2: Upload Your Resume (PDF only)")
uploaded_file = st.file_uploader("Upload resume", type=["pdf"])

# --------------------------------------------------
# 🔍 TEXT EXTRACTION FUNCTION
# --------------------------------------------------
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

# --------------------------------------------------
# 🧹 CLEANING FUNCTION
# --------------------------------------------------
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    return text

# --------------------------------------------------
# 🧮 MATCH SCORE FUNCTION
# --------------------------------------------------
def calculate_match_score(resume_text, jd_text):
    text_data = [resume_text, jd_text]
    cv = CountVectorizer(stop_words='english')
    count_matrix = cv.fit_transform(text_data)
    match_score = cosine_similarity(count_matrix)[0][1]
    return round(match_score * 100, 2)

# --------------------------------------------------
# 🚀 MAIN LOGIC
# --------------------------------------------------
if st.button("🔍 Analyze Match"):
    if uploaded_file is not None and job_description.strip() != "":
        st.success("✅ Resume and Job Description uploaded successfully!")

        # Extract + Clean
        resume_text = extract_text_from_pdf(uploaded_file)
        resume_text = clean_text(resume_text)
        jd_text = clean_text(job_description)

        # Calculate score
        score = calculate_match_score(resume_text, jd_text)

        st.markdown("---")
        st.subheader("📊 Match Analysis Result")
        st.metric(label="Resume–Job Match Score", value=f"{score}%")

        # Smart feedback
        st.markdown("### 💬 AI Feedback")
        if score >= 80:
            st.success("✅ Excellent Match! Your resume aligns very well with this job description. It’s likely ATS-friendly.")
        elif score >= 60:
            st.warning("🟡 Good Match — but try adding keywords from the JD (skills, tools, or responsibilities) to boost your alignment.")
        else:
            st.error("❌ Low Match — your resume may lack key terms from the job description. Add technical keywords, certifications, or relevant experience.")

        st.markdown("---")
        st.caption("💡 Tip: Use power words from the JD (e.g., 'Agile', 'Cloud', 'APIs', 'Data Analysis', 'Machine Learning', etc.) to improve match percentage.")
    else:
        st.warning("⚠️ Please upload a resume and paste a job description before clicking Analyze.")
else:
    st.info("👆 Paste the job description and upload your resume, then click 'Analyze Match'.")
