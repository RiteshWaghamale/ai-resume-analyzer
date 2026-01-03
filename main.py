import streamlit as st
import PyPDF2
import io
import os
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page Config
st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📃",
    layout="centered"
)

st.title("📃 AI Resume Analyzer (Groq Llama 3.3)")
st.markdown("Upload your resume and get AI-powered feedback tailored to your needs!")

# 1. Initialize Groq Client
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.error("GROQ_API_KEY not found. Please check your .env file.")
    st.stop()

client = Groq(api_key=api_key)

# UI Elements
uploaded_file = st.file_uploader(
    "Upload your resume (PDF or TXT)",
    type=["pdf", "txt"]
)

job_role = st.text_input(
    "Enter the job role you're targeting (optional)"
)

analyze = st.button("Analyze Resume")

# ---------- Helper Functions ----------

def extract_text_from_pdf(file_bytes):
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def extract_text_from_file(uploaded_file):
    file_bytes = uploaded_file.read()
    if uploaded_file.type == "application/pdf":
        return extract_text_from_pdf(file_bytes)
    else:
        return file_bytes.decode("utf-8", errors="ignore")

# ---------- Main Logic ----------

if analyze:
    if not uploaded_file:
        st.warning("Please upload a resume file.")
        st.stop()

    try:
        # Extract Text
        resume_text = extract_text_from_file(uploaded_file)
        
        if not resume_text.strip():
            st.error("Could not extract text from the resume.")
            st.stop()

        # Limit text length to avoid token limits
        resume_text = resume_text[:20000]

        # Prepare Prompt
        prompt_content = f"""
        You are an experienced technical recruiter. 
        Analyze the following resume and provide constructive feedback.

        Target Job Role: {job_role if job_role else "General Application"}

        Focus on:
        1. Content clarity and impact
        2. Skills presentation (Identify missing critical skills for the target role)
        3. Experience descriptions (Are they result-oriented?)
        4. Specific improvements

        Resume Content:
        {resume_text}

        Provide the feedback in a clear, structured format with bullet points.
        """

        with st.spinner("Analyzing your resume with Llama 3.3..."):
            
            # 2. Call Groq API 
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful and professional career coach."
                    },
                    {
                        "role": "user",
                        "content": prompt_content,
                    }
                ],
               
                model="llama-3.3-70b-versatile", 
                temperature=0.5,
            )

            # 3. Extract Response
            result = chat_completion.choices[0].message.content

        st.markdown("## ✅ Analysis Results")
        st.markdown(result)

    except Exception as e:
        st.error(f"An error occurred: {e}")