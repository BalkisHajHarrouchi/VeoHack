import streamlit as st
import pandas as pd
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz
import numpy as np
import langchain
import re
from datetime import datetime
import os
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
langchain.debug = True

# ✅ Configure Streamlit page layout
st.set_page_config(layout="wide")

# ✅ Load the embedding model
EMBEDDING_MODEL = "Lajavaness/bilingual-embedding-large"
model_kwargs = {"trust_remote_code": True}
embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs=model_kwargs)



# ✅ Create directories if not exist
os.makedirs("emb", exist_ok=True)

# ✅ Load and split job offers text file
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0, separator="\n\n")
loader_jobs = TextLoader("jobs.txt")  # Load jobs from text file
docs_jobs = loader_jobs.load_and_split(text_splitter=text_splitter)

# ✅ Initialize ChromaDB
db_jobs = Chroma(persist_directory="emb/jobsDB", embedding_function=embedding_function)

# ✅ Add text data to ChromaDB
db_jobs.add_texts(texts=[doc.page_content for doc in docs_jobs])



# ✅ Setup ChromaDB
def setup_chromadb():
    db_jobs = Chroma(persist_directory="emb/jobsDB", embedding_function=embedding_function)
    db_candidates = Chroma(persist_directory="emb/candidatesDB", embedding_function=embedding_function)
    return db_jobs, db_candidates, embedding_function

# ✅ Weight System
SKILL_WEIGHTS = {"Critical": 2, "Advanced": 1.5, "Intermediate": 1}
EXPERIENCE_WEIGHT = {"Highly Relevant": 1.5, "Relevant": 1.2, "Somewhat Relevant": 1.0, "Not Relevant": 0.5}
SOFT_SKILL_BONUS = {"Communication": 0.10, "Problem-Solving": 0.07, "Leadership": 0.05, "Creativity": 0.02}

# ✅ Extract structured job & candidate data with improved skill extraction
def extract_skills_from_text(text):
    data = {}

    if "Job ID" in text:
        data["Job ID"] = re.search(r"Job ID:\s*(.*)", text).group(1).strip() if re.search(r"Job ID:\s*(.*)", text) else None
        data["Title"] = re.search(r"Title:\s*(.*)", text).group(1).strip() if re.search(r"Title:\s*(.*)", text) else None

        skills_text = re.findall(r"- (.+?) \((Critical|Advanced|Intermediate)\)", text)
        data["Required Skills"] = {skill.strip(): level for skill, level in skills_text}

    if "Candidate ID" in text:
        data["Candidate ID"] = re.search(r"Candidate ID:\s*(.*)", text).group(1).strip() if re.search(r"Candidate ID:\s*(.*)", text) else None
        data["Name"] = re.search(r"Name:\s*(.*)", text).group(1).strip() if re.search(r"Name:\s*(.*)", text) else None
        data["Selected Job"] = re.search(r"Selected Job:\s*(.*)", text).group(1).strip() if re.search(r"Selected Job:\s*(.*)", text) else None

        # ✅ Improved Hard Skills Extraction using Cosine Similarity
        hard_skills_match = re.findall(r"Hard Skills:\s*\n((?:\s{4}.+?: .+\n)+)", text)
        if hard_skills_match:
            hard_skills_lines = hard_skills_match[0].strip().split("\n")
            data["Hard Skills"] = {line.split(":")[0].strip(): line.split(":")[1].strip() for line in hard_skills_lines}
        else:
            data["Hard Skills"] = {}

        # ✅ Extract Soft Skills
        soft_skills_match = re.search(r"Soft Skills:\s*(.*)", text)
        data["Soft Skills"] = soft_skills_match.group(1).strip().split(", ") if soft_skills_match else []

        # ✅ Extract Experience
        experience_match = re.findall(r"- Job Title:\s*(.*?)\n.*?Company:\s*(.*?)\n.*?Duration:\s*(.*?)\n", text, re.DOTALL)
        data["Experience"] = [{"Title": title.strip(), "Company": company.strip(), "Duration": duration.strip()} for title, company, duration in experience_match]

        # ✅ Extract Education
        edu_match = re.search(r"Education:\s*(.*?)\n", text)
        data["Education"] = edu_match.group(1).strip() if edu_match else "N/A"

        # ✅ Extract Certifications (Exclude Languages)
        cert_text = text.split("Certifications:")[-1] if "Certifications:" in text else ""
        cert_match = re.findall(r"- ([A-Za-z0-9].+)", cert_text)
        data["Certifications"] = [cert.strip() for cert in cert_match if not re.search(r"(French|English|Arabic|Spanish|German|Italian|Chinese)", cert, re.IGNORECASE)]

        # ✅ Extract Languages (Separated)
        lang_text = text.split("Languages:")[-1] if "Languages:" in text else ""
        lang_match = re.findall(r"- ([A-Za-z0-9].+)", lang_text)
        data["Languages"] = [lang.strip() for lang in lang_match if re.search(r"(French|English|Arabic|Spanish|German|Italian|Chinese)", lang, re.IGNORECASE)]

    return data



# ✅ Compute weighted skill similarity using Cosine Similarity
def compute_weighted_skill_similarity(job_skills, candidate_skills, embedding_function, threshold=0.65):
    if not job_skills or not candidate_skills:
        return {}

    matched_skills = {}
    job_embeddings = embedding_function.embed_documents(list(job_skills.keys()))
    candidate_embeddings = embedding_function.embed_documents(candidate_skills)

    similarity_matrix = cosine_similarity(job_embeddings, candidate_embeddings)

    for i, (job_skill, skill_level) in enumerate(job_skills.items()):
        for j, candidate_skill in enumerate(candidate_skills):
            cosine_sim = similarity_matrix[i, j]

            if cosine_sim >= threshold:
                weight = SKILL_WEIGHTS.get(skill_level, 1)  
                matched_skills[job_skill] = weight  

    return matched_skills

# ✅ Match and rank candidates for a job
def find_best_candidates(job, db_candidates, top_n=10):
    results = db_candidates.similarity_search(job["Title"], k=top_n)
    ranked_candidates = []

    for result in results:
        candidate = extract_skills_from_text(result.page_content)
        matched_skills = compute_weighted_skill_similarity(job["Required Skills"], list(candidate.get("Hard Skills", [])), embedding_function, threshold=0.65)
        total_weighted_match = sum(matched_skills.values())
        soft_skill_bonus = sum(SOFT_SKILL_BONUS.get(skill, 0) for skill in candidate.get("Soft Skills", []))
        applied_boost = 0.1 if candidate.get("Selected Job") == job["Title"] else 0.0
        final_score = round((total_weighted_match * 0.7 + soft_skill_bonus + applied_boost), 2)

        ranked_candidates.append({
            "Name": candidate.get("Name", "N/A"),
            "Extracted Experience": "; ".join([f"{exp['Title']} at {exp['Company']} ({exp['Duration']})" for exp in candidate.get("Experience", [])]) if candidate.get("Experience", []) else "None",
            "Education": candidate.get("Education", "N/A"),
            "Certifications": ", ".join(candidate.get("Certifications", [])) if candidate.get("Certifications", []) else "None",
            "Languages": ", ".join(candidate.get("Languages", [])) if candidate.get("Languages", []) else "None",
            "Matched Skills": ", ".join(matched_skills.keys()) if matched_skills else "None",
            "Skill Score": total_weighted_match,
            "Soft Skills": ", ".join(candidate.get("Soft Skills", [])) if candidate.get("Soft Skills", []) else "None",
            "Soft Skill Score": soft_skill_bonus,
            "Applied Boost": applied_boost,
            "Final Score": final_score,
        })

    return sorted(ranked_candidates, key=lambda x: x["Final Score"], reverse=True)[:top_n]

if __name__ == "__main__":
    st.title("Job Matching System")
    
    # Initialize the databases
    db_jobs, db_candidates, embedding_function = setup_chromadb()
    
    # Perform similarity search to get job offers
    job_texts = db_jobs.similarity_search("Job", k=10)
    job_offers = [extract_skills_from_text(job.page_content) for job in job_texts]
    
    # Debugging: check if job_offers are populated
    st.write(f"Number of job offers loaded: {len(job_offers)}")
    st.write("Sample job offer:", job_offers[0] if job_offers else "No jobs found")

    # Job selection dropdown
    selected_job = st.selectbox("📌 Select a job:", ["Select"] + [job["Title"] for job in job_offers])

    if selected_job != "Select":
        job = next(job for job in job_offers if job["Title"] == selected_job)
        
        # Find best candidates for the selected job
        ranked_candidates = find_best_candidates(job, db_candidates)
        
        if ranked_candidates:
            df = pd.DataFrame(ranked_candidates)
            st.dataframe(df)
        else:
            st.write("No candidates found for this job.")

