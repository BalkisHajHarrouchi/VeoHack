import streamlit as st
import pandas as pd
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz
import numpy as np
import langchain
import re

langchain.debug = True

# ✅ Configure Streamlit page layout
st.set_page_config(layout="wide")

# ✅ Load the embedding model
EMBEDDING_MODEL = "Lajavaness/bilingual-embedding-large"
model_kwargs = {"trust_remote_code": True}
embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs=model_kwargs)

# ✅ Setup ChromaDB
def setup_chromadb():
    db_jobs = Chroma(persist_directory="emb/jobsDB", embedding_function=embedding_function)
    db_candidates = Chroma(persist_directory="emb/candidatesDB", embedding_function=embedding_function)
    return db_jobs, db_candidates, embedding_function

# ✅ Skill Level Weights
SKILL_WEIGHTS = {
    "Critical": 2,
    "Advanced": 1.5,
    "Intermediate": 1,
}

# ✅ Expanded Soft Skill Bonus Mapping (VEO Worldwide Priorities)
SOFT_SKILL_BONUS = {
    "Communication": 0.10,
    "Organizational Skills": 0.08,
    "Interpersonal Skills": 0.08,
    "Problem-Solving": 0.07,
    "Autonomy": 0.07,
    "Leadership": 0.05,
    "Critical Thinking": 0.05,
    "Decision Making": 0.05,
    "Entrepreneurial Mindset": 0.05,
    "Innovation": 0.05,
    "Time Management": 0.04,
    "Conflict Resolution": 0.04,
    "Negotiation": 0.04,
    "Cross-Department Collaboration": 0.04,
    "Adaptability": 0.04,
    "Transparency": 0.04,
    "Emotional Intelligence": 0.03,
    "Presentation Skills": 0.03,
    "Strategic Thinking": 0.03,
    "Work Ethic": 0.03,
    "Learning Agility": 0.03,
    "Multicultural Awareness": 0.03,
    "Collaboration": 0.03,
    "Creativity": 0.02,
    "Flexibility": 0.02,
    "Resilience": 0.02,
    "Attention to Detail": 0.02,
    "Cultural Awareness": 0.02,
    "Persuasion": 0.02,
    "Fast Learning": 0.02,
    "Self-Motivation": 0.01,
    "Networking": 0.01,
    "Positive Attitude": 0.01,
    "Customer Service": 0.01,
    "Public Speaking": 0.01,
    "Active Listening": 0.01,
    "Growth Mindset": 0.01
}

# ✅ Extract structured job and skill data
def extract_skills_from_text(text):
    data = {}

    # ✅ Extract Job Details
    if "Job ID" in text:
        data["Job ID"] = re.search(r"Job ID:\s*(.*)", text).group(1).strip() if re.search(r"Job ID:\s*(.*)", text) else None
        data["Title"] = re.search(r"Title:\s*(.*)", text).group(1).strip() if re.search(r"Title:\s*(.*)", text) else None

        skills_text = re.findall(r"- (.+?) \((Critical|Advanced|Intermediate)\)", text)
        data["Required Skills"] = {skill.strip(): level for skill, level in skills_text}

    # ✅ Extract Candidate Details
    if "Candidate ID" in text:
        data["Candidate ID"] = re.search(r"Candidate ID:\s*(.*)", text).group(1).strip() if re.search(r"Candidate ID:\s*(.*)", text) else None
        data["Name"] = re.search(r"Name:\s*(.*)", text).group(1).strip() if re.search(r"Name:\s*(.*)", text) else None
        data["Selected Job"] = re.search(r"Selected Job:\s*(.*)", text).group(1).strip() if re.search(r"Selected Job:\s*(.*)", text) else None

        # ✅ Extract Hard Skills properly from nested structure
        hard_skills_match = re.findall(r"Hard Skills:\s*\n((?:\s{4}.+?: .+\n)+)", text)
        if hard_skills_match:
            hard_skills_lines = hard_skills_match[0].strip().split("\n")
            data["Hard Skills"] = {line.split(":")[0].strip(): line.split(":")[1].strip() for line in hard_skills_lines}
        else:
            data["Hard Skills"] = {}

        # ✅ Extract Soft Skills
        soft_skills_match = re.search(r"Soft Skills:\s*(.*)", text)
        data["Soft Skills"] = soft_skills_match.group(1).strip().split(", ") if soft_skills_match else []

    return data


# ✅ Compute weighted skill similarity
def compute_weighted_skill_similarity(job_skills, candidate_skills, embedding_function, threshold=0.5):
    if not job_skills or not candidate_skills:
        return {}

    matched_skills = {}
    job_embeddings = embedding_function.embed_documents(list(job_skills.keys()))
    candidate_embeddings = embedding_function.embed_documents(candidate_skills)

    similarity_matrix = cosine_similarity(job_embeddings, candidate_embeddings)

    for i, (job_skill, skill_level) in enumerate(job_skills.items()):
        for j, candidate_skill in enumerate(candidate_skills):
            cosine_sim = similarity_matrix[i, j]
            levenshtein_sim = fuzz.ratio(job_skill.lower(), candidate_skill.lower()) / 100.0
            jaccard_sim = len(set(job_skill.lower().split()) & set(candidate_skill.lower().split())) / max(1, len(set(job_skill.lower().split()) | set(candidate_skill.lower().split())))

            final_score = max(cosine_sim, levenshtein_sim, jaccard_sim)

            if final_score >= threshold:
                weight = SKILL_WEIGHTS.get(skill_level, 1)  
                matched_skills[job_skill] = weight  

    return matched_skills

# ✅ Match and rank candidates for a job
def find_best_candidates(job, db_candidates, embedding_function, top_n=10):
    results = db_candidates.similarity_search(job["Title"], k=top_n)
    ranked_candidates = []
    
    for result in results:
        candidate = extract_skills_from_text(result.page_content)
        candidate_all_skills = list(candidate.get("Hard Skills", []))
        job_required_skills = job["Required Skills"]

        matched_skills = compute_weighted_skill_similarity(job_required_skills, candidate_all_skills, embedding_function, threshold=0.65)
        total_weighted_match = sum(matched_skills.values())
        soft_skill_bonus = sum(SOFT_SKILL_BONUS.get(skill, 0) for skill in candidate.get("Soft Skills", []))
        applied_boost = 0.1 if candidate.get("Selected Job") == job["Title"] else 0.0
        max_possible_weight = sum(SKILL_WEIGHTS.get(level, 1) for level in job_required_skills.values())
        exact_match_score = (total_weighted_match / max(1, max_possible_weight)) * 0.7
        final_score = round((exact_match_score + soft_skill_bonus + applied_boost), 2)

        ranked_candidates.append({
            "Name": candidate.get("Name", "N/A"),
            "Selected Job": candidate.get("Selected Job", "N/A"),
            "Matched Skills": ", ".join(matched_skills.keys()) if matched_skills else "None",
            "Weighted Skill Score": round(exact_match_score, 2),
            "Soft Skills": ", ".join(candidate.get("Soft Skills", [])) if candidate.get("Soft Skills", []) else "None",
            "Soft Skill Score": round(soft_skill_bonus, 2),
            "Applied Boost": applied_boost,
            "Final Score": final_score,
        })

    return sorted(ranked_candidates, key=lambda x: x["Final Score"], reverse=True)[:top_n]

# ✅ Generate job-specific tables
def generate_job_specific_tables(db_jobs, db_candidates, embedding_function):
    job_texts = db_jobs.similarity_search("Job", k=50)
    job_offers = [extract_skills_from_text(job.page_content) for job in job_texts]
    selected_job = st.selectbox("📌 Select a job to display candidates:", ["Select"] + [job["Title"] for job in job_offers])

    if selected_job != "Select":
        job = next(job for job in job_offers if job["Title"] == selected_job)
        ranked_candidates = find_best_candidates(job, db_candidates, embedding_function, top_n=10)
        df = pd.DataFrame(ranked_candidates)
        st.dataframe(df)

if __name__ == "__main__":
    st.title("Job Matching System")
    db_jobs, db_candidates, embedding_function = setup_chromadb()
    generate_job_specific_tables(db_jobs, db_candidates, embedding_function)
