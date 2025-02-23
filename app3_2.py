import streamlit as st
import pandas as pd
import re
from langchain_chroma import Chroma  # ✅ Fix ChromaDB Import
from langchain.embeddings import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

st.set_page_config(layout="wide")


# ✅ Soft Skill Weighting System (Assign Scores to Soft Skills)
SOFT_SKILL_BONUS = {
    "Communication": 0.10,
    "Problem-Solving": 0.07,
    "Leadership": 0.05,
    "Creativity": 0.02,
    "Team Work": 0.08,
    "Time Management": 0.06,
    "Skill Development": 0.04,
    "Attention to Details": 0.05,
    "Adaptability": 0.06,
    "Innovation": 0.03
}



# ✅ Load the embedding model
EMBEDDING_MODEL = "Lajavaness/bilingual-embedding-large"
model_kwargs = {"trust_remote_code": True}
embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs=model_kwargs)

# ✅ Setup ChromaDB
def setup_chromadb():
    db_jobs = Chroma(persist_directory="emb/jobsDB", embedding_function=embedding_function)
    db_candidates = Chroma(persist_directory="emb/candidatesDB", embedding_function=embedding_function)
    return db_jobs, db_candidates, embedding_function

# ✅ Extract job details from ChromaDB
def extract_job_details_from_text(text):
    """Extract job details from stored job descriptions."""
    data = {}

    # ✅ Extract Job Title
    title_match = re.search(r"Job Title:\s*(.*)", text)
    data["Title"] = title_match.group(1).strip() if title_match else "Unknown Job Title"

    # ✅ Extract Experience Requirement
    experience_match = re.search(r"Experience Required:\s*(.*)", text)
    data["Experience Required"] = experience_match.group(1).strip() if experience_match else "Not Specified"

    # ✅ Extract Education Requirement
    edu_match = re.search(r"Education:\s*(.*)", text)
    data["Education"] = edu_match.group(1).strip() if edu_match else "Not Specified"

    # ✅ Extract Required Skills
    skills_match = re.findall(r"- (.+)", text.split("Required Skills:")[-1]) if "Required Skills:" in text else []
    data["Required Skills"] = skills_match if skills_match else []

    # ✅ Extract Certifications
    certs_match = re.findall(r"- (.+)", text.split("Certifications:")[-1]) if "Certifications:" in text else []
    data["Certifications"] = certs_match if certs_match else []

    return data

# ✅ Extract candidate details from ChromaDB
def extract_candidate_details_from_text(text):
    """Extract candidate details from structured candidate profiles."""
    data = {}

    # ✅ Extract Candidate ID & Name
    id_match = re.search(r"Candidate ID:\s*(.*)", text)
    data["Candidate ID"] = id_match.group(1).strip() if id_match else "Unknown Candidate ID"

    name_match = re.search(r"Name:\s*(.*)", text)
    data["Name"] = name_match.group(1).strip() if name_match else "Unknown Candidate"

    # ✅ Extract Contact Information
    email_match = re.search(r"Email:\s*(.*)", text)
    phone_match = re.search(r"Phone:\s*(.*)", text)
    linkedin_match = re.search(r"LinkedIn:\s*(.*)", text)

    data["Contact"] = {
        "Email": email_match.group(1).strip() if email_match else "Not Provided",
        "Phone": phone_match.group(1).strip() if phone_match else "Not Provided",
        "LinkedIn": linkedin_match.group(1).strip() if linkedin_match else "Not Provided",
    }

    # ✅ Extract Hard & Soft Skills (Nested inside "Skills:")
    skills_match = re.search(r"Skills:\s*(.*?)\nExperience:", text, re.DOTALL)
    hard_skills = {}
    soft_skills = []

    if skills_match:
        skills_text = skills_match.group(1)

        # ✅ Extract Hard Skills
        hard_skills_section = re.search(r"Hard Skills:\s*((?:\s{4}.+\n?)+)", skills_text)
        if hard_skills_section:
            hard_skills_lines = hard_skills_section.group(1).strip().split("\n")
            for line in hard_skills_lines:
                skill_match = re.match(r"\s{4}(.+):\s*(.*)", line)
                if skill_match:
                    skill, level = skill_match.groups()
                    hard_skills[skill.strip()] = level.strip() if level.strip() else "N/A"

        # ✅ Extract Soft Skills
        soft_skills_match = re.search(r"Soft Skills:\s*(.*)", skills_text)
        soft_skills = soft_skills_match.group(1).strip().split(", ") if soft_skills_match else []

    data["Skills"] = {
        "Hard Skills": hard_skills,
        "Soft Skills": soft_skills
    }

    # ✅ Extract Experience (If Available)
    experience_match = re.findall(r"Experience:\s*(.*?)\n", text)
    data["Experience"] = experience_match if experience_match else "Not Provided"

    # ✅ Extract Education (If Available)
    education_match = re.search(r"Education:\s*(.*?)\n", text)
    data["Education"] = education_match.group(1).strip() if education_match else "Not Provided"

    # ✅ Extract Preferred Roles (If Available)
    preferred_roles_match = re.findall(r"- (.+)", text.split("Preferred Roles:")[-1]) if "Preferred Roles:" in text else []
    data["Preferred Roles"] = preferred_roles_match if preferred_roles_match else []

    return data


# ✅ Compute weighted skill similarity using Cosine Similarity
def compute_weighted_skill_similarity(job_skills, candidate_skills, embedding_function, threshold=0.65):
    if not job_skills or not candidate_skills:
        return {}

    matched_skills = {}
    job_embeddings = embedding_function.embed_documents(job_skills)
    candidate_embeddings = embedding_function.embed_documents(candidate_skills)

    similarity_matrix = cosine_similarity(job_embeddings, candidate_embeddings)

    for i, job_skill in enumerate(job_skills):
        for j, candidate_skill in enumerate(candidate_skills):
            cosine_sim = similarity_matrix[i, j]

            if cosine_sim >= threshold:
                matched_skills[job_skill] = round(cosine_sim, 2)

    return matched_skills

# ✅ Match and rank candidates for a job
def find_best_candidates(job, db_candidates, top_n=10):
    results = db_candidates.similarity_search(job["Title"], k=top_n)
    st.write(results)  # This will display the results so you can check if they match.

    ranked_candidates = []

    for result in results:
        candidate = extract_candidate_details_from_text(result.page_content)

        # ✅ Fix: Use `.get()` to prevent KeyErrors
        candidate_hard_skills = candidate.get("Skills", {}).get("Hard Skills", {})
        candidate_hard_skills_list = list(candidate_hard_skills.keys())

        matched_skills = compute_weighted_skill_similarity(
            job["Required Skills"], candidate_hard_skills_list, embedding_function, threshold=0.65
        )

        total_weighted_match = sum(matched_skills.values())
        soft_skill_bonus = sum(SOFT_SKILL_BONUS.get(skill, 0) for skill in candidate.get("Skills", {}).get("Soft Skills", []))
        applied_boost = 0.1 if job["Title"] in candidate.get("Preferred Roles", []) else 0.0
        final_score = round((total_weighted_match * 0.7 + soft_skill_bonus + applied_boost), 2)

        ranked_candidates.append({
            "Name": candidate.get("Name", "N/A"),
            "Experience": candidate.get("Experience", "Not Provided"),  # ✅ Fix: Prevents KeyError
            "Education": candidate.get("Education", "Not Provided"),  # ✅ Fix: Prevents KeyError
            "Matched Skills": ", ".join(matched_skills.keys()) if matched_skills else "None",
            "Skill Score": total_weighted_match,
            "Soft Skills": ", ".join(candidate.get("Skills", {}).get("Soft Skills", [])) if candidate.get("Skills", {}).get("Soft Skills") else "None",
            "Soft Skill Score": soft_skill_bonus,
            "Applied Boost": applied_boost,
            "Final Score": final_score,
        })

    return sorted(ranked_candidates, key=lambda x: x["Final Score"], reverse=True)[:top_n]




if __name__ == "__main__":
    st.title("Job Matching System")
    db_jobs, db_candidates, embedding_function = setup_chromadb()

    # ✅ Load jobs from ChromaDB
    job_texts = db_jobs.similarity_search("Job", k=10)
    job_offers = [extract_job_details_from_text(job.page_content) for job in job_texts]

    # ✅ Prevent crash if no jobs are found
    if not job_offers:
        st.error("⚠️ No jobs found in ChromaDB. Please add jobs first.")
    else:
        selected_job = st.selectbox("📌 Select a job:", ["Select"] + [job["Title"] for job in job_offers])

        if selected_job != "Select":
            job = next((job for job in job_offers if job["Title"] == selected_job), None)

            if job:
                # ✅ NEW: Select number of candidates to display
                num_candidates = st.selectbox("📊 How many candidates to display?", [1,2,3,4,5,6,7,8,9,10], index=1)

                # ✅ Find candidates based on selected number
                ranked_candidates = find_best_candidates(job, db_candidates, top_n=num_candidates)

                # ✅ Display table with selected number of candidates
                df = pd.DataFrame(ranked_candidates)
                st.dataframe(df)
