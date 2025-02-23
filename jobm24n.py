import streamlit as st
import pandas as pd
import torch
import json
import re
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz
import numpy as np
import langchain
from datetime import datetime
from langchain.llms import Ollama

langchain.debug = True

# ✅ Configure Streamlit page layout
st.set_page_config(layout="wide")

# ✅ Check if CUDA is available
USE_CUDA = torch.cuda.is_available()
device = "cuda" if USE_CUDA else "cpu"
st.write(f"Using device: {device}")

# ✅ Load the embedding model with CUDA if available
EMBEDDING_MODEL = "Lajavaness/bilingual-embedding-large"
model_kwargs = {"trust_remote_code": True, "device": device}
embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs=model_kwargs)

# ✅ Setup Llama 3 from Ollama
llm = Ollama(model="llama3")

# ✅ Setup ChromaDB
def setup_chromadb():
    db_candidates = Chroma(persist_directory="emb/candidatesDB", embedding_function=embedding_function)
    return db_candidates, embedding_function

# ✅ Define Skill & Tool Mappings
SKILL_MAPPING = {
    "Backend Development": ["Python", "Flask", "Django", "Node.js", "Spring Boot", "Java", "C#"],
    "Database Management": ["SQL", "PostgreSQL", "MongoDB", "MySQL", "NoSQL"],
    "AI Engineering": ["TensorFlow", "PyTorch", "Scikit-Learn", "NLP", "Deep Learning"],
    "Frontend Development": ["React", "Vue.js", "Angular", "HTML", "CSS", "JavaScript"]
}

# ✅ Use Llama 3 to extract job details
def extract_job_details_llm(job_text):
    prompt = f"""
    Extract structured job details from the following job description:
    {job_text}
    
    Return ONLY a valid JSON object with the following format:
    {{
        "Title": "Job Title",
        "Required Skills": ["Skill1", "Skill2", "Skill3"],
        "Experience Required": "X years",
        "Education": "Required Degree",
        "Certifications": ["Certification1", "Certification2"]
    }}
    Do NOT include any extra text outside the JSON format.
    """
    
    response = llm.invoke(prompt)
    
    # Extract JSON part only
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if json_match:
        json_content = json_match.group(0)
        try:
            job_data = json.loads(json_content)
            return job_data
        except json.JSONDecodeError:
            st.error("Error: Failed to parse job details. Invalid JSON format returned by Llama 3.")
            return {}
    else:
        st.error("Error: No valid JSON detected in Llama 3 response.")
        return {}

# ✅ Extract structured candidate data
def extract_skills_from_text(text):
    data = {}
    if "Candidate ID" in text:
        data["Candidate ID"] = re.search(r"Candidate ID:\s*(.*)", text).group(1).strip() if re.search(r"Candidate ID:\s*(.*)", text) else None
        data["Name"] = re.search(r"Name:\s*(.*)", text).group(1).strip() if re.search(r"Name:\s*(.*)", text) else None
        data["Selected Job"] = re.search(r"Selected Job:\s*(.*)", text).group(1).strip() if re.search(r"Selected Job:\s*(.*)", text) else None
        
        hard_skills_match = re.findall(r"Hard Skills:\s*\n((?:\s{4}.+?: .+\n)+)", text)
        if hard_skills_match:
            hard_skills_lines = hard_skills_match[0].strip().split("\n")
            data["Hard Skills"] = {line.split(":")[0].strip(): line.split(":")[1].strip() for line in hard_skills_lines}
        else:
            data["Hard Skills"] = {}
        
        soft_skills_match = re.search(r"Soft Skills:\s*(.*)", text)
        data["Soft Skills"] = soft_skills_match.group(1).strip().split(", ") if soft_skills_match else []
        
    return data

# ✅ Enhanced Skill Matching with Tool Recognition
def expand_skills(job_skills):
    expanded_skills = set(job_skills)
    for skill in job_skills:
        if skill in SKILL_MAPPING:
            expanded_skills.update(SKILL_MAPPING[skill])
    return list(expanded_skills)

# ✅ Compute weighted skill similarity using Cosine Similarity + Fuzzy Matching
def compute_weighted_skill_similarity(job_skills, candidate_skills, embedding_function, threshold=0.6):
    if not job_skills or not candidate_skills:
        return {}

    expanded_job_skills = expand_skills(job_skills)
    matched_skills = {}

    job_embeddings = embedding_function.embed_documents(expanded_job_skills)
    candidate_embeddings = embedding_function.embed_documents(candidate_skills)

    similarity_matrix = cosine_similarity(job_embeddings, candidate_embeddings)

    for i, job_skill in enumerate(expanded_job_skills):
        for j, candidate_skill in enumerate(candidate_skills):
            cosine_sim = similarity_matrix[i, j]
            fuzzy_match_score = fuzz.ratio(job_skill.lower(), candidate_skill.lower()) / 100.0
            combined_score = max(cosine_sim, fuzzy_match_score)

            if combined_score >= threshold:
                matched_skills[job_skill] = round(combined_score, 2)

    return matched_skills


def find_best_candidates(job, db_candidates, top_n=10):
    results = db_candidates.similarity_search(job["Title"], k=top_n)
    ranked_candidates = []

    for result in results:
        candidate = extract_skills_from_text(result.page_content)
        matched_skills = compute_weighted_skill_similarity(job["Required Skills"], list(candidate.get("Hard Skills", [])), embedding_function, threshold=0.65)
        total_weighted_match = sum(matched_skills.values())
        soft_skill_bonus = sum(0.05 for skill in candidate.get("Soft Skills", []))  # Basic soft skill bonus
        
        final_score = round((total_weighted_match * 0.7 + soft_skill_bonus ), 2)

        ranked_candidates.append({
            "Name": candidate.get("Name", "N/A"),
            "Matched Skills": ", ".join(matched_skills.keys()) if matched_skills else "None",
            "Skill Score": total_weighted_match,
            "Soft Skills": ", ".join(candidate.get("Soft Skills", [])) if candidate.get("Soft Skills", []) else "None",
            "Soft Skill Score": soft_skill_bonus,
            
            "Final Score": final_score,
        })

    return sorted(ranked_candidates, key=lambda x: x["Final Score"], reverse=True)[:top_n] 
# ✅ Streamlit UI
if __name__ == "__main__":
    st.title("Job Matching System")
    db_candidates, embedding_function = setup_chromadb()
    
    job_text = st.text_area("Paste Job Description Here:")
    
    if job_text:
        job_details = extract_job_details_llm(job_text)
        if job_details:
            job_details["Required Skills"] = expand_skills(job_details["Required Skills"])
            st.write("### Extracted & Expanded Job Details:", job_details)
            ranked_candidates = find_best_candidates(job_details, db_candidates)
            df = pd.DataFrame(ranked_candidates)
            st.dataframe(df)
