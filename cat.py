import streamlit as st
import pandas as pd
import re
import os
import logging
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from email.mime.text import MIMEText
import base64

import ollama


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load job categories and required skills
def load_jobs():
    """Loads job categories and required skills from jobs (1).txt."""
    job_categories = {}
    try:
        with open("jobs (1).txt", "r", encoding="utf-8") as file:
            jobs = file.read().split("\n\n")
            for job in jobs:
                job_data = job.split("\n")
                title = None
                required_skills = {}
                for line in job_data:
                    if line.startswith("Title:"):
                        title = line.split(": ")[1].strip()
                    elif line.startswith("  - "):
                        skill_line = line.replace("  - ", "").strip()
                        skill_match = re.match(r"(.+) \((Critical|Advanced|Intermediate)\)", skill_line)
                        if skill_match:
                            skill, level = skill_match.groups()
                            required_skills[skill] = level
                if title:
                    job_categories[title] = required_skills
    except Exception as e:
        logger.error(f"Error loading jobs: {e}")
    return job_categories

# Load candidates from candidates.txt
def load_candidates():
    """Loads candidates from candidates.txt dynamically."""
    candidates = []
    try:
        with open("candidates.txt", "r", encoding="utf-8") as file:
            candidates_data = file.read().split("\n\n")
            for candidate in candidates_data:
                candidate_info = {}
                hard_skills = {}
                lines = candidate.split("\n")
                for i, line in enumerate(lines):
                    if ":" in line:
                        key, value = line.split(":", 1)
                        key = key.strip()
                        value = value.strip()
                        if key == "Skills":
                            for j in range(i + 1, len(lines)):
                                if "Soft Skills" in lines[j]:
                                    break
                                skill_match = re.match(r"\s+(.+): (Advanced|Intermediate)", lines[j])
                                if skill_match:
                                    skill, level = skill_match.groups()
                                    hard_skills[skill] = level
                        else:
                            candidate_info[key] = value
                if hard_skills:
                    candidate_info["Hard Skills"] = hard_skills
                candidates.append(candidate_info)
    except Exception as e:
        logger.error(f"Error loading candidates: {e}")
    return candidates

# Match candidates to jobs
def match_candidates_to_jobs(jobs, candidates):
    """Matches candidates to all jobs and ranks them based on score."""
    job_assignments = {job: [] for job in jobs}
    for candidate in candidates:
        if "Hard Skills" in candidate:
            candidate_skills = candidate["Hard Skills"]
            scores = {}
            for job, required_skills in jobs.items():
                score = 0
                for skill, level in required_skills.items():
                    if skill in candidate_skills:
                        if level == "Critical":
                            score += 3
                        elif level == "Advanced":
                            score += 2
                        elif level == "Intermediate":
                            score += 1
                scores[job] = score
            sorted_jobs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            for job, score in sorted_jobs:
                job_assignments[job].append((candidate['Name'], candidate.get('Email', ''), score))
    
    # Sort candidates within each job based on score (highest to lowest)
    for job in job_assignments:
        job_assignments[job].sort(key=lambda x: x[2], reverse=True)
    
    return job_assignments

# Send emails via Gmail API
def send_emails(job_title, candidates, num_candidates):
    """Send interview emails via Gmail API to top N candidates."""
    SCOPES = ["https://www.googleapis.com/auth/gmail.send"]
    creds = None
    try:
        if os.path.exists("token.json"):
            creds = Credentials.from_authorized_user_file("token.json", SCOPES)
        else:
            flow = InstalledAppFlow.from_client_secrets_file("client_secret.json", SCOPES)
            creds = flow.run_local_server(port=0)
            with open("token.json", "w") as token:
                token.write(creds.to_json())
        
        service = build("gmail", "v1", credentials=creds)
        selected_candidates = candidates[:num_candidates]
        
        for name, email, score in selected_candidates:
            if email:
                try:
                    subject = f"Interview Invitation for {job_title}"
                    body = f"Dear {name},\n\nYou have been selected for an interview for the position of {job_title} at VEO.\n\nBest regards,\nHR Team"
                    message = MIMEText(body)
                    message["to"] = email
                    message["subject"] = subject
                    raw = base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")
                    message = {"raw": raw}
                    
                    # Log email content
                    logger.info(f"Email content for {name} ({email}):\nSubject: {subject}\nBody: {body}")
                    
                    # Send email
                    sent_message = service.users().messages().send(userId="me", body=message).execute()
                    logger.info(f"Email sent to {name} ({email}): {sent_message}")
                    st.success(f"Email sent to {name} for {job_title}")
                except Exception as e:
                    logger.error(f"Failed to send email to {name} ({email}): {e}")
                    st.error(f"Failed to send email to {name}: {e}")
            else:
                logger.warning(f"No email address found for {name}")
                st.warning(f"No email address found for {name}")
    except Exception as e:
        logger.error(f"Error setting up Gmail API: {e}")
        st.error(f"Error setting up Gmail API: {e}")
    """Send interview emails via Gmail API to top N candidates."""
    SCOPES = ["https://www.googleapis.com/auth/gmail.send"]
    creds = None
    try:
        if os.path.exists("token.json"):
            creds = Credentials.from_authorized_user_file("token.json", SCOPES)
        else:
            flow = InstalledAppFlow.from_client_secrets_file("client_secret.json", SCOPES)
            creds = flow.run_local_server(port=0)
            with open("token.json", "w") as token:
                token.write(creds.to_json())
        
        service = build("gmail", "v1", credentials=creds)
        selected_candidates = candidates[:num_candidates]
        
        for name, email, score in selected_candidates:
            if email:
                try:
                    subject = f"Interview Invitation for {job_title}"
                    body = f"Dear {name},\n\nYou have been selected for an interview for the position of {job_title} at VEO.\n\nBest regards,\nHR Team"
                    message = MIMEText(body)
                    message["to"] = email
                    message["subject"] = subject
                    raw = base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")
                    message = {"raw": raw}
                    service.users().messages().send(userId="me", body=message).execute()
                    logger.info(f"Email sent to {name} ({email}) for {job_title}")
                    st.success(f"Email sent to {name} for {job_title}")
                except Exception as e:
                    logger.error(f"Failed to send email to {name} ({email}): {e}")
                    st.error(f"Failed to send email to {name}: {e}")
            else:
                logger.warning(f"No email address found for {name}")
                st.warning(f"No email address found for {name}")
    except Exception as e:
        logger.error(f"Error setting up Gmail API: {e}")
        st.error(f"Error setting up Gmail API: {e}")

# Streamlit UI
st.title("Job Matching System")

# Load data
jobs = load_jobs()
candidates = load_candidates()
job_assignments = match_candidates_to_jobs(jobs, candidates)

# Job selection
selected_job = st.selectbox("Select a Job", ["All Jobs"] + list(jobs.keys()))

if selected_job == "All Jobs":
    for job, candidates in job_assignments.items():
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            st.write(f"### {job}")
        with col2:
            num_highlight = st.number_input(f"Highlight top candidates for {job}", min_value=1, max_value=100, value=3, key=job)
        with col3:
            if st.button(f"Send Emails ({job})"):
                send_emails(job, candidates, num_highlight)
        job_table = pd.DataFrame(candidates, columns=["Candidate Name", "Email", "Score"])
        job_table["Highlighted"] = ["✅" if i < num_highlight else "" for i in range(len(job_table))]
        st.table(job_table)
else:
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        st.write(f"### {selected_job}")
    with col2:
        num_highlight = st.number_input(f"Highlight top candidates for {selected_job}", min_value=1, max_value=100, value=3, key=selected_job)
    with col3:
        if st.button(f"Send Emails ({selected_job})"):
            send_emails(selected_job, job_assignments[selected_job], num_highlight)
    job_table = pd.DataFrame(job_assignments[selected_job], columns=["Candidate Name", "Email", "Score"])
    job_table["Highlighted"] = ["✅" if i < num_highlight else "" for i in range(len(job_table))]
    st.table(job_table)
