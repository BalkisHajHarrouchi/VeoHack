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




# Ensure Ollama runs on GPU (CUDA)
os.environ["OLLAMA_USE_CUDA"] = "1"

# File paths for saving templates
ACCEPT_TEMPLATE_FILE = "prompt_temp_accept.txt"
REJECT_TEMPLATE_FILE = "prompt_temp_reject.txt"

# Function to read file content
def read_template(file_path):
    """Read the content of the template file, return an empty string if file doesn't exist."""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read().strip()
    except FileNotFoundError:
        return ""

# Function to save revised message content to a file
def save_template(file_path, content):
    """Save content to the specified template file, overwriting old content."""
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(content)

# Function to clear the content of the selected template file
def reset_template(email_type):
    """Clear the content of the selected template file based on the email type."""
    if email_type == "Accepted":
        with open(ACCEPT_TEMPLATE_FILE, "w", encoding="utf-8") as file:
            file.write("")
        st.success("Acceptance template has been reset successfully!")
    elif email_type == "Rejected":
        with open(REJECT_TEMPLATE_FILE, "w", encoding="utf-8") as file:
            file.write("")
        st.success("Rejection template has been reset successfully!")

# Function to refine the message using Llama 3
def refine_message(message):
    """Use Llama 3 to revise the message, ensuring it meets professional standards and is concise."""
    revision_prompt = (
        "Revise the following email to:\n"
        "- Ensure it meets all professional email standards\n"
        "- Be short and to the point (avoid unnecessary details)\n"
        "- Maintain a formal and respectful tone\n"
        "- Include any mentioned dates, interview duration, and location\n"
        "- Exclude any form of advice or personal recommendations\n\n"
        f"Original Message:\n{message}\n\n"
        "Provide only the improved version without explanations."
    )

    with st.spinner("Refining message for professionalism and conciseness..."):
        response = ollama.chat(model="llama3", messages=[{"role": "user", "content": revision_prompt}])
        refined_content = response["message"]["content"] if response else message  # If LLM fails, keep original

    return refined_content

# Function to extract subject and content from the generated email
def extract_subject_and_content(email_text):
    """Extract the subject and content from the generated email text."""
    subject = ""
    content = ""

    # Remove any extraneous text like "Here is the final acceptance email:"
    email_text = email_text.replace("Here is the final acceptance email:", "").strip()

    # Extract subject (everything after "Subject:" and before "Dear")
    subject_start = email_text.find("Subject:")
    dear_start = email_text.find("Dear")
    
    if subject_start != -1 and dear_start != -1:
        subject = email_text[subject_start + len("Subject:"):dear_start].strip()
    
    # Extract content (everything from "Dear" till the end)
    if dear_start != -1:
        content = email_text[dear_start:].strip()

    return subject, content

# Read existing templates
acceptance_template = read_template(ACCEPT_TEMPLATE_FILE)
rejection_template = read_template(REJECT_TEMPLATE_FILE)

# Define **NEW** email prompts
ACCEPTANCE_PROMPT = (
    "You are an HR assistant responsible for generating a professional acceptance email.\n\n"
    "Your task is to write a formal and concise acceptance email to a candidate who has been selected for the job.\n"
    "Make sure the email respects all professional standards:\n"
    "- Use a clear and professional tone\n"
    "- Be short and to the point\n"
    "- Include any mentioned interview date, duration, and location\n"
    "- No advice or personal recommendations should be given\n\n"
    "The key points for this email should be **extracted from the following saved notes**:\n"
    "{example}\n\n"
    "The interview details are as follows:\n"
    "- Date: {interview_date}\n"
    "- Duration: {interview_duration}\n"
    "- Location: {interview_location}\n\n"
    "Generate the final email based on these key points and details."
)

REJECTION_PROMPT = (
    "You are an HR assistant responsible for generating a professional rejection email.\n\n"
    "Your task is to write a polite and empathetic rejection email to a candidate who has not been selected.\n"
    "Make sure the email is professional and respectful, following these guidelines:\n"
    "- Maintain a formal yet compassionate tone, acknowledging the candidate’s effort\n"
    "- Be short and to the point\n"
    "- Include any mentioned interview date, duration, and location\n"
    "- No advice or personal recommendations should be given\n\n"
    "The key points for this email should be **extracted from the following saved notes**:\n"
    "{example}\n\n"
    "The interview details are as follows:\n"
    "- Date: {interview_date}\n"
    "- Duration: {interview_duration}\n"
    "- Location: {interview_location}\n\n"
    "Generate the final email based on these key points and details."
)

# Function to generate email content and subject using Llama 3
def generate_email(prompt_template, example_message, interview_date, interview_duration, interview_location):
    """Generate email content and subject based on the stored key points and interview details."""
    
    # Format the selected prompt with the extracted key points and interview details
    prompt = prompt_template.format(
        example=example_message,
        interview_date=interview_date,
        interview_duration=interview_duration,
        interview_location=interview_location
    )

    # Generate email content and subject using Llama 3
    with st.spinner("Generating professional email..."):
        response = ollama.chat(model="llama3", messages=[{"role": "user", "content": prompt}])
        generated_content = response["message"]["content"] if response else "Failed to generate content."

        # Extract subject and content from the generated email
        subject, content = extract_subject_and_content(generated_content)

    return content, subject

def email_interface():
    st.title("Send Email")

    # Store generated content and subject in session state
    if "generated_content" not in st.session_state:
        st.session_state.generated_content = ""
    if "generated_subject" not in st.session_state:
        st.session_state.generated_subject = ""

    # Select rejection or acceptance
    email_type = st.selectbox("Select Email Type", ["Accepted", "Rejected"])

    # Date, duration, and location inputs for interview
    interview_date = st.date_input("Interview Date (if fixed)")
    interview_duration = st.text_input("Interview Duration (e.g., '30 minutes')")
    interview_location = st.text_input("Interview Location (e.g., 'Company HQ, Room 405')")

    # Choose the correct template file based on selection
    selected_template = acceptance_template if email_type == "Accepted" else rejection_template

    # Generate button
    if st.button("Generate"):
        selected_prompt = ACCEPTANCE_PROMPT if email_type == "Accepted" else REJECTION_PROMPT
        st.session_state.generated_content, st.session_state.generated_subject = generate_email(
            selected_prompt, selected_template, interview_date, interview_duration, interview_location
        )
        st.rerun()

    # Email form (without "To" input, with "Save" button)
    with st.form("email_form"):
        # Set the subject field with the generated subject
        subject = st.text_input("Subject:", value=st.session_state.generated_subject)
        message = st.text_area("Message:", value=st.session_state.generated_content)
        save = st.form_submit_button("Save")  # Changed from "Send" to "Save"
        
        if save:
            # Refine the message before saving
            revised_message = refine_message(message)

            # Save the revised message to the correct file
            file_path = ACCEPT_TEMPLATE_FILE if email_type == "Accepted" else REJECT_TEMPLATE_FILE
            save_template(file_path, revised_message)

            st.success("Message revised and saved successfully! Future emails will use this refined version as a reference.")

    # Reset button
    if st.button("Reset Template"):
        reset_template(email_type)
        st.rerun()
import streamlit as st
import pandas as pd

# Load jobs from file
def load_jobs(filename="jobs (1).txt"):
    try:
        with open(filename, "r", encoding="utf-8") as file:
            jobs = {line.strip(): 0 for line in file.readlines() if line.strip()}
        return jobs
    except FileNotFoundError:
        st.error("jobs.txt not found!")
        return {}

# Load candidates from file
def load_candidates(filename="candidates.txt"):
    try:
        candidates = []
        with open(filename, "r", encoding="utf-8") as file:
            for line in file.readlines():
                parts = [x.strip() for x in line.split(",")]
                if len(parts) == 4:
                    name, email, score, job = parts
                    candidates.append({"name": name, "email": email, "score": int(score), "job": job})
        return candidates
    except FileNotFoundError:
        st.error("candidates.txt not found!")
        return []

# Match candidates to jobs
def match_candidates_to_jobs(jobs, candidates):
    job_assignments = {job: [] for job in jobs.keys()}
    for candidate in candidates:
        if candidate["job"] in job_assignments:
            job_assignments[candidate["job"]].append(candidate)

    for job in job_assignments:
        job_assignments[job] = sorted(job_assignments[job], key=lambda x: x["score"], reverse=True)

    return job_assignments

# Function to send emails
def send_emails(job, candidates, num_highlight):
    highlighted_candidates = candidates[:num_highlight]
    for candidate in highlighted_candidates:
        st.success(f"Email sent to: {candidate['name']} ({candidate['email']}) for {job}")

# Streamlit UI
st.title("Job Matching System")

# Load data from files
jobs = load_jobs()
candidates = load_candidates()
job_assignments = match_candidates_to_jobs(jobs, candidates)

# Job selection dropdown
selected_job = st.selectbox("Select a Job", ["All Jobs"] + list(jobs.keys()))

if selected_job == "All Jobs":
    for job, candidates in job_assignments.items():
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            st.write(f"### {job}")
        with col2:
            num_highlight = st.number_input(f"Highlight top candidates for {job}", min_value=1, max_value=100, value=3, key=job)
        with col3:
            if st.button(f"Send Emails ({job})", key=f"send_{job}"):
                send_emails(job, candidates, num_highlight)

        if candidates:
            job_table = pd.DataFrame(candidates)
            job_table = job_table.rename(columns=lambda x: x.strip())  # Strip spaces from column names
            job_table["Highlighted"] = ["✅" if i < num_highlight else "" for i in range(len(job_table))]

            st.write("Columns in DataFrame:", job_table.columns.tolist())  # Debugging

            display_columns = [col for col in ["name", "email", "score", "Highlighted"] if col in job_table.columns]
            st.table(job_table[display_columns])
        else:
            st.warning(f"No candidates found for {job}.")

else:
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        st.write(f"### {selected_job}")
    with col2:
        num_highlight = st.number_input(f"Highlight top candidates for {selected_job}", min_value=1, max_value=100, value=3, key=selected_job)
    with col3:
        if st.button(f"Send Emails ({selected_job})", key=f"send_{selected_job}"):
            send_emails(selected_job, job_assignments[selected_job], num_highlight)

    if job_assignments[selected_job]:
        job_table = pd.DataFrame(job_assignments[selected_job])
        job_table = job_table.rename(columns=lambda x: x.strip())  # Strip spaces from column names
        job_table["Highlighted"] = ["✅" if i < num_highlight else "" for i in range(len(job_table))]

        st.write("Columns in DataFrame:", job_table.columns.tolist())  # Debugging

        display_columns = [col for col in ["name", "email", "score", "Highlighted"] if col in job_table.columns]
        st.table(job_table[display_columns])
    else:
        st.warning(f"No candidates found for {selected_job}.")
