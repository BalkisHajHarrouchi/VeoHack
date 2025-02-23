
import streamlit as st
import ollama
import os

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

if __name__ == "__main__":
    email_interface()