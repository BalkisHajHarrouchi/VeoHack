import streamlit as st
import pdfplumber
import docx
import easyocr
import json
import os
import numpy as np
from PIL import Image
from langchain_community.llms import Ollama
import torch
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
import langchain
import os
import shutil


# Set device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Print the device being used
if torch.cuda.is_available():
    print(f"✅ Using CUDA (GPU): {torch.cuda.get_device_name(0)}")
else:
    print("⚠️ CUDA is not available. Using CPU.")

# Initialize OCR for image processing
reader = easyocr.Reader(["en", "fr"])

st.title("📂 AI-Powered CV Processor & JSON Formatter with LangChain")

# Upload multiple CVs
uploaded_files = st.file_uploader(
    "Upload CVs (PDF, DOCX, Images)", 
    type=["pdf", "docx", "jpg", "jpeg", "png"], 
    accept_multiple_files=True
)

# Function to extract text from PDFs
def extract_text_from_pdf(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    return text.strip()

# Function to extract text from DOCX
def extract_text_from_docx(docx_file):
    doc = docx.Document(docx_file)
    return "\n".join([para.text for para in doc.paragraphs]).strip()

# Function to extract text from images using OCR
def extract_text_from_image(image):
    image = np.array(image)  # Convert PIL image to OpenCV format
    text = reader.readtext(image, detail=0, paragraph=True)  # Extract structured text
    return "\n".join(text)


def format_candidate_data(candidates):
    formatted_text = ""
    for candidate in candidates:
        formatted_text += f"Candidate ID: {candidate.get('id', 'N/A')}\n"
        formatted_text += f"Name: {candidate.get('name', 'N/A')}\n"
        
        # Contact Information
        formatted_text += "Contact:\n"
        contact = candidate.get("contact", {})
        formatted_text += f"  Email: {contact.get('email', 'N/A')}\n"
        formatted_text += f"  Phone: {contact.get('phone', 'N/A')}\n"
        formatted_text += f"  LinkedIn: {contact.get('linkedin', 'N/A')}\n"
        
        # Skills
        formatted_text += "Skills:\n"
        skills = candidate.get("skills", {})
        
        # Hard Skills
        hard_skills = skills.get("hard_skills", {})
        if hard_skills:
            formatted_text += "  Hard Skills:\n"
            for skill, level in hard_skills.items():
                formatted_text += f"    {skill}: {level if level else 'N/A'}\n"
        
        # Soft Skills
        soft_skills = skills.get("soft_skills", [])
        if soft_skills:
            formatted_text += f"  Soft Skills: {', '.join(soft_skills)}\n"

        # Experience
        formatted_text += "Experience:\n"
        experience = candidate.get("experience", [])
        for job in experience:
            formatted_text += f"  - Job Title: {job.get('job_title', 'N/A')}\n"
            formatted_text += f"    Company: {job.get('company', 'N/A')}\n"
            formatted_text += f"    Duration: {job.get('duration', 'N/A')}\n"

        # Education
        formatted_text += "Education:\n"
        education = candidate.get("education", [])
        if isinstance(education, list):
            for edu in education:
                formatted_text += f"  - {edu.get('Degree', 'N/A')} from {edu.get('University', 'N/A')} ({edu.get('Year', 'N/A')})\n"
        else:
            formatted_text += f"  {education}\n"

        # Certifications
        certifications = candidate.get("certifications", [])
        if certifications:
            formatted_text += "Certifications:\n"
            for cert in certifications:
                formatted_text += f"  - {cert}\n"

        # Languages
        languages = candidate.get("languages", [])
        if languages:
            formatted_text += "Languages:\n"
            for lang in languages:
                formatted_text += f"  - {lang}\n"

        # Preferred Roles
        preferred_roles = candidate.get("preferred_roles", [])
        if preferred_roles:
            formatted_text += "Preferred Roles:\n"
            for role in preferred_roles:
                formatted_text += f"  - {role}\n"
        
        formatted_text += "\n"

    return formatted_text

# Initialize Llama3 models using LangChain
llama3_categorizer = Ollama(model="llama3")
llama3_json_formatter = Ollama(model="llama3")

# Prompt template for categorizing CV content
categorization_prompt = PromptTemplate(
    input_variables=["cv_text"],
    template="""
    You are an AI that categorizes resume content into sections. Structure the given text correctly into sections:
    
    **Extracted CV Information:**
    ```
    {cv_text}
    ```
    
    **Return the structured CV content with sections:**
    - Name
    - About Me (Description)
    - Contact Info (Email, Phone, LinkedIn)
    - Skills:
        - Hard skills: Include all technical or job-specific skills (e.g., "Python", "Microsoft Word").
        - Soft skills: Include only the following predefined soft skills if they exist in the CV:
            Écoute active, Expression orale, Clarté dans l'écriture, Négociation, Persuasion, Empathie,
            Capacité à donner du feedback, Collaboration, Coopération, Esprit d'équipe, Gestion des conflits,
            Adaptabilité dans un groupe, Répartition des tâches, Respect des opinions des autres, Prise de décision,
            Motivation des autres, Gestion de crise, Vision stratégique, Coaching et mentorat, Responsabilité,
            Gestion du stress, Organisation, Priorisation des tâches, Respect des délais, Gestion de la charge de travail,
            Planification efficace, Délégation, Pensée analytique, Créativité, Esprit critique, Gestion de l’incertitude,
            Capacité à proposer des solutions innovantes, Flexibilité, Gestion du changement, Capacité à apprendre rapidement,
            Résilience, Ouverture d’esprit, Gestion des émotions, Conscience de soi, Contrôle de soi, Planification,
            Suivi des objectifs, Gestion des ressources, Évaluation des risques, Coordination d’équipe, Patience,
            Orientation client, Gestion des réclamations, Courtoisie, Capacité à résoudre les problèmes des clients,
            Curiosité, Autonomie, Esprit d’initiative, Auto-évaluation, Soif d’apprendre.
    - Experience (Job Title, Company, Duration)
    - Education (Degree, University, Year)
    - Certifications
    - Projects (Title, Duration)
    - Languages (Proficiency)
    - Preferred Roles
    - Availability
    
    **Output:**
    Return only structured text, NO extra formatting, NO JSON, just well-organized content.
    **IMPORTANT:** 
    - If a skill from the predefined soft skills list exists in the CV, categorize it as a soft skill.
    - All other skills should be categorized as hard skills.
    """
)

# Prompt template for converting structured text to JSON
json_prompt = PromptTemplate(
    input_variables=["structured_text", "candidate_id"],
    template="""
    Convert the following structured resume data into valid JSON format. 
    
    **Candidate ID:** {candidate_id}
    
    **Structured Resume Data:**
    ```
    {structured_text}
    ```
    
    **Expected JSON Format:**
    {{
        "id": "{candidate_id}",
        "name": "Extracted Name",
        "about me":"description",
        "contact": {{
            "email": "Extracted Email",
            "phone": "Extracted Phone",
            "linkedin": "Extracted LinkedIn"
        }},
        "skills": {{
            "hard_skills": {{
                "Skill1": "Level",
                "Skill2": "Level"
            }},
            "soft_skills": ["Skill1", "Skill2"]
        }},
        "experience": [
            {{
                "job_title": "Extracted Job Title",
                "company": "Extracted Company",
                "duration": "YYYY-MM-DD to YYYY-MM-DD"
            }}
        ],
        "education": "Degree from University (Year)",
        "certifications": ["Cert1", "Cert2"],
        "projects": [
            {{
                "title": "Project Title",
                "duration": "YYYY-MM-DD to YYYY-MM-DD"
            }}
        ],
        "languages": ["Language1 (Proficiency)", "Language2 (Proficiency)"],
        "preferred_roles": ["Role1", "Role2"],
        "availability": "Availability Status"
    }}
    
    **Return only the JSON, nothing else.**
    **IMPORTANT:** Do not include any extra formatting, such as triple backticks or "json" markers.
    """
)

# Initialize LangChain Chains
categorization_chain = LLMChain(llm=llama3_categorizer, prompt=categorization_prompt)
json_chain = LLMChain(llm=llama3_json_formatter, prompt=json_prompt)

# Process uploaded CVs
candidates = []
if uploaded_files:
    for idx, file in enumerate(uploaded_files, start=1):
        file_ext = file.name.split(".")[-1].lower()
        st.subheader(f"📄 Processing CV: {file.name}")

        # Extract text based on file type
        if file_ext == "pdf":
            extracted_text = extract_text_from_pdf(file)
        elif file_ext == "docx":
            extracted_text = extract_text_from_docx(file)
        elif file_ext in ["jpg", "jpeg", "png"]:
            image = Image.open(file)
            st.image(image, caption="📄 CV Preview", use_container_width=True)
            extracted_text = extract_text_from_image(image)
        else:
            extracted_text = "⚠️ Unsupported file format!"

        if extracted_text:
            # Step 1: Categorize content using the first Llama3 model
            structured_cv = categorization_chain.run(cv_text=extracted_text)
            
            # Debug: Print the structured CV content
            print("Structured CV Content:")
            print(structured_cv)

            # Step 2: Convert categorized content into JSON using the second Llama3 model
            candidate_id = f"CAND{idx:03}"
            formatted_json = json_chain.run(structured_text=structured_cv, candidate_id=candidate_id)

            try:
                # Clean the JSON response
                formatted_json = formatted_json.strip()
                formatted_json = formatted_json.replace("```json", "").replace("```", "")

                # Convert response into JSON object
                candidate_data = json.loads(formatted_json)
                candidates.append(candidate_data)
            except json.JSONDecodeError as e:
                st.error(f"⚠️ Error parsing JSON for {file.name}. Skipping this file.")
                # Debug: Print the invalid JSON and error message
                print(f"Error parsing JSON for {file.name}:")
                print(formatted_json)
                print(f"Error details: {e}")
                continue  # Skip to the next file

# Final structured JSON output
final_output = {"candidates": candidates}
if candidates:
    st.subheader("📌 Structured JSON Output")
    st.json(final_output)
    
    # Provide a download button for the JSON file
    json_data = json.dumps(final_output, indent=4, ensure_ascii=False)
    st.download_button(
        label="📥 Download JSON",
        data=json_data,
        file_name="structured_cvs.json",
        mime="application/json"
    )

    # Path to the JSON file
    json_file_path = "candidates.json"
    text_file_path = "candidates.txt"

    # Initialize an empty JSON structure if the file is empty or does not exist
    if not os.path.exists(json_file_path) or os.path.getsize(json_file_path) == 0:
        with open(json_file_path, "w", encoding="utf-8") as file:
            json.dump({"candidates": []}, file)  # Initialize with empty candidates list

    # Read the existing data
    # Read the existing data and ensure "candidates" is a list
    try:
        with open(json_file_path, "r", encoding="utf-8") as existing_file:
            existing_data = json.load(existing_file)
        
        # Ensure "candidates" is always a list
        if not isinstance(existing_data.get("candidates"), list):
            existing_data["candidates"] = []
    except (json.JSONDecodeError, FileNotFoundError):  # Handle invalid JSON or missing file
        st.error("⚠️ Invalid or missing JSON file. Initializing with empty data.")
        existing_data = {"candidates": []}  # Reset to an empty list


    # Merge the existing candidates with the new ones
    existing_data["candidates"].extend(final_output["candidates"])

    # Write the updated data back to the file
    with open(json_file_path, "w", encoding="utf-8") as outfile:
        json.dump(existing_data, outfile, indent=4, ensure_ascii=False)

    st.success("✅ CV Processing & JSON Formatting Complete! Data appended to candidates.json.")
    # Load the JSON file
    with open(json_file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    # Convert JSON to formatted text
    formatted_text = format_candidate_data(data.get("candidates", []))

    # Save as text file
    with open(text_file_path, "w", encoding="utf-8") as text_file:
        text_file.write(formatted_text)

    print(f"✅ Conversion completed! Check {text_file_path}")

    ############################################"
    # """
    langchain.debug = True

    # ✅ Set Hugging Face cache directory
    os.environ["HF_HOME"] = "C:/Users/MSI/.cache/huggingface"

    # ✅ Load the embedding model (LaJavaness with 1024 dimensions)
    EMBEDDING_MODEL = "Lajavaness/bilingual-embedding-large"
    model_kwargs = {"trust_remote_code": True}
    embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs=model_kwargs)

    # ✅ Ensure ChromaDB is reset
    chroma_db_path = "emb/"
    if os.path.exists(chroma_db_path):
        print("🛑 Deleting old ChromaDB to prevent conflicts...")
        shutil.rmtree(chroma_db_path)  # Delete existing database

    # ✅ Create directories if not exist
    os.makedirs("emb", exist_ok=True)

    # ✅ Load and split job offers text file
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0, separator="\n\n")


    # ✅ Load and split candidates text file
    loader_candidates = TextLoader("candidates.txt")  # Load candidates from text file
    docs_candidates = loader_candidates.load_and_split(text_splitter=text_splitter)

    # ✅ Initialize ChromaDB
    db_candidates = Chroma(persist_directory="emb/candidatesDB", embedding_function=embedding_function)

    # ✅ Add text data to ChromaDB
    db_candidates.add_texts(texts=[doc.page_content for doc in docs_candidates])

    print("✅ Jobs and candidates stored successfully in ChromaDB as text!")
    
else:
    st.warning("⚠️ No candidates processed or found.")

# st.title("App 1")

# # Button to go to App 2
# st.page_link("http://localhost:8502", label="Go to App 2")

import streamlit as st

st.title("App 1")

# Styled button to open App 2 in a new tab
st.markdown(
    """
    <style>
        .open-app-btn {
            display: inline-block;
            padding: 12px 24px;
            font-size: 18px;
            font-weight: bold;
            color: white;
            background-color: #007BFF;
            border: none;
            border-radius: 8px;
            text-decoration: none;
            cursor: pointer;
            transition: background-color 0.3s ease;
        }
        .open-app-btn:hover {
            background-color: #0056b3;
        }
    </style>
    <a href="http://localhost:8502" target="_blank" class="open-app-btn">Go to App 2 🚀</a>
    """,
    unsafe_allow_html=True
)
