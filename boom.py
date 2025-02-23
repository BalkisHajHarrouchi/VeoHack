import os
import time
import re
import PyPDF2
import streamlit as st
from langchain.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from fuzzywuzzy import fuzz
import speech_recognition as sr
import pyttsx3
from transformers import pipeline
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from facenet_pytorch import InceptionResnetV1, MTCNN, extract_face
from deepface import DeepFace
from collections import Counter
import threading
from dotenv import load_dotenv

load_dotenv()
st.title("Welcome to the Streamlit Interview App")
st.write("This is a test page launched from FastAPI.")
# Load emotion recognition model
emotion_model = pipeline("sentiment-analysis", model="j-hartmann/emotion-english-distilroberta-base")

groq_api_key = os.getenv("GROQ_API_KEY")
llm_groq = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.3-70b-versatile",
    temperature=0.2
)

def call_llm_with_retry(prompt, retries=5):
    delay = 2
    for attempt in range(retries):
        try:
            response = llm_groq.invoke(prompt)
            return response.content.strip()
        except Exception:
            if attempt < retries - 1:
                time.sleep(delay)
                delay *= 2
            else:
                return "Error: Groq API unavailable."

EMBEDDING_MODEL = 'Lajavaness/bilingual-embedding-large'
embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={"trust_remote_code": True})

def calculate_score(user_answer, ideal_answer):
    answer_embedding = embedding_function.embed_query(user_answer)
    ideal_embedding = embedding_function.embed_query(ideal_answer)
    
    cosine_similarity = sum(a * b for a, b in zip(answer_embedding, ideal_embedding)) / (
        (sum(a ** 2 for a in answer_embedding) ** 0.5) * (sum(b ** 2 for b in ideal_embedding) ** 0.5)
    )
    
    score = max(0, min(5, round(cosine_similarity * 5, 2)))  # Scale similarity score to 0-5 range
    return score

def generate_question(skill_type, extracted_skills):
    question_prompt = f"Generate one really short and clear {skill_type} interview question and its ideal answer based on these skills: {', '.join(extracted_skills)}. Only return a question and its answer. Keep it really short."
    return call_llm_with_retry(question_prompt)

def analyze_emotion(text):
    result = emotion_model(text)
    return result[0]['label']

def speak(text, emotion="neutral"):
    engine = pyttsx3.init()
    voices = engine.getProperty("voices")
    
    if emotion == "joy":
        engine.setProperty("rate", 180)
        engine.setProperty("volume", 1.0)
    elif emotion == "sadness":
        engine.setProperty("rate", 120)
        engine.setProperty("volume", 0.5)
    elif emotion == "anger":
        engine.setProperty("rate", 200)
        engine.setProperty("volume", 1.0)
    else:
        engine.setProperty("rate", 150)
        engine.setProperty("volume", 0.8)
    
    for voice in voices:
        if "en" in voice.languages or "english" in voice.name.lower():
            engine.setProperty("voice", voice.id)
            break
    
    engine.say(text)
    engine.runAndWait()

def listen():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Listening...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        collected_words = []
        silence_threshold = 3  # Time in seconds to detect silence
        last_speech_time = time.time()
        
        while True:
            try:
                audio = recognizer.listen(source, timeout=10, phrase_time_limit=5)
                word = recognizer.recognize_google(audio)
                collected_words.append(word)
                last_speech_time = time.time()
            except sr.UnknownValueError:
                pass
            except sr.RequestError:
                return "Could not request results."
            except sr.WaitTimeoutError:
                if time.time() - last_speech_time > silence_threshold:
                    break
        
        return " ".join(collected_words)

# Load YOLOv3 model and class labels
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
classes = []

with open('coco.names', 'r') as f:
    classes = f.read().splitlines()

# Load gesture detection model
gesture_model = load_model('GestureModellili.keras')

# Load FaceNet model for face recognition
facenet_model = InceptionResnetV1(pretrained='vggface2').eval()
mtcnn = MTCNN()

# Define recruitment-related emotional states
recruitment_emotions = {
    "stressed": ["fear", "angry", "disgust"],
    "confident": ["happy"],
    "relieved": ["neutral"],
    "enthusiastic": ["surprise"]
}

# Map detected emotions to recruitment states
def map_emotion(deepface_emotion):
    for category, emotions in recruitment_emotions.items():
        if deepface_emotion in emotions:
            return category
    return "neutral"

def extract_face_embeddings(face_image):
    preprocessed_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    boxes, _ = mtcnn.detect(preprocessed_image)
    if boxes is not None:
        aligned = extract_face(preprocessed_image, boxes[0])
        aligned = aligned.unsqueeze(0)
        embeddings = facenet_model(aligned)
        return embeddings.detach().numpy()
    return None

# Load reference image for face comparison
reference_image = cv2.imread("bal.png")
reference_embeddings = extract_face_embeddings(reference_image)

def count_persons_and_phones(image, confidence_threshold=0.5, nms_threshold=0.3):
    height, width = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)
    
    boxes_persons, confidences_persons = [], []
    boxes_phones, confidences_phones = [], []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confidence_threshold:
                center_x, center_y = int(detection[0] * width), int(detection[1] * height)
                w, h = int(detection[2] * width), int(detection[3] * height)
                x, y = int(center_x - w / 2), int(center_y - h / 2)
                
                if class_id == 0:  # 'person'
                    boxes_persons.append([x, y, w, h])
                    confidences_persons.append(float(confidence))
                elif class_id == 67:  # 'cell phone'
                    boxes_phones.append([x, y, w, h])
                    confidences_phones.append(float(confidence))

    num_persons = len(cv2.dnn.NMSBoxes(boxes_persons, confidences_persons, confidence_threshold, nms_threshold))
    num_phones = len(cv2.dnn.NMSBoxes(boxes_phones, confidences_phones, confidence_threshold, nms_threshold))

    return num_persons, num_phones

def compare_embeddings(embeddings1, embeddings2, threshold=1.0):
    distance = np.linalg.norm(embeddings1 - embeddings2)
    return distance < threshold

def video_analysis():
    cap = cv2.VideoCapture(0)
    start_time = time.time()
    ema_score = 0
    alpha = 0.2  # Smoothing factor

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        total_frames = st.session_state.get("total_frames", 0)
        total_frames += 1
        st.session_state["total_frames"] = total_frames

        # Count persons and phones
        num_persons, num_phones = count_persons_and_phones(frame)
        cv2.putText(frame, f"Persons: {num_persons}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(frame, f"Phones: {num_phones}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        # Detect fraud possibility
        if num_persons > 1 or num_phones > 0:
            fraud_count = st.session_state.get("fraud_count", 0)
            fraud_count += 1
            st.session_state["fraud_count"] = fraud_count

        # Perform face recognition
        frame_embeddings = extract_face_embeddings(frame)
        if frame_embeddings is not None:
            is_same_person = compare_embeddings(frame_embeddings, reference_embeddings)
            text = "Same person" if is_same_person else "Different person"
            cv2.putText(frame, text, (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if is_same_person else (0, 0, 255), 2)

            if is_same_person and num_persons == 1:
                processed_frame = preprocess_input(cv2.resize(frame, (224, 224)))
                prediction = gesture_model.predict(np.expand_dims(processed_frame, axis=0))
                confidence_score = (1 - prediction[0][0]) * 10

                ema_score = alpha * confidence_score + (1 - alpha) * ema_score

                if ema_score < 4:
                    color = (0, 0, 255)
                elif 4 <= ema_score < 5:
                    color = (0, 165, 255)
                else:
                    color = (0, 255, 0)

                cv2.putText(frame, f"Gesture Confidence: {ema_score:.2f}/10", (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Perform emotion detection
        try:
            analysis = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False)
            deepface_emotion = analysis[0]["dominant_emotion"]
            mapped_emotion = map_emotion(deepface_emotion)
            emotions_detected = st.session_state.get("emotions_detected", [])
            emotions_detected.append(mapped_emotion)
            st.session_state["emotions_detected"] = emotions_detected
            cv2.putText(frame, f"Emotion: {mapped_emotion}", (20, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        except Exception as e:
            print(f"Error during emotion analysis: {e}")

        cv2.imshow('Integrated Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    st.title("🔍 AI-Powered Job Skill Matcher and Video Analysis")

    # Initialize session state variables
    if "total_frames" not in st.session_state:
        st.session_state["total_frames"] = 1
    if "fraud_count" not in st.session_state:
        st.session_state["fraud_count"] = 0
    if "emotions_detected" not in st.session_state:
        st.session_state["emotions_detected"] = []

    # Start video analysis in a separate thread
    if "video_thread" not in st.session_state:
        st.session_state["video_thread"] = threading.Thread(target=video_analysis)
        st.session_state["video_thread"].start()

    # Job Skill Matcher Section
    st.sidebar.title("Job Skill Matcher")
    job_roles = [
        "Software Engineer", "Data Scientist", "Front-End Developer",
        "Back-End Developer", "Full-Stack Developer", "Machine Learning Engineer",
        "DevOps Engineer", "Cybersecurity Analyst", "Database Administrator", "Cloud Engineer"
    ]
    selected_role = st.sidebar.radio("Choose a role:", job_roles)

    uploaded_file = st.sidebar.file_uploader("Upload a CV (PDF)", type=["pdf"])
    if uploaded_file:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        pdf_text = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
        extract_prompt = f"Extract all relevant skills from this CV:\n\n{pdf_text}\n\nOnly return a comma-separated list.If you find a core skill like deep learning or machine learning you can add problem solving , if you find projects you can add teamwork if you find certificates you can find self development ."
        extracted_skills_raw = call_llm_with_retry(extract_prompt)
        extracted_skills = {skill.strip() for skill in extracted_skills_raw.split(",") if skill.strip()}
        
        st.sidebar.subheader("📌 Extracted Skills")
        st.sidebar.write(", ".join(extracted_skills) if extracted_skills else "No skills extracted.")
        
        retrieval_prompt = f"What are the required skills for a {selected_role}? Only return a comma-separated list."
        required_skills_raw = call_llm_with_retry(retrieval_prompt)
        required_skills = {skill.strip() for skill in required_skills_raw.split(",") if skill.strip()}
        
        st.sidebar.subheader(f"📋 Required Skills for {selected_role}")
        st.sidebar.write(", ".join(required_skills) if required_skills else "No required skills found.")
        
        technical_question = generate_question("technical", extracted_skills)
        soft_skill_question = generate_question("soft skills", extracted_skills)
        questions = [technical_question, soft_skill_question]
        scores = []
        
        for idx, qa in enumerate(questions, 1):
            if "\n" in qa:
                question, ideal_answer = qa.split("\n", 1)
            else:
                question, ideal_answer = qa, "No ideal answer provided."
            
            st.sidebar.subheader(f"📝 Question {idx}")
            st.sidebar.write(question)
            speak(question)
            
            user_answer = listen()
            user_emotion = analyze_emotion(user_answer) if user_answer else "neutral"
            
            st.sidebar.write(f"User answer : {user_answer}")
            st.sidebar.write(f"Detected emotion: {user_emotion}")
            #speak(f"I detected that you are feeling {user_emotion}.", user_emotion)
            
            score = calculate_score(user_answer, ideal_answer)
            scores.append(score)
            
            st.sidebar.write(f"📊 Score for this question: {score}/5")
            speak(f"Your answer is rated {score} out of 5.")
        
        total_score = round(sum(scores), 2)
        st.sidebar.subheader("🏆 Final Score")
        if total_score < 4:
            st.sidebar.write("Unfortunately, you could not pass the interview")
            speak("Unfortunately, you could not pass the interview")
        else:
            st.sidebar.write("You have passed the interview!")
            speak("You have passed the interview!")
        st.sidebar.write(f"Total Score: {total_score}/10")
        speak(f"Your final score is {total_score} out of 10.")

    # Video Analysis Section
    st.title("Video Analysis")
    st.write("Real-time video analysis is running in the background.")

    # Summarize results
    emotion_counts = Counter(st.session_state.get("emotions_detected", []))
    total_emotions = sum(emotion_counts.values())
    emotion_percentages = {k: (v / total_emotions) * 100 for k, v in emotion_counts.items()} if total_emotions else {}

    fraud_percentage = (st.session_state.get("fraud_count", 0) / st.session_state.get("total_frames", 1)) * 100

    
if __name__ == "__main__":
    main() 