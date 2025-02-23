import torch
import librosa
import numpy as np
import sounddevice as sd  # Pour capturer l'audio en temps réel
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor

# Charger le modèle et le feature extractor
model_name = "superb/wav2vec2-base-superb-er"  # Modèle de reconnaissance d'émotion
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)

# Définir les paramètres de capture audio
sample_rate = 16000  # Taux d'échantillonnage
duration = 5  # Durée de chaque segment audio en secondes

# Mapping des émotions du modèle à vos catégories
emotion_mapping = {
    "neutral": "secure",       # Neutral -> Secure
    "happy": "confident",      # Happy -> Confident
    "sad": "insecure",         # Sad -> Insecure
    "angry": "insecure",       # Angry -> Insecure
    "fear": "insecure",        # Fear -> Insecure
    "disgust": "insecure",     # Disgust -> Insecure
    "surprise": "confident"    # Surprise -> Confident
}

# Dictionnaire pour stocker les comptes d'émotions
emotion_counts = {
    "insecure": 0,
    "secure": 0,
    "confident": 0
}

# Fonction pour capturer l'audio en temps réel
def capture_audio():
    print("Enregistrement en cours...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()  # Attendre que l'enregistrement soit terminé
    audio = np.squeeze(audio)  # Supprimer la dimension supplémentaire
    return audio

# Fonction pour prédire l'émotion
def predict_emotion(audio):
    # Extraire les caractéristiques audio
    inputs = feature_extractor(audio, sampling_rate=sample_rate, return_tensors="pt")
    
    # Prédire l'émotion
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=-1).item()
    
    # Convertir la classe prédite en émotion
    emotion_labels = ["neutral", "happy", "sad", "angry", "fear", "disgust", "surprise"]
    predicted_emotion = emotion_labels[predicted_class]
    
    # Mapper l'émotion prédite à votre catégorie
    return emotion_mapping.get(predicted_emotion, "unknown")

# Fonction pour calculer les pourcentages d'émotion
def calculate_emotion_percentages(emotion_counts, total_predictions):
    percentages = {}
    for emotion, count in emotion_counts.items():
        percentages[emotion] = (count / total_predictions) * 100
    return percentages

# Boucle principale pour la détection en temps réel
try:
    total_predictions = 0  # Compteur total de prédictions
    while True:
        # Capturer l'audio
        audio = capture_audio()
        
        # Prédire l'émotion
        emotion = predict_emotion(audio)
        print(f"Émotion détectée : {emotion}")
        
        # Mettre à jour le compteur d'émotions
        if emotion in emotion_counts:
            emotion_counts[emotion] += 1
            total_predictions += 1

except KeyboardInterrupt:
    print("\nArrêt du programme.")
    
    # Calculer les pourcentages d'émotion
    emotion_percentages = calculate_emotion_percentages(emotion_counts, total_predictions)
    
    # Afficher les pourcentages
    print("\nPourcentages d'émotion détectées :")
    for emotion, percentage in emotion_percentages.items():
        print(f"{emotion}: {percentage:.2f}%")
#------------------try the code of the top and the code on the bottom------------------

# import torch
# import librosa
# import numpy as np
# import sounddevice as sd
# from collections import deque
# from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor

# # Charger un modèle open-source ne nécessitant pas de token
# model_name = "superb/wav2vec2-base-superb-er"
# feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
# model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)

# # Définir les paramètres de capture audio
# sample_rate = 16000
# duration = 5

# # Mapping des émotions vers les classes "stress" et "confident"
# emotion_mapping = {
#     "neutral": "stress",
#     "happy": "confident",
#     "sad": "stress",
#     "angry": "stress",
#     "fear": "stress",
#     "disgust": "stress",
#     "surprise": "confident"
# }

# # Dictionnaire pour stocker les comptes d'émotions
# emotion_counts = {"stress": 0, "confident": 0}

# # Historique des dernières prédictions pour stabiliser les résultats
# recent_predictions = deque(maxlen=5)

# def capture_audio():
#     print("Enregistrement en cours...")
#     audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
#     sd.wait()
#     return np.squeeze(audio)

# def predict_emotion(audio):
#     inputs = feature_extractor(audio, sampling_rate=sample_rate, return_tensors="pt")
#     with torch.no_grad():
#         outputs = model(**inputs)
#         probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
#         predicted_class = torch.argmax(probs, dim=-1).item()
    
#     emotion_labels = ["neutral", "happy", "sad", "angry", "fear", "disgust", "surprise"]
#     predicted_emotion = emotion_labels[predicted_class]
#     return emotion_mapping.get(predicted_emotion, "unknown")

# def calculate_emotion_percentages(emotion_counts, total_predictions):
#     return {emotion: (count / total_predictions) * 100 for emotion, count in emotion_counts.items()}

# try:
#     total_predictions = 0
#     while True:
#         audio = capture_audio()
#         emotion = predict_emotion(audio)
#         recent_predictions.append(emotion)

#         # Appliquer un filtre basé sur les 5 dernières prédictions
#         if recent_predictions.count("stress") > recent_predictions.count("confident"):
#             final_emotion = "stress"
#         else:
#             final_emotion = "confident"
        
#         print(f"Émotion détectée : {final_emotion}")
        
#         if final_emotion in emotion_counts:
#             emotion_counts[final_emotion] += 1
#             total_predictions += 1
# except KeyboardInterrupt:
#     print("\nArrêt du programme.")
#     emotion_percentages = calculate_emotion_percentages(emotion_counts, total_predictions)
#     print("\nPourcentages d'émotion détectées :")
#     for emotion, percentage in emotion_percentages.items():
#         print(f"{emotion}: {percentage:.2f}%")




