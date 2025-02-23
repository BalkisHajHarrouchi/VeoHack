# import ollama
# import torch
# import numpy as np
# from email_env import EmailEnhancementEnv
# from dqn_model import DQN

# # Load the trained DQN model
# STATE_SIZE = 768
# ACTION_SIZE = 3

# model = DQN(STATE_SIZE, ACTION_SIZE)
# model.load_state_dict(torch.load("dqn_email_enhancer.pth"))
# model.eval()  # Set to evaluation mode

# # Initialize Email Environment
# env = EmailEnhancementEnv()

# def generate_email(prompt):
#     """Generates an email using Ollama's local LLaMA 3 (8B)"""
#     response = ollama.chat(model="llama3", messages=[{"role": "user", "content": prompt}])
#     return response["message"]["content"].strip()

# def enhance_email(email_text):
#     """Uses the trained DQN model to enhance the email"""
#     state = env.reset()
#     env.current_email = email_text
#     state = np.array(env._get_text_embedding(), dtype=np.float32)

#     done = False
#     while not done:
#         with torch.no_grad():
#             q_values = model(torch.tensor(state, dtype=torch.float32))
#             action = torch.argmax(q_values).item()

#         next_state, reward, done, _ = env.step(action)
#         state = next_state

#     return env.current_email  # Return the enhanced email

# # Step 1: Generate a sample email using LLaMA 3 (8B) with Ollama
# prompt = "Write a formal email to request a job interview of a chosen person because their resume matches."
# generated_email = generate_email(prompt)
# print("\n🔹 **Generated Email:**\n", generated_email)

# # Step 2: Enhance the generated email using DQN
# enhanced_email = enhance_email(generated_email)
# print("\n✅ **Enhanced Email:**\n", enhanced_email)



# import streamlit as st
# import ollama
# import torch
# import numpy as np
# from email_env import EmailEnhancementEnv
# from dqn_model import DQN

# # Load the trained DQN model
# STATE_SIZE = 768
# ACTION_SIZE = 3

# model = DQN(STATE_SIZE, ACTION_SIZE)
# model.load_state_dict(torch.load("dqn_email_enhancer.pth", map_location=torch.device("cpu")))
# model.eval()  # Set to evaluation mode

# # Initialize Email Environment
# env = EmailEnhancementEnv()

# # Streamlit UI
# st.title("✉️ AI Email Generator & Enhancer")
# st.subheader("Generate and Improve Emails Using LLaMA 3 (8B) & DQN")

# prompt = st.text_area("📌 Enter the email prompt:", "Write a formal email to request a job interview.")

# if st.button("🚀 Generate & Enhance Email"):
#     with st.spinner("Generating email using LLaMA 3 (8B)..."):
#         def generate_email(prompt):
#             """Generates an email using Ollama's local LLaMA 3 (8B)"""
#             response = ollama.chat(model="llama3", messages=[{"role": "user", "content": prompt}])
#             return response["message"]["content"].strip()

#         generated_email = generate_email(prompt)
#         st.subheader("🔹 Generated Email")
#         st.write(generated_email)

#     with st.spinner("Enhancing email using DQN..."):
#         def enhance_email(email_text):
#             """Uses the trained DQN model to enhance the email"""
#             state = env.reset()
#             env.current_email = email_text
#             state = np.array(env._get_text_embedding(), dtype=np.float32)

#             done = False
#             while not done:
#                 with torch.no_grad():
#                     q_values = model(torch.tensor(state, dtype=torch.float32))
#                     action = torch.argmax(q_values).item()

#                 next_state, reward, done, _ = env.step(action)
#                 state = next_state

#             return env.current_email  # Return the enhanced email

#         enhanced_email = enhance_email(generated_email)
#         st.subheader("✅ Enhanced Email")
#         st.write(enhanced_email)

# st.info("🚀 Powered by LLaMA 3 (8B) & Reinforcement Learning (DQN)")



import streamlit as st
import ollama
import torch
import numpy as np
from email_env import EmailEnhancementEnv
from dqn_model import DQN

# ✅ Check if GPU is available and force PyTorch to use it
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using Device: {device}")

# ✅ Load the trained DQN model
STATE_SIZE = 768
ACTION_SIZE = 3

try:
    model = DQN(STATE_SIZE, ACTION_SIZE)
    model.load_state_dict(torch.load("dqn_email_enhancer.pth", map_location=device))
    model.to(device)  # Move model to GPU
    model.eval()  # Set to evaluation mode
    print("✅ DQN model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading DQN model: {e}")
    st.error("DQN model loading failed! Please check the model file.")

# ✅ Initialize Email Environment
env = EmailEnhancementEnv()

# ✅ Streamlit UI
st.title("✉️ AI Email Generator & Enhancer")
st.subheader("Generate and Improve Emails Using LLaMA 3 (8B) & DQN")

prompt = st.text_area("📌 Enter the email prompt:", "Write a formal email to request a job interview.")

if st.button("🚀 Generate & Enhance Email"):
    with st.spinner("⏳ Generating email using LLaMA 3 (8B)..."):
        def generate_email(prompt):
            """Generates an email using Ollama's local LLaMA 3 (8B)"""
            try:
                response = ollama.chat(model="llama3", messages=[{"role": "user", "content": prompt}])
                return response["message"]["content"].strip()
            except Exception as e:
                print(f"❌ Error in email generation: {e}")
                return "⚠️ Error generating email. Please check LLaMA 3 setup."

        generated_email = generate_email(prompt)
        st.subheader("🔹 Generated Email")
        st.write(generated_email)

    with st.spinner("⏳ Enhancing email using DQN..."):
        def enhance_email(email_text):
            """Uses the trained DQN model to enhance the email"""
            try:
                state = env.reset()
                env.current_email = email_text
                state = torch.tensor(env._get_text_embedding(), dtype=torch.float32).to(device)

                done = False
                while not done:
                    with torch.no_grad():
                        q_values = model(state)
                        action = torch.argmax(q_values).item()

                    next_state, reward, done, _ = env.step(action)
                    state = torch.tensor(next_state, dtype=torch.float32).to(device)

                return env.current_email  # Return the enhanced email
            except Exception as e:
                print(f"❌ Error in email enhancement: {e}")
                return "⚠️ Error enhancing email. Please check the DQN model."

        enhanced_email = enhance_email(generated_email)
        st.subheader("✅ Enhanced Email")
        st.write(enhanced_email)

st.info("🚀 Powered by LLaMA 3 (8B) & Reinforcement Learning (DQN)")
