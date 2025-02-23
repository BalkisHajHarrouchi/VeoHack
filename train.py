import random
import torch
import torch.optim as optim
import numpy as np
from collections import deque
from email_env import EmailEnhancementEnv  # Import environment
from dqn_model import DQN  # Import DQN model
import os

# Hyperparameters
STATE_SIZE = 768
ACTION_SIZE = 3
BATCH_SIZE = 32
GAMMA = 0.95
LR = 0.001
EPSILON = 1.0  # Exploration rate
EPSILON_DECAY = 0.995  # Decay factor
EPSILON_MIN = 0.1  # Minimum epsilon for exploration

MODEL_PATH = "dqn_email_enhancer.pth"

# Initialize components
env = EmailEnhancementEnv()
model = DQN(STATE_SIZE, ACTION_SIZE)
optimizer = optim.Adam(model.parameters(), lr=LR)
memory = deque(maxlen=2000)

# **Load existing model if available**
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    print(f"✅ Loaded existing model: {MODEL_PATH}")

def train_dqn(episodes=70):
    global EPSILON
    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            if random.random() < EPSILON:
                action = env.action_space.sample()  # Explore
            else:
                with torch.no_grad():
                    q_values = model(torch.tensor(state, dtype=torch.float32))
                    action = torch.argmax(q_values).item()  # Exploit
            
            next_state, reward, done, _ = env.step(action)
            memory.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward

            # Train with batch updates
            if len(memory) > BATCH_SIZE:
                minibatch = random.sample(memory, BATCH_SIZE)
                for s, a, r, s_, d in minibatch:
                    target = r + (GAMMA * torch.max(model(torch.tensor(s_, dtype=torch.float32)))) if not d else r
                    q_values = model(torch.tensor(s, dtype=torch.float32))
                    loss = (target - q_values[a]) ** 2
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

        EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)  # Decay epsilon
        print(f"📌 Episode {episode+1}, Reward: {total_reward}, EPSILON: {EPSILON:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"✅ Model saved: {MODEL_PATH}")

train_dqn()
