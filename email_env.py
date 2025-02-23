import gym
import numpy as np
import textstat
from gym import spaces
from transformers import pipeline

class EmailEnhancementEnv(gym.Env):
    def __init__(self):
        super(EmailEnhancementEnv, self).__init__()
        self.action_space = spaces.Discrete(3)  # 3 actions: grammar correction, rephrasing, summarization
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(768,), dtype=np.float32)
        self.current_email = "hELLO sir I hope you well. i am ask about meeting next week."
        self.nlp_model = pipeline("feature-extraction", model="bert-base-uncased")

    def reset(self):
        """Resets the environment with a new email"""
        self.current_email = "hELLO sir I hope you well. i am ask about meeting next week."
        return np.array(self._get_text_embedding(), dtype=np.float32)

    def _get_text_embedding(self):
        """Extracts BERT-based embeddings of the email"""
        embedding = self.nlp_model(self.current_email)
        return np.mean(embedding[0], axis=0)

    def step(self, action):
        """Performs an enhancement action on the email"""
        if action == 0:
            self.current_email = self._correct_grammar()
        elif action == 1:
            self.current_email = self._rephrase_text()
        elif action == 2:
            self.current_email = self._summarize_text()
        reward = self._calculate_reward()
        done = reward > 0.9
        return np.array(self._get_text_embedding(), dtype=np.float32), reward, done, {}

    def _correct_grammar(self):
        correction_pipeline = pipeline("text2text-generation", model="t5-small")
        return correction_pipeline(f"correct: {self.current_email}")[0]['generated_text']

    def _rephrase_text(self):
        paraphrase_pipeline = pipeline("text2text-generation", model="t5-small")
        return paraphrase_pipeline(f"paraphrase: {self.current_email}")[0]['generated_text']

    def _summarize_text(self):
        summarizer = pipeline("summarization")
        return summarizer(self.current_email, max_length=20, min_length=10, do_sample=False)[0]['summary_text']

    def _calculate_reward(self):
        readability_score = textstat.flesch_reading_ease(self.current_email) / 100
        return max(0, min(1, readability_score))
