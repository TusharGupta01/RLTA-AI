import gym
import numpy as np
import pydash as ps

class OpenAIEnv():
	def __init__(self, envname, seed):
		self.env = gym.make(envname)
		
		pass