import numpy as np


# Hyperparameters
LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 25000


# The main trainer for QLearning Algorithm for a an agent with continous state
# obs_dimension: the dimension of observational state  
# observations_low: array of the lowest values of the state space
# observations_highest: array of the highest values of the state space
# action_space: the number of actions 
class QTableModel:
	def __init__(self,obs_dimension,observations_low,observations_high,action_num):
		# Observation space
		self.observation_space_size =  [20] * obs_dimension
		# Size of bin for co ntinuous observation, should be the dimension of the state
		self.observation_bin_size = (observations_high - observations_low)/self.observation_space_size
		# Initialize Q table to store the action-state with the Q values, the highest is the goal reward
		self.q_table  = np.random.uniform(low=-2, high=0, size = (self.observation_space_size + action_num))

	# discretize a continous state to the defined bins
	def get_discrete_state(self,state,observations_low):
		discrete_state = (state - observations_low) / self.observation_bin_size
		return tuple(discrete_state.astype(np.int))
