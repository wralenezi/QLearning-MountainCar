import numpy as np


# Hyper-parameters
LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 25000
# the number of bins for the state space; the more it is the larger the Q table is
STATE_SLOTS = 20


# The main trainer for QLearning Algorithm for a an agent with continuous state
# obs_dimension: the dimension of observational state  
# observations_low: array of the lowest values of the state space
# observations_highest: array of the highest values of the state space
# action_space: the number of actions 
class QTableModel:
	def __init__(self,obs_dimension,observations_low,observations_high,action_num):
		self.observations_low = observations_low
		# Observation space 
		self.observation_space_size =  [STATE_SLOTS] * obs_dimension
		# Size of bin for co continuous observation, should be the dimension of the state
		self.observation_bin_size = (observations_high - observations_low)/self.observation_space_size
		# Initialize Q table to store the action-state with the Q values, the highest is the goal reward		
		self.q_table  = np.random.uniform(low=-2, high=0, size = (self.observation_space_size + [action_num]))		

	# discretize a continuous state to the defined bins
	def get_discrete_state(self,state):
		discrete_state = (state - self.observations_low) / self.observation_bin_size
		# return the state in a tuple as integer
		return tuple(discrete_state.astype(np.int))

	def get_qvalues_for_state(self,state):
		return self.q_table[state]

	def get_new_qvalue(self,current_q,max_future_q,reward):
		new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)		
		return new_q

	def set_qvalue_on_qtable(self,discrete_state,action,new_q):
		self.q_table[discrete_state + (action, )] = new_q

