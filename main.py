import gym
import numpy as np
import qtable_train as ql

SHOW_EVERY = 2000
EPISODES = 25000

# Set the environment we want to work with
env = gym.make("MountainCar-v0")


# Set the DQN for training 
dqn_model = ql.QTableModel(len(env.observation_space.high),env.observation_space.low,env.observation_space.high,env.action_space.n)

# Learning Loop
for episode in range(EPISODES):
	if episode % SHOW_EVERY == 0:
		render = True
	else:
		render = False
	# Reset the environment 
	discrete_state = dqn_model.get_discrete_state(env.reset())
	done = False

	# Loop through the environment
	while not done:
		# choose the action based on the highest q value
		action = np.argmax(dqn_model.get_qvalues_for_state(discrete_state))
		# make the action and receive the outcomes
		new_state, reward, done, _ = env.step(action)
		# get the new state and discretize it
		new_discrete_state = dqn_model.get_discrete_state(new_state)
		if render:
			# render the environment 
			env.render()
		# Get the 
		if not done:
			# get the max q value for the new discrete state
			max_future_q = np.max(dqn_model.get_qvalues_for_state(new_discrete_state))
			# Get the current q value for the action taken
			current_q = dqn_model.get_qvalues_for_state(discrete_state)[action]
			# Calculate the new q value for that instance
			new_q = dqn_model.get_new_qvalue(current_q,max_future_q,reward)
			# assign it on the q table
			dqn_model.set_qvalue_on_qtable(discrete_state,action,new_q)
		# Setting the reward or rather the max q value at the instance
		elif new_state[0] >= env.goal_position:
			print(f"Solved at {episode}")
			dqn_model.set_qvalue_on_qtable(discrete_state,action,0)		

		discrete_state = new_discrete_state

		env.close()