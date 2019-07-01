import gym
import qtable_train as ql

# Set the environment we want to work with
env = gym.make("MountainCar-v0")


# Set the DQN for training 
#dqn_model = ql.QTableModel(env.observation_space.high,env.observation_space.low,env.observation_space.high,env.action_space.n)

# Reset the environment 
env.reset()

done = False

# Loop through the environment
while not done:
	action = 2
	# make the action and recive the outcomes
	new_state, reward, done, _ = env.step(action)
	# render the enviroment 
	env.render()

env.close()