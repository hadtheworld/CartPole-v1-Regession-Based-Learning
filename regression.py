import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense

env = gym.make('CartPole-v1')
epsilon= 1.0
epsilon_decay=0.995
epsilon_min = 0.01

def play_action(model,state):
    if np.random.rand()<=epsilon:
        return random.randrange(number_of_actions)
    else:
        action_values= model.predict(state)
        return np.argmax(action_values[0])



# creating the custom fit function for the regression model

gamma = 0.95
def fit(model,memory,epsilon):
    batch=random.sample(memory, batch_size)
    for state, action, reward, next_state, done in batch:
        target = reward
        if not done:
            target = reward + gamma*np.amax(model.predict(next_state)[0])
        state_target= model.predict(state)
        state_target[0][action]=target
        model.fit(state,state_target,epochs=1, verbose=0)
    if epsilon>epsilon_min:
        epsilon*=epsilon_decay
    return model,epsilon


    
state_size=env.observation_space.shape[0]
number_of_actions = env.action_space.n
model = Sequential()
model.add(Dense(24,input_dim=state_size,activation='relu'))
model.add(Dense(24,activation='relu'))
model.add(Dense(number_of_actions,activation='linear'))

model.compile(loss= 'mse', optimizer='adam')

memory = deque(maxlen=2000)

number_of_episodes=1000
batch_size=32

for e in range(number_of_episodes):
    state=env.reset()
    state = np.reshape(state, [1, state_size])

    for time_step in range(500):
        env.render()
        action=play_action(model,state)
        next_state, reward, done,_ = env.step(action)
        
        if done:
            reward-=15
        else:
            reward = reward
        next_state= np.reshape(next_state,[1,state_size])
        memory.append((state,action,reward, next_state,done))
        state= next_state

        if done:
            print("Number of points: ", time_step)
            break
        if len(memory)>batch_size:
            model,epsilon= fit(model,memory,epsilon)