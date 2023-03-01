# CartPole-v1-Regession-Based-Learning
In this project I have used neural networks to train a regression based model for "CartPole-v1" in gym Module by OpenAI, with a custom fit function, using e-Greedy approach, Bellman Equation(Q-Network) to find reward values at a particular state, Temporal Difference as loss functions and adam as the optimizer


Type of Machine Learning:
   - Regression based game playing

Model Used:
   - Neural Neteork created using Sequential() class of keras module
   - Hidden and Output Layers :- created using Dense() in keras.layers

Loss function and optimiser used for the neural network:
   - MSE:- mean squared error, here this would be similar to the Temporal Difference in Q-Network
   - 'adam':= this is the optimizer used. It is a variation of the Grdient Descend.

Policy used:
   - e-Greedy approach:- is used for selecting actions in a particular state. 'e' stands for 'epsilon' which decides till ewhat iteration will the random states will be explored and after it, the model will be used to predict. This gives a better training and extra information about the environment to the neural network, which makes the performance even better.
   
Custom fit function to train the neural network:
   - A custom fit function is coded to train the model to the evironment:
          - It returns random states in the begining based on the value of 'e' which is also reduces by a margin 'epsilon_decay' every time to reduce the randomness
          - the random states returned give the model idea about the parts of environment it would not have traversed till some time. This boosts the predisction and training result of the neural network
          - the reduction of 'e' stops after a limit 'epsilon_min' to still keep some amount of randomness. Here it is 0.01 i.e 1%.
 
 Replay_Memory: 
    - memory = deque(maxlen=2000):- This memty stores the states explored, their history and future steps to be taken. The random state is is chosen from this replay memory to train the reinforcement model(neural network).
 
 
 **Detailed Information of the code is given from this point: look it for a deeper understanding**
 
  **here's a step-by-step documentation of the code:**

Import necessary libraries:
   - **random**: for generating random numbers
   - **gym**: for creating the CartPole-v1 environment
   - **numpy**: for working with numerical arrays
   - **deque**: for implementing a circular buffer memory
   - **Sequential** and **Dense** from Keras, for building and training the neural network


Create the CartPole-v1 environment using the **gym.make()** method and set the initial values of the exploration rate (epsilon), its decay rate (**epsilon_decay**), and the minimum exploration rate (**epsilon_min**).


Define the **play_action()** function that takes in the neural network model and a state as inputs, and returns either a random action or the action with the highest predicted Q-value for the given state, based on the exploration rate.

Set the discount factor (**gamma**) for the rewards.

Define the **fit()** function that takes in the neural network model, the memory buffer, and the exploration rate as inputs, and returns the updated model and exploration rate. This function does the following:
  - Randomly samples a batch of transitions from the memory buffer.
  - Calculates the target Q-values for each transition, based on whether the transition resulted in a terminal state or not.
  - Updates the Q-values for the given state-action pair in the neural network.
  - Trains the neural network on the updated Q-values for the given state.
  - Decays the exploration rate, if it is greater than the minimum exploration rate.


Define the **state_size** and **number_of _actions** in the environment, and build the neural network model using the **Sequential()** and **Dense()** functions from Keras. The model has two hidden layers of 24 neurons each, and a linear output layer with a number of neurons equal to the number of actions in the environment.

Compile the neural network model using the mean squared error loss and the Adam optimizer.

Create a memory buffer with a maximum size of 2000 transitions.

Set the number of episodes to be run (**number_of_episodes**) and the batch size for training (**batch_size**).

Run a loop for each episode:
  - Reset the environment and get the initial state.
  - Reshape the state into a 2D array.
  - Run a loop for each time step within the episode:
      - Render the environment.
      - Choose an action using the play_action() function.
      - Take the chosen action and get the next state, reward, and terminal status.
      - Adjust the reward if the episode is terminal.
      - Reshape the next state into a 2D array.
      - Add the transition to the memory buffer.
      - Update the current state to be the next state.
      - If the episode is terminal, print the number of time steps and break out of the loop.
      - If the memory buffer is larger than the batch size, train the neural network using the **fit()** function.
      

**This code implements a Q-learning algorithm with deep Q-networks (DQN) to learn how to balance a pole on a cart in the CartPole-v1 environment.**
