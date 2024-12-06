# Define DQN Model Architecture
import numpy as np
import keras
import random
from collections import deque
from matplotlib import pyplot as plt
import tqdm
import time

import gymnasium as gym

import os
# os.environ["TF_USE_LEGACY_KERAS"] = '1'

class DQN(keras.Model):
    def __init__(self, state_size, action_size, hidden_size=128, lr=0.0001):
        super().__init__()
        # define model layers in keras
        model = keras.models.Sequential()
        model.add(keras.Input(shape=(state_size,)))
        model.add(keras.layers.Dense(hidden_size, activation='relu'))
        model.add(keras.layers.Dense(hidden_size, activation='relu'))
        model.add(keras.layers.Dense(action_size, activation='linear'))

        # compile model in keras
        model.compile(loss='mse', optimizer=keras.optimizers.legacy.Adam(learning_rate=lr))
        # save model to DQN instance
        self.model = model


class Agent:
    # def __init__(self, window_size, num_features, test_mode=False, model_name=''):
    def __init__(self, observation_shape, action_size, window_size, test_mode=False, model_name=''):
        self.window_size = window_size # How many days of historical data do we want to include in our state representation?
        self.num_features = observation_shape[0] # How many training features do we have? 
        self.state_size = self.window_size*self.num_features # State size includes number of training features per day, and number of lookback days 
        self.action_size = action_size # 0=hold, 1=buy, 2=sell
        self.memory = deque(maxlen=5000) # Bound memory size: once the memory reaches 1000 units, the lefthand values are discarded as righthand values are added
        self.inventory = [] # Inventory to hold trades
        self.model_name = model_name # filename for saved model checkpoint loading
        self.test_mode = test_mode # flag for testing (allows model load from checkpoint model_name)

        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        if test_mode:
            self.model = keras.models.load_model(model_name)
        else:
            self.model = DQN(self.state_size, self.action_size).model
        self.target = keras.models.clone_model(self.model)
        self.target.set_weights(self.model.get_weights())
        # self.model = keras.models.load_model(model_name) if test_mode else self._model()

    #Deep Q Learning (DQL) model
    # def _model(self):
    #     model = DQN(self.state_size, self.action_size).model
    #     target = keras.models.clone_model(model)
    #     target.set_weights(model.get_weights())
    #     return model, target
    
    # DQL Predict (with input reshaping)
    #   Input = State
    #   Output = Q-Table of action Q-Values
    def get_q_values_for_state(self, state):
        return self.model.predict(state.flatten().reshape(1, self.state_size), verbose=0)
    
    def get_q_target(self, state):
        return self.target.predict(state.flatten().reshape(1, self.state_size), verbose=0)
    
    # DQL Fit (with input reshaping)
    #   Input = State, Target Q-Table 
    #   Output = MSE Loss between Target Q-Table and Actual Q-Table for State
    def fit_model(self, input_state, target_output):
        return self.model.fit(input_state.flatten().reshape(1, self.state_size), target_output, epochs=1, verbose=0)    
    
    # Agent Action Selector
    #   Input = State
    #   Policy = epsilon-greedy (to minimize possibility of overfitting)
    #   Intitially high epsilon = more random, epsilon decay = less random later
    #   Output = Action (0, 1, or 2)
    def act(self, state): 
        # Choose any action at random (Probablility = epsilon for training mode, 0% for testing mode)
        if not self.test_mode and random.random() <= self.epsilon:
            # **select random action here**
            return random.randrange(self.action_size)
        # Choose the action which has the highest Q-value (Probablitly = 1-epsilon for training mode, 100% for testing mode)
        # **use model to select action here - i.e. use model to assign q-values to all actions in action space (buy, sell, hold)**
        options = self.get_q_values_for_state(state)
        # **return the action that has the highest value from the q-value function.**
        return np.argmax(options[0])
    
    def update_target(self):
        self.target.set_weights(self.model.get_weights())

    # Experience Replay (Learning Function)
    #   Input = Batch of (state, action, next_state) tuples
    #   Optimal Q Selection Policy = Bellman equation
    #   Important Notes = Model fitting step is in this function (fit_model)
    #                     Epsilon decay step is in this function
    #   Output = Model loss from fitting step
    def exp_replay(self, batch_size):
        losses = []
        # define a mini-batch which holds batch_size most recent previous memory steps (i.e. states)
        # mini_batch = []
        # l = len(self.memory)
        # for i in range(l - batch_size + 1, l):
        #     mini_batch.append(self.memory[i])
        mini_batch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in mini_batch:
            # reminders: 
            #   - state is a vector containing close & MA values for the current time step
            #   - action is an integer representing the action taken by the act function at the current time step- buy, hold, or sell
            #   - reward represents the profit of a given action - it is either 0 (for buy, hold, and sells which loose money) or the profit in dollars (for a profitable sell)
            #   - next_state is a vector containing close & MA values for the next time step
            #   - done is a boolean flag representing whether or not we are in the last iteration of a training episode (i.e. True when next_state does not exist.)
            if done:
                # special condition for last training epoch in batch (no next_state)
                optimal_q_for_action = reward  
            else:
                next_q_values = self.get_q_target(next_state)
                # target Q-value is updated using the Bellman equation: reward + gamma * max(predicted Q-value of next state)
                optimal_q_for_action = reward + self.gamma * np.max(next_q_values)#  reward + gamma * max(predicted Q-value of next state)
            # Get the predicted Q-values of the current state
            target_q_table = self.get_q_values_for_state(state)
            # Update the output Q table - replace the predicted Q value for action with the target Q value for action 
            target_q_table[0][action] = optimal_q_for_action
            # Fit the model where state is X and target_q_table is Y
            history = self.fit_model(state, target_q_table)
            losses += history.history['loss']
           
        # define epsilon decay (for the act function)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return losses
    

# Plot training loss
def plot_losses(losses, title):
    plt.plot(losses)
    plt.title(title)
    plt.ylabel('MSE Loss Value')
    plt.xlabel('batch')
    plt.show()
    

if __name__ == "__main__":
    env = gym.make("CartPole-v1", render_mode="rgb_array")

    print(env.action_space.n)
    print(env.observation_space.shape[0])
    window_size = 1
    agent = Agent(env.observation_space.shape, env.action_space.n, window_size)
    
    
    batch_size = 128
    batch_losses = [0]*100
    num_batches_trained = 0
    episode_count = 10
    rewards = []
    total_time_start = time.time()
    
    # time_steps = train_df.shape[0]
    

    for e in range(episode_count + 1):
        done = False
        ep_time_start = time.time()
        time_step = 0
        obs, info = env.reset()

        # for t in tqdm.tqdm(range(time_steps), desc=f'Running episode {e}/{episode_count}'):
        while not done:
            time_step += 1
            action = agent.act(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            rewards.append(reward)
            done = terminated or truncated
            agent.memory.append((obs, action, reward, next_obs, done))
            obs = next_obs

            if len(agent.memory) > batch_size:
                # when the size of the memory is greater than the batch size, run the exp_replay function on the batch to fit the model and get losses for the batch
                losses = agent.exp_replay(batch_size)    
                # then sum the losses for the batch and append them to the batch_losses list
                batch_losses.append(sum(losses))
            # if (time_step) % 10 == 0:
            #     print(f"Step: {time_step}, Mean reward: {np.sum(rewards)}")

            num_batches_trained = len(batch_losses)

        if e % 2 == 0:
            agent.update_target()

        ep_time_end = time.time()
        print(f"Ep{e} -  Reward: {np.sum(rewards)} - in: {ep_time_end - ep_time_start:.4f} sec")
        rewards = []
        # print(f'Max Loss: {max(batch_losses[num_batches_trained:len(batch_losses)])}')
        # print(f'Total Loss: {sum(batch_losses[num_batches_trained:len(batch_losses)])}')
        # # print('--------------------------------')
        # plot_losses(batch_losses[num_batches_trained:len(batch_losses)], f'Episode {e} DQN model loss')
    total_time_end = time.time()
    print(f"Total time: {total_time_end - total_time_start:.2f}")
        