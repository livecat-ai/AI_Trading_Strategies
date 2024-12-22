import numpy as np
import tensorflow as tf
from collections import deque
import gymnasium as gym


class DQN:

    def __init__(self, input_shape, n_outputs, n_hidden=64, buf_size=100000, batch_size=64):
        self.model = tf.keras.Sequential([
                tf.keras.layers.Dense(n_hidden, activation='elu', input_shape=input_shape),
                tf.keras.layers.Dense(n_hidden, activation='elu'),
                tf.keras.layers.Dense(n_outputs)
            ])
        self.target = tf.keras.models.clone_model(self.model)
        self.target.set_weights(self.model.get_weights())
        self.replay_buffer = deque(maxlen=buf_size)
        self.batch_size = batch_size
        self.discount_factor = 0.95
        # self.optimizer = tf.keras.optimizers.Nadam(learning_rate=1e-2)
        self.optimizer = tf.keras.optimizers.legacy.Nadam(learning_rate=1e-2)
        self.loss_fn = tf.keras.losses.mean_squared_error
        self.n_outputs = n_outputs
        self.input_shape = input_shape

    def epsilon_greedy_policy(self, state, epsilon=0):
        if np.random.rand() < epsilon:
            return np.random.randint(self.n_outputs)
        else:
            Q_values = self.model.predict(state[np.newaxis], verbose=0)[0]
        return Q_values.argmax()
    
    def sample_experiences(self, batch_size):
        indices = np.random.randint(len(self.replay_buffer), size=batch_size)
        batch = [self.replay_buffer[index] for index in indices]
        return [
            np.array([experience[field_index] for experience in batch])
            for field_index in range(6)
        ] # [states, actions, rewards, next_states, dones, truncateds]
    
    def play_one_step(self, env, state, epsilon):
        action = self.epsilon_greedy_policy(state, epsilon)
        next_state, reward, done, truncated, info = env.step(action)
        self.replay_buffer.append((state, action, reward, next_state, done, truncated))
        return next_state, reward, done, truncated, info
    
    def training_step(self):
        experiences = self.sample_experiences(self.batch_size)
        states, actions, rewards, next_states, dones, truncateds = experiences
        next_Q_values = self.target.predict(next_states, verbose=0)
        max_next_Q_values = next_Q_values.max(axis=1)
        runs = 1.0 - (dones | truncateds)
        target_Q_values = rewards + runs * self.discount_factor * max_next_Q_values
        target_Q_values = target_Q_values.reshape(-1, 1)
        mask = tf.one_hot(actions, self.n_outputs)
        with tf.GradientTape() as tape:
            all_Q_values = self.model(states)
            Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
            loss = tf.reduce_mean(self.loss_fn(target_Q_values, Q_values))

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

    def update_target_weights(self):
        self.target.set_weights(self.model.get_weights())


if __name__ == "__main__":

    env = gym.make("CartPole-v1", render_mode="rgb_array")

    input_shape = env.observation_space.shape
    n_outputs = env.action_space.n
    dqn = DQN(input_shape, n_outputs, n_hidden=128, buf_size=100000, batch_size=128)

    rewards = []
    
    for episode in range(600):
        ep_rewards = []
        obs, info = env.reset()
        for step in range(600):
            epsilon = max(1 - episode / 500, 0.01)
            obs, reward, done, truncated, info = dqn.play_one_step(env, obs, epsilon)
            if done or truncated:
                break
            ep_rewards.append(reward)
        
        rewards.append(np.sum(ep_rewards))
        
        if episode > 50:
            dqn.training_step()

        if episode % 50 == 0:
            print(f"Ep: {episode} - Mean rewards: {np.mean(rewards[-50:])}")
            avg_rewards = []
            dqn.update_target_weights()