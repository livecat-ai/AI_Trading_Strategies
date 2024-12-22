
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from dqn import DQN
from trading_environment import (Trading_Environment,
                                 read_data,
                                 add_features,
                                 train_test_split,
                                 StandardScaler,
                                 format_price
                                )

# Plot behavior of trade output
def plot_behavior(data_input, bb_upper_data, bb_lower_data, states_buy, states_sell, profit, train=True):
    fig = plt.figure(figsize = (15,5))
    plt.plot(data_input, color='k', lw=2., label= 'Close Price')
    plt.plot(bb_upper_data, color='b', lw=2., label = 'Bollinger Bands')
    plt.plot(bb_lower_data, color='b', lw=2.)
    plt.plot(data_input, '^', markersize=10, color='r', label = 'Buying signal', markevery = states_buy)
    plt.plot(data_input, 'v', markersize=10, color='g', label = 'Selling signal', markevery = states_sell)
    plt.title('Total gains: %f'%(profit))
    plt.legend()
    if train:
        plt.xticks(range(0, len(train_df.index.values), int(len(train_df.index.values)/15)), train_df.index.values[0:: int(len(train_df.index.values)/15)], rotation=45, fontsize='small')
    else:
        plt.xticks(range(0, len(test_df.index.values), int(len(test_df.index.values)/15)), test_df.index.values[0::int(len(test_df.index.values)/15)], rotation=45, fontsize='small')
    plt.show()

# Plot training loss
def plot_losses(losses, title):
    plt.plot(losses)
    plt.title(title)
    plt.ylabel('MSE Loss Value')
    plt.xlabel('batch')
    plt.show()

if __name__ == "__main__":
    csv_file = "data/ftse100.csv"
    start_date = "2019-01-01"
    end_date = "2024-01-01"
    feature_names = ['adj_close', 'upper_bb', 'lower_bb']
    
    # Read the price data
    data = read_data(csv_file, start_date, end_date)

    # Add features
    data = add_features(data)

    # Split the data into train and test sets
    train_df, test_df = train_test_split(data, 0.6)

    # Fit the StandardScaler to the training data
    scaler = StandardScaler()
    scaler.fit(train_df)

    env = Trading_Environment(train_df, 2, feature_names, scaler)

    obs_shape = (env.num_features, env.observation_size)

    agent = DQN([3], env.num_action, n_hidden=128, buf_size=20000, batch_size=128)
    # dqn = DQN(input_shape, n_outputs, n_hidden=128, buf_size=20000, batch_size=128)
    avg_rewards = []
    ep_rewards = []
    
    # obs, info = env.reset()

    batch_size = 32
    batch_losses = []
    num_batches_trained = 0
    episode_count = 500
    time_steps = train_df.shape[0]


    for episode in range(1001):
        obs, info = env.reset()
        for step in range(time_steps + 1):
        # for step in tqdm(range(time_steps + 1), desc=f'Running episode {episode}/{episode_count}'):
            epsilon = max(1 - episode / 500, 0.01)
            obs, rewards, done, truncated, info = agent.play_one_step(env, obs, epsilon)
            if done or truncated:
                break
            ep_rewards.append(rewards)
        
        avg_rewards.append(np.sum(ep_rewards))
        ep_rewards = []

        if episode > 50:
            loss = agent.training_step()
            batch_losses.append(loss)

        if episode % 50 == 0:
            print(f"Ep: {episode} - Mean rewards: {np.mean(avg_rewards)}")
            avg_rewards = []
            agent.update_target_weights()

    # for episode in range(episode_count + 1):

    #     for t in tqdm.tqdm(range(time_steps), desc=f'Running episode {e}/{episode_count}'):
    #         action = agent.act(obs)
    #         next_obs, reward, terminated, truncated, info = env.next(action)
    #         done = terminated or truncated
    #         if t >= (time_steps-1):
    #             done = True

    #         agent.memory.append((obs, action, reward, next_obs, done))
    #         obs = next_obs
            
    #         if done:
    #             print('--------------------------------')
    #             print(f'Episode {e}')
    #             print(f'Total Profit: {format_price(env.total_profit)}')
    #             print(f'Total Winners: {format_price(env.total_winners)}')
    #             print(f'Total Losers: {format_price(env.total_losers)}')
    #             print(f'Max Loss: {max(batch_losses[num_batches_trained:len(batch_losses)])}')
    #             print(f'Total Loss: {sum(batch_losses[num_batches_trained:len(batch_losses)])}')
    #             print('--------------------------------')
    #             X_train = scaler.transform(train_df[feature_names]).values
    #             idx_close = 0
    #             idx_bb_upper = 1
    #             idx_bb_lower = 2
    #             plot_behavior(X_train[:, idx_close].flatten(), X_train[:, idx_bb_upper].flatten(), X_train[:, idx_bb_lower].flatten(), env.states_buy, env.states_sell, env.total_profit)
    #             plot_losses(batch_losses[num_batches_trained:len(batch_losses)], f'Episode {e} DQN model loss')
    #             num_batches_trained = len(batch_losses)

    #         if len(agent.memory) > batch_size:
    #             # when the size of the memory is greater than the batch size, run the exp_replay function on the batch to fit the model and get losses for the batch
    #             losses = agent.exp_replay(batch_size)    
    #             # then sum the losses for the batch and append them to the batch_losses list
    #             batch_losses.append(sum(losses))