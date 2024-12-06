import numpy as np
import pandas as pd
import math
from matplotlib import pyplot as plt
import random

# import yfinance as yf



def read_data(csv_file: str, start_date=None, end_date=None):
    """
    Reads a CSV file containing time-series data, cleans the column names, and filters the data by date range.

    Args:
        csv_file (str): Path to the CSV file to be read.
        start_date (str, datetime, optional): The start date to filter the data. 
                                               Can be a string (e.g. 'YYYY-MM-DD') or a datetime object.
                                               Defaults to None, meaning no filtering at the start.
        end_date (str, datetime, optional): The end date to filter the data. 
                                             Can be a string (e.g. 'YYYY-MM-DD') or a datetime object.
                                             Defaults to None, meaning no filtering at the end.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the filtered and cleaned data. The 'date' column is set as the index.

    Raises:
        FileNotFoundError: If the specified CSV file cannot be found.
        ValueError: If the CSV file does not contain a 'date' column or if the date values cannot be converted.

    Notes:
        - The 'date' column is expected to be in a format that can be parsed by pandas' `pd.to_datetime`.
        - Column names are converted to lowercase and spaces are replaced with underscores for consistency.
        - If `start_date` or `end_date` are provided, the data will be filtered to include only the rows within that range.
        - The filtering is inclusive of `start_date` and exclusive of `end_date` (i.e., `df.index >= start_date` and `df.index < end_date`).
    """
    # Try reading the CSV file
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        raise FileNotFoundError(f"The file {csv_file} was not found.")
    
    # Clean column names
    df.columns = [col.lower().replace(' ', '_') for col in df.columns]
    
    # Convert the 'date' column to datetime format, if not already
    if 'date' not in df.columns:
        raise ValueError("The CSV file must contain a 'date' column.")
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # Set 'date' column as index
    df.set_index('date', inplace=True)
    
    # Filter data based on start_date and end_date if provided
    if start_date:
        df = df[df.index >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df.index < pd.to_datetime(end_date)]

    return df

def add_features(df: pd.DataFrame):
    # Calculate 20-day bollinger bands
    df['ma_20'] = df.adj_close.rolling(20).mean()
    vol_20 = df.adj_close.rolling(20).std()
    df['upper_bb'] = df.ma_20 + vol_20 * 2
    df['lower_bb'] = df.ma_20 - vol_20 * 2
    df.dropna(inplace=True)
    return df

def train_test_split(df: pd.DataFrame, split=0.7):
    # split dataset df into train (50%) and test (50%) datasets
    split_idx = int(len(df) * split) + 1 
    train_df =  df.iloc[:split_idx]# define training dtaframe under this variable name
    test_df =  df.iloc[split_idx:]# define testing dtaframe under this variable name
    return train_df, test_df

# Format price string
def format_price(n):
    return ('-$' if n < 0 else '$') + '{0:.2f}'.format(abs(n))

class StandardScaler:

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data:pd.DataFrame):
        self.mean = data.mean()
        self.std = data.std()

    def transform(self, data:pd.DataFrame):
        return (data - self.mean) / self.std
    
    def fit_transform(self, data:pd.DataFrame):
        self.fit(data)
        return self.transform(data)


class Trading_Environment:

    def __init__(self, data:pd.DataFrame, window_size:int, feature_names:list, scaler:StandardScaler):
        self.data = data
        self.window_size = window_size
        self.time_step = 0
        self.last_time_step = len(self.data) - 1
        
        self.feature_names = feature_names
        self.scaler = scaler
        self.time_step=0
        self.terminated = False
        self.states_sell = []
        self.states_buy = []
        self.inventory = []
        self.total_profit = 0
        self.total_winners = 0
        self.total_losers = 0

        self.num_features = len(self.feature_names)
        self.observation_size = window_size - 1

    def reset(self):
        self.time_step = 0
        self.last_time_step = len(self.data) - 1
        self.terminated = False
        self.truncated = False
        self.states_sell_test = []
        self.states_buy_test = []
        self.inventory = []
        self.total_profit = 0
        self.total_winners = 0
        self.total_losers = 0
        self.current_data = self._get_current_data()
        state = self._get_state(self.data, self.time_step, self.window_size, self.feature_names, self.scaler)
        info = self._get_info()
        return state, info

    def next(self, action):
        self.time_step += 1
        print(self.time_step)
        if self.time_step >= self.last_time_step:
            self.terminated = True

        self.current_data = self._get_current_data()
        
        reward = 0
        if action == 1: # buy
            # inverse transform to get true buy price in dollars
            buy_price = self.current_data['adj_close']
            # append buy prive to inventory
            self.inventory.append(buy_price)
            # append time step to states_buy_test
            self.states_buy_test.append(self.time_step)
            print(f'Buy: {format_price(buy_price)}')

        elif action == 2 and len(self.inventory) > 0: # sell
            # get bought price from beginning of inventory
            bought_price = self.inventory.pop(0)
            # inverse transform to get true sell price in dollars
            sell_price = self.current_data['adj_close']
            # reward is max of profit (close price at time of sell - close price at time of buy)
            profit = sell_price - bought_price
            reward = np.max(profit, 0)
            # update total_test_profit
            self.total_profit += profit
            if profit >=0:
                self.total_winners += profit
            else:
                self.total_losers += profit
            # append time step to states_sell_test
            self.states_sell_test.append(self.time_step)
            print(f'Sell: {format_price(sell_price)} | Profit: {format_price(sell_price - bought_price)}')

        observation = self._get_state(self.data, 
                                      self.time_step, 
                                      self.window_size, 
                                      self.feature_names, 
                                      self.scaler)
        info = self._get_info()

        return observation, reward, self.terminated, self.truncated, info

    def sample(self):
        return random.randint(0, 2)
    
    def _get_current_data(self):
        current_data = self.data.iloc[self.time_step]
        return current_data

    def render(self):
        pass

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    # returns an an n-day state representation ending at time t
    def _get_state(self, data, t, n, feature_names, scaler): 
        # data is the dataset of interest which holds the state values (i.e. Close , BB Upper, BB Lower)
        # t is the current time step 
        # n is the size of the training window
        d = t - n
        # the first step is to get the window of the dataset at the current time step (eg. if window size is 1, we grab the previous and the current time step)
        # remember to define the special case for the first iteration, where there is no previous time step. See lesson X for a reminder of how to do this.
        if d >= 0:
            scaled_df = scaler.transform(data.iloc[d:t])
            window = scaled_df[feature_names].values
        else:
            scaled_df = scaler.transform(data.iloc[0])
            window = np.array([scaled_df[feature_names]] * n)
        res = []
        # print(window)
        for i in range(n - 1):
            res.append(self._sigmoid(window[i+1] - window[i]))
        
        # once we have our state data, we need to apply the sigmoid to each feature.
        # return an array holding the n-day sigmoid state representation
        return np.array(res)
    
    def _get_info(self):
        current_data = self._get_current_data()
        return {
            "date": current_data.index,
            "price": current_data['adj_close']
        }
    


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
    # print(len(train_df))
    # print(test_df.shape)

    # Fit the StandardScaler to the training data
    scaler = StandardScaler()
    scaler.fit(train_df)
    # print(scaler.transform(2000))
    # print(train_df)
    

    # scaled_df = scaler.transform(train_df.iloc[0])
    # window = np.array([scaled_df[feature_names]] * 2)
    # print(window)

    env = Trading_Environment(train_df, 2, feature_names, scaler)
    print(env.last_time_step)
    obs, info = env.reset()
    print(obs, info)

    while True:
        action = env.sample()
        next_obs, rewards, terminated, truncated, info = env.next(action)
        print(next_obs)
        if terminated or truncated:
            break

    
    
    # test_df.close.plot()
    # plt.show()