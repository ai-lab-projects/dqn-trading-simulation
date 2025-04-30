import datetime
import os
import pandas as pd
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
import math
import yfinance as yf
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer

def is_running_on_colab():
    return 'COLAB_GPU' in os.environ

def remove_comma(val):
    return float(val.replace(',', ''))

def split_data(data, train_ratio=0.6, val_ratio=0.2):
    train_end = int(len(data) * train_ratio)
    val_end = train_end + int(len(data) * val_ratio)
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    return train_data, val_data, test_data

def calculate_rsi(prices):
    deltas = np.diff(prices)
    positive_deltas = deltas.copy()
    negative_deltas = deltas.copy()
    positive_deltas[deltas < 0] = 0
    negative_deltas[deltas > 0] = 0
    average_gain = np.mean(positive_deltas)
    average_loss = np.abs(np.mean(negative_deltas))
    
    if average_gain == 0 and average_loss == 0:
        rsi = 50
    else:
        rsi = 100 - 100 * (average_loss / (average_gain + average_loss))
    
    return rsi

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.memory.append(experience)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

class DQN:
    def __init__(self, input_size, output_size, params):
        self.input_size = input_size
        self.output_size = output_size
        self.params = params
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()
        self.epsilon = params["epsilon"]

    def build_layer(self, input_dim=None):
        layer = tf.keras.models.Sequential()

        kwargs = {
            'units': self.params['nodes'],
            'kernel_regularizer': self.params['regularizer'],
            'kernel_initializer': self.params['initializer'](),
            'use_bias': not self.params['batch_normalization']
        }

        if input_dim is not None:
            kwargs['input_dim'] = input_dim

        layer.add(tf.keras.layers.Dense(**kwargs))

        # Optional Batch Normalization before Activation
        if self.params['batch_normalization'] and self.params['batchnorm_before_activation']:
            layer.add(tf.keras.layers.BatchNormalization())

        # Optional Dropout before Activation
        if self.params['dropout'] and self.params['dropout_before_activation']:
            layer.add(tf.keras.layers.Dropout(self.params['dropout_rate']))

        # Activation
        layer.add(tf.keras.layers.Activation(self.params['activation']))

        # Optional Batch Normalization after Activation
        if self.params['batch_normalization'] and not self.params['batchnorm_before_activation']:
            layer.add(tf.keras.layers.BatchNormalization())

        # Optional Dropout after Activation
        if self.params['dropout'] and not self.params['dropout_before_activation']:
            layer.add(tf.keras.layers.Dropout(self.params['dropout_rate']))

        return layer


    def build_model(self):
        model = tf.keras.models.Sequential()

        # First layer
        model.add(self.build_layer(input_dim=self.input_size))

        # Optional second layer
        if self.params['two_layers']:
            model.add(self.build_layer())

        model.add(tf.keras.layers.Dense(self.output_size, activation='linear'))

        model.compile(loss=self.params['loss'], optimizer=self.params['optimizer'](learning_rate=self.params['learning_rate']))

        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def get_action(self, state, epsilon = 0):
        if np.random.rand() <= epsilon:
            return random.randrange(self.output_size)
        q_values = self.model.predict(state, verbose = 0)
        return np.argmax(q_values[0])

    def train(self, state_batch, q_values):
        self.model.fit(state_batch, q_values, batch_size=self.params['batch_size'], epochs=1, verbose=0)

    def update_epsilon(self):
        if self.epsilon > self.params['epsilon_min']:
            self.epsilon *= self.params['epsilon_decay']

def calculate_input_data(close_prices, buy_price, t, t_buy):
    recent_close_prices = close_prices[t_buy:t+1]
    average = np.mean(recent_close_prices)
    input_data = [
        100*(close_prices[t] - buy_price) / buy_price,
        100*(close_prices[t] - average) / average,
        0.01*calculate_rsi(recent_close_prices),
        0.5*np.log(t-t_buy)
    ]
    return np.reshape(input_data, [1, 4])

def train_dqn(dqn, memory, batch_size, gamma):
    if len(memory.memory) >= batch_size:
        batch = memory.sample(batch_size)
        state_batch = np.array([experience[0] for experience in batch])
        action_batch = np.array([experience[1] for experience in batch])
        reward_batch = np.array([experience[2] for experience in batch])
        next_state_batch = np.array([experience[3] for experience in batch])
        done_batch = np.array([experience[4] for experience in batch])

        q_values_next = dqn.target_model.predict(next_state_batch, verbose=0)
        max_q_values_next = np.amax(q_values_next, axis=1)
        q_target_batch = np.where(done_batch, reward_batch, reward_batch + gamma * max_q_values_next)

        q_values = dqn.model.predict(state_batch, verbose=0)
        
        try:
            q_values[np.arange(batch_size), action_batch] = q_target_batch
        except IndexError as e:
            print("np.arange(batch_size) is:")
            print(np.arange(batch_size))
            print("action_batch is:")
            print(action_batch)
            print("type(action_batch[0]) is:")
            print(type(action_batch[0]))
            raise e  # raise the error again so that it is not ignored
        
        dqn.train(state_batch, q_values)

def calculate_p_value(open_prices, look_back, hold_rate=0.5, threshold=0.05):
    # remove first 'look_back' elements
    open_prices = open_prices[look_back:]

    # calculate the number of random samples
    num_samples = round((len(open_prices)-1) * hold_rate)
    
    # initialize counters
    count = 0
    exceed = 0
    profits = []

    if 0<hold_rate<0.5 or 0.5<hold_rate<1:
        # repeat until the inverse of the square root of the exceed count drops below 0.1
        while True:
            count += 1
            # draw random samples without replacement
            samples = sorted(np.random.choice(range(len(open_prices)-1), size=num_samples, replace=False))
            profit = 0

            # iterate over the randomly chosen samples
            i = 0
            while i < len(samples):
                start = samples[i]
                end = start
                while i + 1 < len(samples) and samples[i + 1] == samples[i] + 1:
                    end = samples[i + 1]
                    i += 1
                profit += (open_prices[end + 1] - open_prices[start]) / open_prices[start]
                profits.append(profit)
                i += 1

            # check if the sum exceeds the threshold
            if profit > threshold:
                exceed += 1

            # stop if the number of iterations exceeds 1000 and return p_value = 0.001
            if count > 1000:
                break

        # calculate p-value
        if exceed>0:
            p_value = exceed / count
        else:
            p_value = 0.001
    else:
        p_value = 1
        
    if len(profits)>0:
        total_reward_over_mean = threshold/np.mean(profits)
    else:
        total_reward_over_mean = 0

    return p_value, total_reward_over_mean

def evaluate(look_back, close_prices, open_prices, buyer, seller, scaler):
    total_reward = 0
    total_win = 0
    total_trade = 0
    total_hold_days = 0
    records = []
    t = look_back
    t_end = len(close_prices) - 2
    while t <= t_end:
        state = np.array(scaler.fit_transform(close_prices[t-look_back+1:t+1].reshape(-1, 1)))  
        state = np.reshape(state, [1, look_back])
        buyer_action = buyer.get_action(state)  
        if buyer_action == 1:  
            buy_price = open_prices[t+1]
            t_buy = t
            t += 1
            while t <= t_end:
                input_data = calculate_input_data(close_prices, buy_price, t, t_buy)
                seller_action = seller.get_action(input_data)  
                if seller_action == 1:  
                    total_trade += 1
                    total_hold_days += (t - t_buy)
                    sell_price = open_prices[t + 1]  
                    reward = sell_price / buy_price - 1  
                    total_reward += reward
                    records.append([t_buy,buy_price,t,sell_price,reward])
                    if reward > 0:
                        total_win += 1
                    t += 1
                    break
                t += 1
        else:  
            t += 1

    win_rate = total_win / total_trade if total_trade != 0 else 0
    average_return = total_reward / total_trade if total_trade != 0 else 0
    hold_rate = total_hold_days / (t_end - look_back)
    average_hold_days = total_hold_days / total_trade if total_trade != 0 else 0
    p_value, total_reward_over_mean = calculate_p_value(open_prices, look_back, hold_rate=hold_rate, threshold=total_reward)
    print("p value: ", p_value)

    return total_reward, win_rate, average_return, total_trade, hold_rate, average_hold_days, p_value, total_reward_over_mean, records
#               0           1             2             3           4             5            6            7                     8

def save_dqn_instance(instance, role, execution_id, trial, episode):
    
    now = datetime.datetime.now()
    
    date_time = now.strftime("%Y%m%d_%H%M%S")

    filename = f"{role}_{execution_id}_{trial}_{episode}_{date_time}.pkl"

    directory = "DQNmodels"

    if not os.path.exists(directory):
        os.makedirs(directory)
    
    filepath = os.path.join(directory, filename)

    with open(filepath, 'wb') as f:
        pickle.dump(instance, f)
        
    return filename

def train_and_evaluate(look_back, params, memory_size, episodes, scaler, gamma, update_target_interval, trial):
    #epsilon = params["epsilon"]
    batch_size = params["batch_size"]
    
    tf.keras.backend.clear_session()
    
    buyer = DQN(look_back, 2, params)
    seller = DQN(4, 2, params)

    buyer_memory = ReplayMemory(memory_size)
    seller_memory = ReplayMemory(memory_size)

    train_results = []
    val_results = []

    buyer_experience_counter = 0
    seller_experience_counter = 0

    print("evaluation on training data")
    train_results.append(evaluate(look_back, train_close_prices, train_open_prices, buyer, seller, scaler))
    
    print("evaluation on validation data")
    val_results.append(evaluate(look_back, val_close_prices, val_open_prices, buyer, seller, scaler))

    prev_total_rewards = []
    
    pp_best = 0.1**2
    files = []
    
    for episode in range(episodes):
        print("episode:", episode)
        t = look_back
        t_end = len(train_close_prices)-2
        state = None
        while t <= t_end:
            if state is None:
                state = np.array(scaler.fit_transform(train_close_prices[t-look_back+1:t+1].reshape(-1, 1)))  
                state = np.reshape(state, [1, look_back])
            buyer_action = buyer.get_action(state, buyer.epsilon)
            if buyer_action == 0:
                reward = 0  
                t += 1
            else:
                buy_price = train_open_prices[t+1]  
                t_buy=t
                t += 1
                input_data = None
                while t <= t_end:
                    if input_data is None:
                        input_data = calculate_input_data(train_close_prices, buy_price, t, t_buy)
                    seller_action = seller.get_action(input_data, seller.epsilon)
                    done = False
                    if seller_action == 0:
                        reward = 0  
                        t += 1
                        if t <= t_end:
                            pass
                        else:
                            done = True
                    else:
                        sell_price = train_open_prices[t + 1]  
                        reward = 100*(sell_price - buy_price) / buy_price  
                        t += 1
                        done = True
                    next_input_data = calculate_input_data(train_close_prices, buy_price, t, t_buy)
                    seller_memory.push(input_data[0], seller_action, reward, next_input_data[0], done)
                    input_data = next_input_data
                    
                    seller_experience_counter += 1
                    if seller_experience_counter % batch_size == 0:
                        train_dqn(seller, seller_memory, batch_size, gamma)

                    if seller_experience_counter % update_target_interval == 0:
                        seller.update_target_model()

                    if done:
                        break

            next_state = np.array(scaler.fit_transform(train_close_prices[t-look_back+1:t+1].reshape(-1, 1)))  
            next_state = np.reshape(state, [1, look_back])
            done = t > t_end
            buyer_memory.push(state[0], buyer_action, reward, next_state[0], done)
            state = next_state
            
            buyer_experience_counter += 1
            if buyer_experience_counter % batch_size == 0:
                train_dqn(buyer, buyer_memory, batch_size, gamma)
            if buyer_experience_counter % update_target_interval == 0:
                buyer.update_target_model()

        print("evaluation on training data")
        train_result = evaluate(look_back, train_close_prices, train_open_prices, buyer, seller, scaler)
        
        print("evaluation on validation data")
        val_result = evaluate(look_back, val_close_prices, val_open_prices, buyer, seller, scaler)
        
        train_results.append(train_result)
        val_results.append(val_result)

        # Add the latest total reward to the list
        prev_total_rewards.append(val_result[0])
        # If the list is too long, remove the oldest total reward
        if len(prev_total_rewards) > 5:
            prev_total_rewards.pop(0)

        # Check if the total reward has been the same for the last 5 episodes
        if len(prev_total_rewards) == 5 and all(r == prev_total_rewards[0] for r in prev_total_rewards):
            break
        
        pp = train_result[6]*val_result[6]
        if pp < pp_best:
            pp_best = pp
            filename_buyer = save_dqn_instance(buyer, "buyer", execution_id, trial, episode)
            filename_seller = save_dqn_instance(seller, "seller", execution_id, trial, episode)
            files.append([filename_buyer,filename_seller])
        
    return train_results, val_results, files

def create_params():
    params_choices = {
        'nodes': [16, 32],
        'initializer': [
            tf.keras.initializers.Ones, tf.keras.initializers.RandomNormal, tf.keras.initializers.RandomUniform, tf.keras.initializers.TruncatedNormal, tf.keras.initializers.VarianceScaling, tf.keras.initializers.Orthogonal, tf.keras.initializers.Identity, tf.keras.initializers.lecun_uniform, tf.keras.initializers.glorot_normal, tf.keras.initializers.glorot_uniform, tf.keras.initializers.he_normal, tf.keras.initializers.he_uniform],
        'regularizer': [None, tf.keras.regularizers.l1(0.01), tf.keras.regularizers.l2(0.01), tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01)],
        'batch_normalization': [True, False],
        'batchnorm_before_activation': [True, False],
        'dropout': [True, False],
        'dropout_before_activation': [True, False],
        'dropout_rate': [0.25, 0.5, 0.75],
        'activation': ['relu', 'sigmoid', 'tanh', 'elu', 'selu', 'softplus', 'softsign'],
        'two_layers': [True, False],
        'loss': ['mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error', 'mean_squared_logarithmic_error', 'squared_hinge', 'hinge', 'huber', 'logcosh', 'poisson'],
        'optimizer': [tf.keras.optimizers.SGD, tf.keras.optimizers.RMSprop, tf.keras.optimizers.Adam, 
                      tf.keras.optimizers.Adagrad, tf.keras.optimizers.Adamax, tf.keras.optimizers.Nadam, tf.keras.optimizers.Ftrl],
        'learning_rate': [0.01, 0.001, 0.0001],
        'epsilon': [1],
        'epsilon_min': [0.01, 0.001, 0.0001],
        'epsilon_decay': [0.99, 0.995, 0.999],
        'batch_size': [32, 64]
    }
    
    # Randomly select one value for each hyperparameter and return as a dict
    params = {param: random.choice(values) for param, values in params_choices.items()}
    
    return params

def load_etf_data():
    ticker = "1655.T"
    start_date = "2017-09-29"
    end_date = "2023-03-31"

    etf_1655 = yf.download(ticker, start=start_date, end=end_date)

    etf_1655_cleaned = etf_1655[etf_1655['Close'] > 50]

    close_prices = etf_1655_cleaned['Close'].values.flatten()
    open_prices = etf_1655_cleaned['Open'].values.flatten()

    train_close_prices, val_close_prices, test_close_prices = split_data(close_prices)
    train_open_prices, val_open_prices, test_open_prices = split_data(open_prices)
    
    return train_close_prices, val_close_prices, test_close_prices, train_open_prices, val_open_prices, test_open_prices

if __name__ == "__main__":
    
    save = True
    execution_id = 1
    file = f"results_20250428_{execution_id}.pkl"

    train_close_prices, val_close_prices, test_close_prices, train_open_prices, val_open_prices, test_open_prices = load_etf_data()

    if os.path.exists(file):
        with open(file, 'rb') as f:
            results = pickle.load(f)
    else:
        results = []
    
    trial = 1
    
    while True:

        print("trial:", trial)

        look_back = random.choice([10, 20])
        params = create_params()
        memory_size = random.choice([1000])
        episodes = 30

        scaler_choices = [StandardScaler(), MinMaxScaler(),RobustScaler()]
        scaler = random.choice(scaler_choices)

        gamma = random.choice([0.95,0.99,0.995])
        update_target_interval = random.choice([100,200])

        params.update({
        'look_back': look_back,
        'memory_size': memory_size,
        'episodes': episodes,
        'scaler': scaler,
        'gamma': gamma,
        'update_target_interval': update_target_interval
        })

        print(params)

        train_results, val_results, files = train_and_evaluate(look_back, params, memory_size, episodes, scaler, gamma, update_target_interval, trial)

        # Store the results in a dictionary
        trial_results = {
            'params': params,
            'train_results': train_results,
            'val_results': val_results, 
            'files': files
        }

        # Append to the results list
        results.append(trial_results)

        # Save to a pickle file after each trial
        if save:
            with open(file, 'wb') as f:
                pickle.dump(results, f)
        
        trial += 1
