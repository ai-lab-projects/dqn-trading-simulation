import yfinance as yf
from train_dqn_etf import DQN, evaluate, RobustScaler, load_etf_data
import pickle

def load_latest_etf_data(ticker="1655.T", num_latest=100):
    
    start_date = "2017-01-01"
    end_date = None  

    etf_data = yf.download(ticker, start=start_date, end=end_date)

    etf_cleaned = etf_data[etf_data['Close'] > 50].tail(num_latest)
    

    close_prices = etf_cleaned['Close'].values.flatten()
    open_prices = etf_cleaned['Open'].values.flatten()

    return close_prices, open_prices

def latest_action():

    close_prices, open_prices = load_latest_etf_data()

    with open('DQNmodels/buyer_2_14_20_20230727_173632.pkl', 'rb') as f:
        buyer = pickle.load(f)
    with open('DQNmodels/seller_2_14_20_20230727_173632.pkl', 'rb') as f:
        seller = pickle.load(f)

    look_back = 20
    scaler = RobustScaler()
    result = evaluate(look_back, close_prices, open_prices, buyer, seller, scaler)
    return result[-1][-1][-1]

if __name__ == "__main__":
    print(latest_action())