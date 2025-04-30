from train_dqn_etf import DQN, evaluate, RobustScaler, load_etf_data
import pickle

# モデル読み込み
with open('DQNmodels/buyer_2_14_20_20230727_173632.pkl', 'rb') as f:
    buyer = pickle.load(f)
with open('DQNmodels/seller_2_14_20_20230727_173632.pkl', 'rb') as f:
    seller = pickle.load(f)

# データ取得
_, _, test_close_prices, _, _, test_open_prices = load_etf_data()

# 評価
look_back = 20
scaler = RobustScaler()
result = evaluate(look_back, test_close_prices, test_open_prices, buyer, seller, scaler)
print(result)