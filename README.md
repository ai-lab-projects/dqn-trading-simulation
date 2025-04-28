# DQN Trading Agent for 1655 ETF

This project implements a reinforcement learning (DQN) based trading agent for the 1655 ETF (iShares S&P 500 ETF listed on the Tokyo Stock Exchange).

The agent learns from historical price data to make buy and sell decisions, aiming to maximize returns through sequential trading.

---

## 📖 Project Overview

- **Objective**: Develop an agent that can automatically decide when to buy and sell the 1655 ETF based on price movements.
- **Approach**:  
  - Use Deep Q-Networks (DQN) separately for buying and selling decisions.
  - Fetch historical price data automatically using `yfinance`.
  - Randomly search hyperparameters to explore better model settings.

This project focuses on **basic reinforcement learning** techniques applied to **real financial market data**.

---

## 🛠️ Setup

**Environment:**
- Python 3.10.7
- TensorFlow 2.13.0
- yfinance 0.2.56
- numpy 1.23.5
- pandas 2.0.3
- matplotlib 3.7.1
- scikit-learn 1.2.2

**Installation:**

Install required libraries if needed:

\`\`\`bash
pip install tensorflow yfinance numpy pandas matplotlib scikit-learn
\`\`\`

---

## ⚙️ Parameters

The agent uses a **random search** to explore hyperparameters such as:

| Parameter | Description |
|:---|:---|
| `nodes` | Number of nodes in each Dense layer (e.g., 16, 32) |
| `initializer` | Weight initialization method |
| `regularizer` | Weight regularization method |
| `batch_normalization` | Whether to use batch normalization |
| `dropout` | Whether to use dropout |
| `activation` | Activation function (e.g., relu, sigmoid) |
| `two_layers` | Use one or two Dense layers |
| `optimizer` | Optimizer type (e.g., SGD, Adam) |
| `learning_rate` | Learning rate |
| `epsilon` | Initial exploration rate |
| `epsilon_decay` | Decay rate of exploration |
| `batch_size` | Mini-batch size during training |

---

## 📈 Output

The script will automatically:
- Download historical price data (from 2017/09/29 to 2023/03/31)
- Train two DQN agents (buyer, seller)
- Save the trained models into `DQNmodels/`
- Save trial results (including parameters and evaluation metrics) into a pickle file (e.g., `results_20250428_1.pkl`)

✅ Models and results are saved periodically.

---

## 🧪 Example Result

After training, you will get evaluation metrics such as:

| Metric | Meaning |
|:---|:---|
| `total_reward` | Total cumulative return |
| `win_rate` | Winning rate (percentage of profitable trades) |
| `average_return` | Average return per trade |
| `total_trade` | Total number of trades made |
| `hold_rate` | Fraction of time spent holding positions |
| `p_value` | Statistical measure comparing to random trading |

Typical output examples:
- p-value: 0.034
- Total trades: 58
- Average return per trade: 0.52%

---

## 🚀 Future Work

- Train over longer historical periods
- Introduce additional technical indicators (e.g., moving averages, RSI)
- Explore advanced reinforcement learning methods (e.g., PPO, SAC)
- Improve reward shaping and risk management

---

## ⚠️ Disclaimer

- This project is for **educational and research purposes only**.
- **Do not use these models for actual trading or investment without careful consideration**.
- The model performance may vary depending on market conditions, and there is no guarantee of profitability.
