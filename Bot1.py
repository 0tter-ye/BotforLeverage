import ccxt  # Import ccxt to interact with Binance API
import time  # Used for sleeping between API requests
import pandas as pd  # Used for handling data
import numpy as np  # Needed for Markov Chain and numerical operations
import random  # Needed for random state selection
import logging  # For logging errors and status updates
from concurrent.futures import ThreadPoolExecutor
from config import API_KEY, API_SECRET  # Load API keys

# Scikit-learn imports for ML models
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error

# TensorFlow/Keras imports for RNN model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Initialize Binance Futures API
exchange = ccxt.binance({
    'apiKey': API_KEY,
    'secret': API_SECRET,
    'options': {'defaultType': 'future'}  # Enable futures trading
})
exchange.set_sandbox_mode(False)  # Test mode (set to False for live trading)

# Configure logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Function to Handle Errors
def handle_error(error):
    logging.error(f"Error: {error}")

# Function to Set Leverage
def set_leverage(symbol, leverage):
    """Set leverage for a specific trading pair."""
    try:
        exchange.fapiPrivatePostLeverage({
            'symbol': symbol.replace("/", ""),
            'leverage': leverage
        })
        logging.info(f"Leverage set to {leverage}x for {symbol}")
    except Exception as e:
        handle_error(e)

# Function to Fetch OHLCV Data
def fetch_data(symbol, timeframe='1h', limit=100):
    """Fetch historical market data for analysis."""
    candles = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

# Function to Calculate Market Volatility
def calculate_volatility(df):
    """Calculate volatility using ATR (Average True Range)."""
    df['hl'] = df['high'] - df['low']
    df['hc'] = abs(df['high'] - df['close'].shift(1))
    df['lc'] = abs(df['low'] - df['close'].shift(1))
    df['true_range'] = df[['hl', 'hc', 'lc']].max(axis=1)
    atr = df['true_range'].rolling(window=14).mean()  # 14-period ATR
    return atr.iloc[-1]  # Return latest ATR value

# Function to Determine Optimal Leverage
def determine_leverage(symbol):
    """Dynamically adjust leverage based on market volatility."""
    df = fetch_data(symbol, '1h', 100)  # Fetch 1-hour data
    volatility = calculate_volatility(df)
    
    # Define thresholds for volatility-based leverage scaling
    if volatility < 100:
        leverage = 20  # Low volatility → Max leverage
    elif volatility < 300:
        leverage = 10  # Medium volatility → Moderate leverage
    elif volatility < 500:
        leverage = 5  # High volatility → Lower leverage
    else:
        leverage = 1  # Extreme volatility → Min leverage

    set_leverage(symbol, leverage)
    return leverage

# Function to Place a Trade with AI-Decided Leverage
def place_order(symbol, side, amount, sl=None, tp=None, tsl=None, price=None, order_type='market'):
    """Place a trade with AI-adjusted leverage and risk management."""
    
    # Ensure leverage is set before placing an order
    leverage = determine_leverage(symbol)  # AI decides leverage
    set_leverage(symbol, leverage)  # Set leverage for this symbol

    try:
        logging.info(f"Placing order: {side} {amount} of {symbol} at {order_type} price")
        # Place the order based on the type (market or limit)
        if order_type == 'market':
            order = exchange.create_order(
                symbol=symbol,
                type='market',
                side=side,  # 'buy' for long, 'sell' for short
                amount=amount
            )
        elif order_type == 'limit' and price:
            order = exchange.create_order(
                symbol=symbol,
                type='limit',
                side=side,
                amount=amount,
                price=price
            )
        else:
            raise ValueError("Invalid order type or missing price for limit order.")
        
        logging.info(f"Order placed: {order}")

        # Get the filled price of the order
        order_price = float(order['price']) if 'price' in order else None

        # Set Stop-Loss & Take-Profit Orders
        if order_price:
            if sl:
                stop_loss_price = round(order_price * (1 - sl/100) if side == 'buy' else order_price * (1 + sl/100), 2)
                sl_order = exchange.create_order(
                    symbol=symbol,
                    type='stop_market',
                    side='sell' if side == 'buy' else 'buy',
                    amount=amount,
                    params={'stopPrice': stop_loss_price}
                )
                logging.info(f"Stop-Loss Order placed at {stop_loss_price}")

            if tp:
                take_profit_price = round(order_price * (1 + tp/100) if side == 'buy' else order_price * (1 - tp/100), 2)
                tp_order = exchange.create_order(
                    symbol=symbol,
                    type='take_profit_market',
                    side='sell' if side == 'buy' else 'buy',
                    amount=amount,
                    params={'stopPrice': take_profit_price}
                )
                logging.info(f"Take-Profit Order placed at {take_profit_price}")

            # Set Trailing Stop-Loss (TSL)
            if tsl:
                tsl_order = exchange.create_order(
                    symbol=symbol,
                    type='trailing_stop_market',
                    side='sell' if side == 'buy' else 'buy',
                    amount=amount,
                    params={'activationPrice': order_price, 'callbackRate': tsl}
                )
                logging.info(f"Trailing Stop-Loss set at {tsl}% callback rate")

        return order

    except Exception as e:
        handle_error(e)
        return None

# Function to Fetch Order Book and Imbalance
def fetch_order_book(symbol, depth=10):
    """Fetch order book imbalance for a given symbol."""
    order_book = exchange.fetch_order_book(symbol)
    bids = order_book['bids'][:depth]
    asks = order_book['asks'][:depth]
    
    bid_volume = sum([bid[1] for bid in bids])
    ask_volume = sum([ask[1] for ask in asks])
    
    imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)
    return imbalance, bids, asks

# Function to Fetch Data for Multiple Timeframes Concurrently
def fetch_data_threaded(symbol, timeframes, limit=100):
    """Fetch historical market data for multiple timeframes concurrently."""
    all_data = {}
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(fetch_data, symbol, timeframe, limit): timeframe for timeframe in timeframes}
        results = {future.result(): timeframe for future in futures}

    # Apply microstructure noise and order book imbalance to data
    for df, timeframe in results.items():
        noise = np.random.normal(0, 0.0005, size=df.shape[0])
        df['close'] += df['close'] * noise
        df['open'] += df['open'] * noise
        df['high'] += df['high'] * noise
        df['low'] += df['low'] * noise
        
        df['vbi'], _, _ = fetch_order_book(symbol, depth=10)
        all_data[timeframe] = df

    return all_data

# Function to Train Machine Learning Model (Random Forest Example)
def train_rf_model(df):
    """Train Random Forest model to predict price movement."""
    df['momentum'] = df['close'].diff()
    df['volatility'] = df['high'] - df['low']
    X = df[['momentum', 'volatility']]
    y = np.sign(df['close'].diff().shift(-1))  # Predict price movement

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    logging.info(f"Model Accuracy: {accuracy}")
    return model

# Example Usage: Place an Order with AI-Decided Leverage
symbol = 'BTC/USDT'
trade_amount = 0.001  # Adjust based on risk
place_order(symbol, 'buy', trade_amount, sl=2, tp=5, tsl=1, order_type='market')  # Open long position
place_order(symbol, 'sell', trade_amount, sl=3, tp=6, tsl=1.5, order_type='market')  # Open short position

# Example: Train the Random Forest Model on Historical Data
df = fetch_data(symbol, '1h', limit=100)
model = train_rf_model(df)

# Fetch data for multiple timeframes
timeframes = ['1m', '5m', '15m', '1h']
data_dict = fetch_data_threaded(symbol, timeframes, limit=100)

# Example of viewing fetched data
for tf, df in data_dict.items():
    logging.info(f"\nTimeframe: {tf}")
    logging.info(df.head())

class MarkovChain:
    def __init__(self, states, transition_matrix):
        self.states = states
        self.transition_matrix = transition_matrix
        self.current_state = random.choice(states)  # Start in a random state

    def next_state(self):
        """Transitions to the next state based on the transition matrix."""
        probabilities = self.transition_matrix[self.states.index(self.current_state)]
        self.current_state = np.random.choice(self.states, p=probabilities)
        return self.current_state

# Define the states
states = ["ML Assessment and Rescore", "Markov Chain 2", "ML Validation",
          "Random Forest", "RNN Strategy Selection", "Execution"]

# Define the transition matrix
transition_matrix = np.array([
    [0.0, 0.7, 0.3, 0.0, 0.0, 0.0],  # ML Assessment → Markov Chain or Validation
    [0.0, 0.0, 0.0, 0.7, 0.3, 0.0],  # Markov Chain → RF or RNN
    [0.0, 0.0, 0.0, 0.7, 0.3, 0.0],  # Validation → RF or RNN
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # RF → Execution
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # RNN → Execution
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]   # Execution (end)
], dtype=float)

markov_chain = MarkovChain(states, transition_matrix)

def ml_assessment_and_rescore(data):
    """Dummy ML Assessment function."""
    print("Performing ML Assessment and Rescore...")
    return data + 0.1  

def ml_validation(data, labels):
    """Performs train-test split."""
    print("Performing ML Validation...")
    return train_test_split(data, labels, test_size=0.2, random_state=42)

def train_random_forest(X_train, y_train, problem_type="classification"):
    """Trains a Random Forest model."""
    print("Training Random Forest...")
    if problem_type == "classification":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def rnn_strategy_selection(data, look_back=10):
    """Trains an RNN for strategy selection."""
    print("Training RNN for Strategy Selection...")
    X, y = [], []
    for i in range(len(data) - look_back - 1):
        X.append(data[i:(i + look_back)])
        y.append(data[i + look_back])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    model = Sequential([
        LSTM(50, input_shape=(look_back, 1)),
        Dense(1)
    ])
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X, y, epochs=1, batch_size=1, verbose=0)  

    return model, look_back 

def execute_strategy(model, data, strategy_type="RF", look_back=10):
    """Executes a trading strategy based on model predictions."""
    print(f"Executing Strategy ({strategy_type})...")
    if strategy_type == "RF":
        prediction = model.predict(data)
        decision = "Buy" if prediction[0] > 0.5 else "Sell"
    elif strategy_type == "RNN":
        last_sequence = np.reshape(data[-look_back:], (1, look_back, 1))
        prediction = model.predict(last_sequence)
        decision = "Buy" if prediction[0][0] > 0.5 else "Sell"
    else:
        print("Error: Invalid strategy type.")
        return
    print(f"Decision: {decision}")
    return decision

# Initial Data
initial_data = np.random.rand(100, 5)  
labels = np.random.randint(0, 2, 100)  

current_data = initial_data
current_state = markov_chain.current_state
strategy = None

print(f"Starting State: {current_state}")

while current_state != "Execution":
    if current_state == "ML Assessment and Rescore":
        current_data = ml_assessment_and_rescore(current_data)
    elif current_state == "ML Validation":
        X_train, X_test, y_train, y_test = ml_validation(current_data, labels)
        current_data, labels = X_test, y_test
    elif current_state == "Random Forest":
        problem_type = "classification" if len(np.unique(labels)) <= 10 else "regression"
        model = train_random_forest(X_train, y_train, problem_type)
        strategy = "RF"
    elif current_state == "RNN Strategy Selection":
        rnn_model, look_back_value = rnn_strategy_selection(current_data[:, 0])
        strategy = "RNN"

    current_state = markov_chain.next_state()
    print(f"Next State: {current_state}")

if strategy == "RF":
    execute_strategy(model, current_data, "RF")
elif strategy == "RNN":
    execute_strategy(rnn_model, current_data[:, 0], "RNN", look_back_value)

print("Finished.")

