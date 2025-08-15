import numpy as np
import pandas as pd
import gym
from gym import spaces
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class TradingEnv(gym.Env):
    """
    A custom OpenAI Gym environment for cryptocurrency trading.
    Compatible with stable-baselines3 for reinforcement learning.
    """
    metadata = {'render.modes': ['human']}
    
    def __init__(
        self,
        df: pd.DataFrame,
        window_size: int = 24,
        initial_balance: float = 10000.0,
        transaction_fee: float = 0.001,
        reward_scaling: float = 1.0,
        use_position_features: bool = True
    ):
        """
        Initialize the trading environment.
        
        Args:
            df: DataFrame with OHLCV data (must have datetime index and columns: open, high, low, close, volume)
            window_size: Number of previous candles to include in the observation
            initial_balance: Starting balance in USD
            transaction_fee: Fee per transaction as a fraction (0.001 = 0.1%)
            reward_scaling: Scale factor for rewards
            use_position_features: Whether to include current position info in the observation
        """
        super(TradingEnv, self).__init__()
        
        # Validate and store input data
        self._validate_dataframe(df)
        self.df = df
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        self.reward_scaling = reward_scaling
        self.use_position_features = use_position_features
        
        # Set up state variables
        self.current_step = 0
        self.balance = initial_balance
        self.asset_held = 0.0
        self.current_price = 0.0
        self.total_fee_paid = 0.0
        self.total_trades = 0
        self.trade_history = []
        
        # Define action and observation spaces
        # Actions: 0=hold, 1=buy, 2=sell
        self.action_space = spaces.Discrete(3)
        
        # Calculate number of features in the observation space
        base_features = self._calculate_indicators().shape[1] * self.window_size
        position_features = 2 if use_position_features else 0  # Balance and asset held
        
        # Observation space: normalized prices, indicators, and optionally position data
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(base_features + position_features,), dtype=np.float32
        )
    
    def _validate_dataframe(self, df: pd.DataFrame) -> None:
        """Validate that the dataframe has the required columns"""
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in df.columns]
            raise ValueError(f"DataFrame missing required columns: {missing}")
        
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DatetimeIndex")
    
    def _calculate_indicators(self) -> pd.DataFrame:
        """Calculate trading indicators for the dataset"""
        df = self.df.copy()
        
        # Price features
        df['returns'] = df['close'].pct_change().fillna(0)
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1)).fillna(0)
        
        # Volume features
        df['volume_change'] = df['volume'].pct_change().fillna(0)
        
        # Simple moving averages
        df['sma_10'] = df['close'].rolling(10).mean().bfill() / df['close'] - 1
        df['sma_30'] = df['close'].rolling(30).mean().bfill() / df['close'] - 1
        
        # Volatility
        df['volatility'] = df['returns'].rolling(24).std().fillna(0)
        
        # RSI - Relative Strength Index
        delta = df['close'].diff().fillna(0)
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean().fillna(0)
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean().fillna(0)
        # Add small epsilon to prevent division by zero
        epsilon = 1e-10
        rs = gain / (loss + epsilon)
        df['rsi'] = (100 - (100 / (1 + rs))).fillna(50)  # Default RSI to neutral 50 if NaN
        
        # MACD - Moving Average Convergence Divergence
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        df['macd'] = (macd - signal).fillna(0)
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(20).mean().fillna(df['close'])
        std = df['close'].rolling(20).std().fillna(0)
        df['bb_upper'] = (df['bb_middle'] + 2 * std) / df['close'] - 1
        df['bb_lower'] = (df['bb_middle'] - 2 * std) / df['close'] - 1
        
        # Drop any NaN values
        # df = df.dropna()  # We're now handling NaNs with fillna instead of dropping
        
        # Check if there are any NaN values left
        if df.isna().any().any():
            logger.warning("NaN values found in calculated indicators. Filling remaining NaNs with 0.")
            df = df.fillna(0)
        
        # Normalize using MinMaxScaler principles
        features = df.drop(['open', 'high', 'low', 'close', 'volume'], axis=1)
        
        return features
    
    def reset(self) -> np.ndarray:
        """Reset the environment to the initial state and return the initial observation"""
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.asset_held = 0.0
        self.current_price = self.df['close'].iloc[self.current_step]
        self.total_fee_paid = 0.0
        self.total_trades = 0
        self.trade_history = []
        
        return self._get_observation()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take an action in the environment.
        
        Args:
            action: 0=hold, 1=buy, 2=sell
            
        Returns:
            observation: Current observation
            reward: Reward from the action
            done: Whether the episode is over
            info: Additional information
        """
        previous_portfolio_value = self._calculate_portfolio_value()
        
        # Execute the trading action
        self._take_action(action)
        
        # Move to the next time step
        self.current_step += 1
        
        # Check if the episode is over
        done = self.current_step >= len(self.df) - 1
        
        # Calculate reward based on portfolio change
        current_portfolio_value = self._calculate_portfolio_value()
        reward = ((current_portfolio_value / previous_portfolio_value) - 1) * self.reward_scaling
        
        # Get the current observation
        observation = self._get_observation()
        
        # Prepare info dictionary
        info = {
            'portfolio_value': current_portfolio_value,
            'balance': self.balance,
            'asset_held': self.asset_held,
            'current_price': self.current_price,
            'total_fee_paid': self.total_fee_paid,
            'total_trades': self.total_trades
        }
        
        return observation, reward, done, info
    
    def _take_action(self, action: int) -> None:
        """Execute the trading action"""
        self.current_price = self.df['close'].iloc[self.current_step]
        
        if action == 1:  # Buy
            # Calculate maximum amount of asset that can be bought with current balance
            max_asset_to_buy = self.balance / (self.current_price * (1 + self.transaction_fee))
            
            if max_asset_to_buy > 0:
                # Buy the maximum amount
                fee = max_asset_to_buy * self.current_price * self.transaction_fee
                self.balance -= (max_asset_to_buy * self.current_price + fee)
                self.asset_held += max_asset_to_buy
                self.total_fee_paid += fee
                self.total_trades += 1
                
                # Record the trade
                self.trade_history.append({
                    'timestamp': self.df.index[self.current_step],
                    'action': 'buy',
                    'price': self.current_price,
                    'amount': max_asset_to_buy,
                    'fee': fee
                })
                
        elif action == 2:  # Sell
            if self.asset_held > 0:
                # Sell all assets
                sale_value = self.asset_held * self.current_price
                fee = sale_value * self.transaction_fee
                self.balance += sale_value - fee
                self.total_fee_paid += fee
                self.total_trades += 1
                
                # Record the trade
                self.trade_history.append({
                    'timestamp': self.df.index[self.current_step],
                    'action': 'sell',
                    'price': self.current_price,
                    'amount': self.asset_held,
                    'fee': fee
                })
                
                self.asset_held = 0
    
    def _get_observation(self) -> np.ndarray:
        """Return the current observation"""
        # Get feature data for the window
        features = self._calculate_indicators()
        window_data = features.iloc[self.current_step-self.window_size+1:self.current_step+1].values.flatten()
        
        if self.use_position_features:
            # Add portfolio state features
            portfolio_value = self._calculate_portfolio_value()
            
            # Ensure non-zero values to prevent division by zero
            safe_portfolio_value = max(portfolio_value, 1e-8)
            safe_initial_balance = max(self.initial_balance, 1e-8)
            
            position_features = np.array([
                self.balance / safe_initial_balance,  # Normalized balance
                self.asset_held * self.current_price / safe_portfolio_value if portfolio_value > 0 else 0  # Position size
            ])
            observation = np.concatenate([window_data, position_features])
        else:
            observation = window_data
            
        # Clip extreme values to prevent numerical issues
        observation = np.clip(observation, -10.0, 10.0)
        
        # Replace any NaN or infinite values
        observation = np.nan_to_num(observation, nan=0.0, posinf=10.0, neginf=-10.0)
            
        return observation.astype(np.float32)
    
    def _calculate_portfolio_value(self) -> float:
        """Calculate the total portfolio value"""
        return self.balance + (self.asset_held * self.current_price)
    
    def render(self, mode='human'):
        """Render the environment visualization"""
        portfolio_value = self._calculate_portfolio_value()
        print(f"Step: {self.current_step}, Portfolio Value: ${portfolio_value:.2f}")
        print(f"Balance: ${self.balance:.2f}, Asset Held: {self.asset_held:.6f}")
        print(f"Current Price: ${self.current_price:.2f}")
        print(f"Total Trades: {self.total_trades}, Total Fee Paid: ${self.total_fee_paid:.2f}")
        print("-------------------------------------------")
    
    def get_trade_history(self) -> pd.DataFrame:
        """Return the trade history as a DataFrame"""
        if not self.trade_history:
            return pd.DataFrame()
        return pd.DataFrame(self.trade_history).set_index('timestamp')
        
    def close(self):
        """Clean up resources"""
        pass

# Wrapper environment that provides a standard interface
class MultiAssetTradingEnv(gym.Env):
    """
    A wrapper for handling multiple trading environments for different assets.
    """
    def __init__(
        self, 
        dfs: Dict[str, pd.DataFrame],
        window_size: int = 24,
        initial_balance: float = 10000.0,
        transaction_fee: float = 0.001,
        reward_scaling: float = 1.0,
        active_asset: Optional[str] = None
    ):
        """
        Initialize multiple trading environments, one per asset.
        
        Args:
            dfs: Dictionary mapping asset symbols to their OHLCV DataFrames
            window_size: Number of previous candles to include in the observation
            initial_balance: Starting balance in USD per asset
            transaction_fee: Fee per transaction as a fraction
            reward_scaling: Scale factor for rewards
            active_asset: If provided, only this asset's environment will be active
        """
        super(MultiAssetTradingEnv, self).__init__()
        
        self.envs = {}
        for symbol, df in dfs.items():
            self.envs[symbol] = TradingEnv(
                df=df,
                window_size=window_size,
                initial_balance=initial_balance,
                transaction_fee=transaction_fee,
                reward_scaling=reward_scaling
            )
        
        # Set active asset if provided, otherwise use the first one
        self.active_asset = active_asset or list(self.envs.keys())[0]
        
        # Use the observation and action space from the active environment
        self.observation_space = self.envs[self.active_asset].observation_space
        self.action_space = self.envs[self.active_asset].action_space
    
    def reset(self):
        """Reset all environments and return observation from the active one"""
        observations = {}
        for symbol, env in self.envs.items():
            observations[symbol] = env.reset()
        return observations[self.active_asset]
    
    def step(self, action):
        """Take a step in the active environment"""
        return self.envs[self.active_asset].step(action)
    
    def set_active_asset(self, symbol: str):
        """Change the active asset"""
        if symbol not in self.envs:
            raise ValueError(f"Asset {symbol} not found in available environments")
        self.active_asset = symbol
        # Update observation and action spaces
        self.observation_space = self.envs[self.active_asset].observation_space
        self.action_space = self.envs[self.active_asset].action_space
    
    def render(self, mode='human'):
        """Render the active environment"""
        print(f"Active Asset: {self.active_asset}")
        self.envs[self.active_asset].render(mode)
    
    def close(self):
        """Close all environments"""
        for env in self.envs.values():
            env.close() 