import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import our components
from agents.ppo_agent import PPOTradingAgent
from envs.trading_env import TradingEnv

# Set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BacktestEngine:
    """
    Engine for backtesting trading strategies on historical data.
    Supports traditional strategies as well as RL agent evaluation.
    """
    
    def __init__(
        self,
        data: Dict[str, pd.DataFrame],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        initial_capital: float = 10000.0,
        transaction_fee: float = 0.001
    ):
        """
        Initialize the backtesting engine.
        
        Args:
            data: Dictionary mapping asset symbols to DataFrames with OHLCV data
            start_date: Start date for backtesting
            end_date: End date for backtesting
            initial_capital: Initial capital per asset
            transaction_fee: Transaction fee rate
        """
        self.data = data
        self.initial_capital = initial_capital
        self.transaction_fee = transaction_fee
        
        # Filter data by date range if provided
        if start_date or end_date:
            filtered_data = {}
            for symbol, df in data.items():
                filtered_df = df.copy()
                if start_date:
                    filtered_df = filtered_df[filtered_df.index >= start_date]
                if end_date:
                    filtered_df = filtered_df[filtered_df.index <= end_date]
                filtered_data[symbol] = filtered_df
            self.data = filtered_data
    
    def backtest_strategy(
        self, 
        strategy_fn: Callable, 
        strategy_params: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Backtest a trading strategy function.
        
        Args:
            strategy_fn: Strategy function that takes (df, params) and returns signals DataFrame
            strategy_params: Parameters to pass to the strategy function
            
        Returns:
            Dictionary with backtest results
        """
        if strategy_params is None:
            strategy_params = {}
            
        results = {}
        
        # Run strategy on each asset
        for symbol, df in self.data.items():
            logger.info(f"Backtesting {symbol} with strategy...")
            
            # Generate trading signals (1 for buy, -1 for sell, 0 for hold)
            signals = strategy_fn(df, strategy_params)
            
            # Calculate returns and statistics
            asset_results = self._calculate_returns(symbol, df, signals)
            results[symbol] = asset_results
        
        # Calculate portfolio-level statistics
        portfolio_results = self._aggregate_results(results)
        results["portfolio"] = portfolio_results
        
        return results
    
    def backtest_rl_agent(
        self, 
        agent_paths: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Backtest RL agents on historical data.
        
        Args:
            agent_paths: Dictionary mapping asset symbols to agent model paths
            
        Returns:
            Dictionary with backtest results
        """
        results = {}
        
        # Run backtest for each agent
        for symbol, model_path in agent_paths.items():
            if symbol not in self.data:
                logger.warning(f"No data found for {symbol}, skipping")
                continue
                
            logger.info(f"Backtesting {symbol} with RL agent...")
            
            try:
                # Create environment
                env = TradingEnv(
                    df=self.data[symbol],
                    initial_balance=self.initial_capital,
                    transaction_fee=self.transaction_fee
                )
                
                # Load agent
                agent = PPOTradingAgent(symbol=symbol, env=env)
                agent.load(model_path)
                
                # Run backtest
                backtest_result = agent.backtest(env)
                results[symbol] = backtest_result["results"]
                
            except Exception as e:
                logger.error(f"Error backtesting {symbol}: {str(e)}")
                
        # Calculate portfolio-level statistics
        if results:
            portfolio_results = self._aggregate_results(results)
            results["portfolio"] = portfolio_results
        
        return results
    
    def _calculate_returns(
        self, 
        symbol: str, 
        df: pd.DataFrame, 
        signals: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Calculate returns and statistics for a single asset.
        
        Args:
            symbol: Asset symbol
            df: DataFrame with OHLCV data
            signals: DataFrame with trading signals
            
        Returns:
            Dictionary with asset-specific results
        """
        # Initialize variables
        capital = self.initial_capital
        position = 0
        trade_history = []
        equity_curve = []
        
        # Add indicators to dataframe
        data = pd.concat([df, signals], axis=1)
        
        # Simulate trading
        for i in range(1, len(data)):
            # Get current state
            current_price = data['close'].iloc[i]
            current_date = data.index[i]
            current_signal = data['signal'].iloc[i]
            prev_signal = data['signal'].iloc[i-1]
            
            # Check for signal change and execute trades
            if current_signal != prev_signal:
                if current_signal > 0 and position == 0:  # Buy signal
                    # Calculate max asset we can buy
                    max_asset = capital / (current_price * (1 + self.transaction_fee))
                    position = max_asset
                    fee = current_price * position * self.transaction_fee
                    capital -= (current_price * position + fee)
                    
                    # Record trade
                    trade_history.append({
                        'date': current_date,
                        'action': 'buy',
                        'price': current_price,
                        'amount': position,
                        'fee': fee
                    })
                    
                elif current_signal <= 0 and position > 0:  # Sell signal
                    # Sell all position
                    proceeds = position * current_price
                    fee = proceeds * self.transaction_fee
                    capital += (proceeds - fee)
                    
                    # Record trade
                    trade_history.append({
                        'date': current_date,
                        'action': 'sell',
                        'price': current_price,
                        'amount': position,
                        'fee': fee
                    })
                    
                    position = 0
            
            # Calculate current equity
            equity = capital + (position * current_price)
            equity_curve.append({
                'date': current_date,
                'equity': equity,
                'price': current_price,
                'position': position,
                'capital': capital
            })
        
        # Create equity curve DataFrame
        equity_df = pd.DataFrame(equity_curve)
        if not equity_df.empty:
            equity_df = equity_df.set_index('date')
            
            # Calculate returns
            equity_df['returns'] = equity_df['equity'].pct_change()
            
            # Calculate statistics
            total_return = (equity_df['equity'].iloc[-1] / self.initial_capital) - 1
            sharpe = 0
            if len(equity_df) > 1:
                sharpe = equity_df['returns'].mean() / (equity_df['returns'].std() + 1e-10) * np.sqrt(252)
                
            # Calculate drawdowns
            equity_df['cummax'] = equity_df['equity'].cummax()
            equity_df['drawdown'] = (equity_df['equity'] - equity_df['cummax']) / equity_df['cummax']
            max_drawdown = equity_df['drawdown'].min()
            
            # Count trades
            num_trades = len(trade_history)
            
            # Calculate win rate
            if num_trades > 0:
                wins = 0
                for i in range(0, len(trade_history), 2):
                    if i+1 < len(trade_history):
                        buy = trade_history[i]
                        sell = trade_history[i+1]
                        if sell['price'] > buy['price']:
                            wins += 1
                win_rate = wins / (num_trades // 2) if num_trades > 1 else 0
            else:
                win_rate = 0
                
            # Prepare results
            results = {
                'symbol': symbol,
                'initial_capital': self.initial_capital,
                'final_equity': equity_df['equity'].iloc[-1],
                'total_return': total_return,
                'annual_return': total_return / (len(equity_df)/252),
                'sharpe_ratio': sharpe,
                'max_drawdown': max_drawdown,
                'num_trades': num_trades,
                'win_rate': win_rate,
                'equity_curve': equity_df,
                'trade_history': trade_history
            }
            
            return results
        else:
            return {
                'symbol': symbol,
                'error': 'No trades executed'
            }
    
    def _aggregate_results(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate individual asset results into portfolio-level statistics.
        
        Args:
            results: Dictionary with individual asset results
            
        Returns:
            Dictionary with portfolio-level statistics
        """
        # Extract key metrics
        symbols = [k for k in results.keys() if k != 'portfolio']
        
        # If no valid results, return empty dict
        if not symbols:
            return {}
            
        # Calculate portfolio statistics
        initial_capital = sum(results[s].get('initial_capital', 0) for s in symbols)
        final_equity = sum(results[s].get('final_equity', 0) for s in symbols)
        
        # Total return
        if initial_capital > 0:
            total_return = (final_equity / initial_capital) - 1
        else:
            total_return = 0
            
        # Get equity curves
        equity_curves = []
        for symbol in symbols:
            if 'equity_curve' in results[symbol]:
                eq_curve = results[symbol]['equity_curve']['equity'].copy()
                eq_curve.name = symbol
                equity_curves.append(eq_curve)
                
        # Create combined equity curve if available
        portfolio_equity = None
        if equity_curves:
            combined_df = pd.concat(equity_curves, axis=1)
            portfolio_equity = combined_df.sum(axis=1)
            portfolio_returns = portfolio_equity.pct_change().dropna()
            
            # Calculate portfolio Sharpe
            sharpe = portfolio_returns.mean() / (portfolio_returns.std() + 1e-10) * np.sqrt(252)
            
            # Calculate portfolio drawdown
            cummax = portfolio_equity.cummax()
            drawdown = (portfolio_equity - cummax) / cummax
            max_drawdown = drawdown.min()
        else:
            sharpe = 0
            max_drawdown = 0
            portfolio_returns = pd.Series()
            
        # Count total trades
        total_trades = sum(results[s].get('num_trades', 0) for s in symbols)
        
        # Calculate aggregate win rate
        if total_trades > 0:
            weighted_win_rate = sum(
                results[s].get('win_rate', 0) * results[s].get('num_trades', 0) 
                for s in symbols
            ) / total_trades
        else:
            weighted_win_rate = 0
            
        # Portfolio metrics
        portfolio_results = {
            'initial_capital': initial_capital,
            'final_equity': final_equity,
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'total_trades': total_trades,
            'win_rate': weighted_win_rate
        }
        
        # Add equity curve if available
        if portfolio_equity is not None:
            portfolio_results['equity_curve'] = portfolio_equity
            
        return portfolio_results
    
    def plot_results(self, results: Dict[str, Any], save_path: Optional[str] = None):
        """
        Plot backtest results.
        
        Args:
            results: Dictionary with backtest results
            save_path: If provided, save the plot to this path
        """
        if 'portfolio' not in results or 'equity_curve' not in results['portfolio']:
            logger.warning("Insufficient data for plotting")
            return
            
        # Create figure
        fig, axes = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot portfolio equity
        portfolio_equity = results['portfolio']['equity_curve']
        portfolio_equity.plot(ax=axes[0], label='Portfolio Equity', color='blue', linewidth=2)
        axes[0].set_title('Backtest Results')
        axes[0].set_ylabel('Equity')
        axes[0].grid(True)
        axes[0].legend()
        
        # Plot individual asset equities if available
        symbols = [k for k in results.keys() if k != 'portfolio']
        for symbol in symbols:
            if 'equity_curve' in results[symbol]:
                asset_equity = results[symbol]['equity_curve']['equity']
                asset_equity.plot(ax=axes[0], label=f'{symbol} Equity', alpha=0.5)
        
        # Plot drawdowns
        if isinstance(portfolio_equity, pd.Series):
            cummax = portfolio_equity.cummax()
            drawdown = (portfolio_equity - cummax) / cummax
            drawdown.plot(ax=axes[1], color='red', label='Drawdown')
            axes[1].set_title('Portfolio Drawdown')
            axes[1].set_ylabel('Drawdown %')
            axes[1].set_ylim(bottom=drawdown.min() * 1.1, top=0.01)
            axes[1].grid(True)
            
        # Add metrics as text
        metrics_text = (
            f"Total Return: {results['portfolio']['total_return']:.2%}\n"
            f"Sharpe Ratio: {results['portfolio']['sharpe_ratio']:.2f}\n"
            f"Max Drawdown: {results['portfolio']['max_drawdown']:.2%}\n"
            f"# Trades: {results['portfolio']['total_trades']}\n"
            f"Win Rate: {results['portfolio']['win_rate']:.2%}"
        )
        axes[0].text(
            0.02, 0.05, metrics_text, 
            transform=axes[0].transAxes, 
            bbox=dict(facecolor='white', alpha=0.7)
        )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Plot saved to {save_path}")
        else:
            plt.show()

# Strategy functions for backtesting

def sma_crossover_strategy(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """
    Simple Moving Average crossover strategy.
    
    Args:
        df: DataFrame with OHLCV data
        params: Dictionary with parameters:
            - short_window: Short SMA window
            - long_window: Long SMA window
            
    Returns:
        DataFrame with signal column
    """
    short_window = params.get('short_window', 10)
    long_window = params.get('long_window', 50)
    
    # Calculate moving averages
    signals = pd.DataFrame(index=df.index)
    signals['price'] = df['close']
    signals['short_ma'] = df['close'].rolling(window=short_window, min_periods=1).mean()
    signals['long_ma'] = df['close'].rolling(window=long_window, min_periods=1).mean()
    
    # Generate signals
    signals['signal'] = 0.0
    signals['signal'] = np.where(signals['short_ma'] > signals['long_ma'], 1.0, 0.0)
    
    return signals

def rsi_strategy(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """
    Relative Strength Index (RSI) strategy.
    
    Args:
        df: DataFrame with OHLCV data
        params: Dictionary with parameters:
            - rsi_window: RSI calculation window
            - overbought: Overbought threshold
            - oversold: Oversold threshold
            
    Returns:
        DataFrame with signal column
    """
    rsi_window = params.get('rsi_window', 14)
    overbought = params.get('overbought', 70)
    oversold = params.get('oversold', 30)
    
    # Calculate RSI
    signals = pd.DataFrame(index=df.index)
    signals['price'] = df['close']
    
    # Calculate RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_window).mean()
    
    # Calculate RS and RSI
    rs = gain / loss
    signals['rsi'] = 100 - (100 / (1 + rs))
    
    # Generate signals
    signals['signal'] = 0.0
    signals['signal'] = np.where(signals['rsi'] < oversold, 1.0, 0.0)
    signals['signal'] = np.where(signals['rsi'] > overbought, -1.0, signals['signal'])
    
    return signals

def macd_strategy(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """
    Moving Average Convergence Divergence (MACD) strategy.
    
    Args:
        df: DataFrame with OHLCV data
        params: Dictionary with parameters:
            - fast: Fast EMA window
            - slow: Slow EMA window
            - signal: Signal line window
            
    Returns:
        DataFrame with signal column
    """
    fast = params.get('fast', 12)
    slow = params.get('slow', 26)
    signal_window = params.get('signal', 9)
    
    # Calculate MACD
    signals = pd.DataFrame(index=df.index)
    signals['price'] = df['close']
    
    # Calculate MACD and signal line
    exp1 = df['close'].ewm(span=fast, adjust=False).mean()
    exp2 = df['close'].ewm(span=slow, adjust=False).mean()
    signals['macd'] = exp1 - exp2
    signals['signal_line'] = signals['macd'].ewm(span=signal_window, adjust=False).mean()
    signals['histogram'] = signals['macd'] - signals['signal_line']
    
    # Generate signals
    signals['signal'] = 0.0
    signals['signal'] = np.where(signals['macd'] > signals['signal_line'], 1.0, 0.0)
    
    return signals 