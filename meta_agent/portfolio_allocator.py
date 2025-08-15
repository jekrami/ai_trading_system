import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import joblib
from scipy.optimize import minimize
import cvxpy as cp

# Set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PortfolioAllocator:
    """
    Meta-agent that allocates capital across individual asset agents 
    based on their signals and historical performance.
    """
    
    def __init__(
        self, 
        mode: str = "sharpe", 
        risk_aversion: float = 1.0,
        max_allocation: float = 0.5,
        min_allocation: float = 0.0,
        method: str = "risk_parity"
    ):
        """
        Initialize the portfolio allocator.
        
        Args:
            mode: Optimization objective - 'sharpe', 'returns', 'min_variance', 'sortino'
            risk_aversion: Risk aversion parameter (lambda)
            max_allocation: Maximum allocation to a single asset (0.5 = 50%)
            min_allocation: Minimum allocation to a single asset if included
            method: Allocation method - 'risk_parity', 'markowitz', 'equal', 'kelly'
        """
        self.mode = mode
        self.risk_aversion = risk_aversion
        self.max_allocation = max_allocation
        self.min_allocation = min_allocation
        self.method = method
        self.historical_allocations = []
        self.historical_returns = []
        
    def calculate_weights_risk_parity(self, returns: pd.DataFrame) -> np.ndarray:
        """Calculate asset weights using risk parity approach"""
        # Calculate covariance matrix
        cov_matrix = returns.cov().values
        n = len(returns.columns)
        
        # Initialize weights equally
        init_weights = np.ones(n) / n
        
        # Define risk contribution function
        def risk_contribution(weights, cov_matrix):
            portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)
            marginal_contrib = (cov_matrix @ weights) / portfolio_vol
            risk_contrib = weights * marginal_contrib
            return risk_contrib
        
        # Define optimization objective: variance of risk contributions
        def objective(weights, cov_matrix):
            weights = np.array(weights)
            risk_contribs = risk_contribution(weights, cov_matrix)
            target_risk_contrib = np.ones(n) * (1/n)
            return np.sum((risk_contribs - target_risk_contrib) ** 2)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Weights sum to 1
        ]
        
        # Bounds: each weight between min and max allocation
        bounds = [(self.min_allocation, self.max_allocation) for _ in range(n)]
        
        # Optimize
        result = minimize(
            objective, 
            init_weights, 
            args=(cov_matrix,), 
            method='SLSQP', 
            bounds=bounds,
            constraints=constraints,
            options={'disp': False}
        )
        
        # Normalize weights to ensure they sum to 1
        weights = result['x']
        weights = weights / np.sum(weights)
        
        return weights
    
    def calculate_weights_markowitz(self, returns: pd.DataFrame) -> np.ndarray:
        """Calculate asset weights using Markowitz mean-variance optimization"""
        # Calculate expected returns and covariance
        expected_returns = returns.mean().values
        cov_matrix = returns.cov().values
        n = len(returns.columns)
        
        # Define variables for optimization
        weights = cp.Variable(n)
        
        # Expected portfolio return
        expected_portfolio_return = expected_returns @ weights
        
        # Portfolio variance
        portfolio_variance = cp.quad_form(weights, cov_matrix)
        
        # Objective: maximize expected return - lambda * variance
        objective = cp.Maximize(expected_portfolio_return - self.risk_aversion * portfolio_variance)
        
        # Constraints
        constraints = [
            cp.sum(weights) == 1,  # Fully invested
            weights >= self.min_allocation,  # Minimum allocation
            weights <= self.max_allocation   # Maximum allocation
        ]
        
        # Solve problem
        problem = cp.Problem(objective, constraints)
        try:
            problem.solve()
            
            if problem.status == 'optimal':
                return weights.value
            else:
                logger.warning(f"Optimization problem status: {problem.status}")
                # Fall back to equal weights
                return np.ones(n) / n
        except Exception as e:
            logger.error(f"Error in Markowitz optimization: {str(e)}")
            # Fall back to equal weights
            return np.ones(n) / n
    
    def calculate_weights_equal(self, returns: pd.DataFrame) -> np.ndarray:
        """Calculate equal weights for all assets"""
        n = len(returns.columns)
        return np.ones(n) / n
    
    def calculate_weights_kelly(self, returns: pd.DataFrame) -> np.ndarray:
        """Calculate weights using Kelly criterion"""
        # For Kelly, we need expected returns and covariance
        expected_returns = returns.mean().values
        cov_matrix = returns.cov().values
        inv_cov_matrix = np.linalg.inv(cov_matrix)
        
        # Kelly formula: weights = inv(cov) * expected_returns
        kelly_weights = inv_cov_matrix @ expected_returns
        
        # Normalize to sum to 1
        kelly_weights = kelly_weights / np.sum(kelly_weights)
        
        # Apply bounds
        n = len(returns.columns)
        kelly_weights = np.clip(kelly_weights, self.min_allocation, self.max_allocation)
        
        # Re-normalize after clipping
        if np.sum(kelly_weights) > 0:
            kelly_weights = kelly_weights / np.sum(kelly_weights)
        else:
            # Fall back to equal weights if all weights are 0 after clipping
            kelly_weights = np.ones(n) / n
        
        return kelly_weights
    
    def allocate_portfolio(
        self, 
        agent_predictions: Dict[str, float], 
        returns_history: Dict[str, pd.Series],
        allocation_date: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Allocate portfolio based on agent predictions and historical returns.
        
        Args:
            agent_predictions: Dictionary mapping asset symbols to prediction scores
            returns_history: Dictionary mapping asset symbols to historical return series
            allocation_date: Date string for the allocation
            
        Returns:
            Dictionary mapping asset symbols to allocation weights
        """
        if not agent_predictions:
            logger.warning("No agent predictions provided for allocation")
            return {}
        
        if not returns_history:
            logger.warning("No historical returns provided for allocation")
            return {symbol: 1.0 / len(agent_predictions) for symbol in agent_predictions}
        
        # Convert returns to a DataFrame
        returns_df = pd.DataFrame({symbol: returns for symbol, returns in returns_history.items()})
        
        # Filter to only include assets with predictions
        symbols = list(agent_predictions.keys())
        returns_df = returns_df[symbols].dropna()
        
        # Calculate prediction-weighted expected returns
        # (We're using the model's prediction score to adjust expected returns)
        expected_returns = returns_df.mean()
        for symbol in symbols:
            if symbol in agent_predictions:
                # Adjust expected returns based on model predictions
                expected_returns[symbol] *= max(0, agent_predictions[symbol])
        
        # Get appropriate allocation method
        if self.method == "risk_parity":
            weights = self.calculate_weights_risk_parity(returns_df)
        elif self.method == "markowitz":
            weights = self.calculate_weights_markowitz(returns_df)
        elif self.method == "equal":
            weights = self.calculate_weights_equal(returns_df)
        elif self.method == "kelly":
            weights = self.calculate_weights_kelly(returns_df)
        else:
            logger.warning(f"Unknown allocation method: {self.method}. Using equal weights.")
            weights = self.calculate_weights_equal(returns_df)
        
        # Convert weights to dictionary
        allocation = {symbol: float(weight) for symbol, weight in zip(returns_df.columns, weights)}
        
        # Record allocation for history
        self.historical_allocations.append({
            "date": allocation_date or datetime.now().isoformat(),
            "allocation": allocation
        })
        
        return allocation
    
    def update_returns(self, realized_returns: Dict[str, float], date: Optional[str] = None):
        """
        Update historical returns with realized returns.
        
        Args:
            realized_returns: Dictionary mapping asset symbols to realized returns
            date: Date string for the returns
        """
        self.historical_returns.append({
            "date": date or datetime.now().isoformat(),
            "returns": realized_returns
        })
    
    def calculate_portfolio_metrics(self) -> Dict[str, float]:
        """Calculate portfolio performance metrics"""
        if not self.historical_allocations or not self.historical_returns:
            return {}
        
        # Match allocations with subsequent returns
        portfolio_returns = []
        
        for i, alloc in enumerate(self.historical_allocations[:-1]):  # Skip the most recent allocation
            alloc_date = alloc["date"]
            alloc_weights = alloc["allocation"]
            
            # Find the next return entry after this allocation
            next_returns = None
            for ret in self.historical_returns:
                if ret["date"] > alloc_date:
                    next_returns = ret["returns"]
                    break
                    
            if next_returns:
                # Calculate weighted portfolio return
                port_return = 0
                for symbol, weight in alloc_weights.items():
                    if symbol in next_returns:
                        port_return += weight * next_returns[symbol]
                portfolio_returns.append(port_return)
        
        if not portfolio_returns:
            return {}
            
        # Calculate metrics
        returns_series = pd.Series(portfolio_returns)
        metrics = {
            "cumulative_return": (1 + returns_series).prod() - 1,
            "mean_return": returns_series.mean(),
            "volatility": returns_series.std(),
            "sharpe_ratio": returns_series.mean() / (returns_series.std() + 1e-10),
            "max_drawdown": (returns_series.cumsum() - returns_series.cumsum().cummax()).min(),
            "win_rate": (returns_series > 0).mean()
        }
        
        # Add Sortino ratio
        negative_returns = returns_series[returns_series < 0]
        if len(negative_returns) > 0:
            metrics["sortino_ratio"] = returns_series.mean() / (negative_returns.std() + 1e-10)
        else:
            metrics["sortino_ratio"] = float('inf')
            
        return metrics
    
    def save(self, filepath: str):
        """Save allocator state to a file"""
        state = {
            "mode": self.mode,
            "risk_aversion": self.risk_aversion,
            "max_allocation": self.max_allocation,
            "min_allocation": self.min_allocation,
            "method": self.method,
            "historical_allocations": self.historical_allocations,
            "historical_returns": self.historical_returns
        }
        
        with open(filepath, "w") as f:
            json.dump(state, f, indent=4)
            
        logger.info(f"Portfolio allocator state saved to {filepath}")
    
    def load(self, filepath: str):
        """Load allocator state from a file"""
        if not os.path.exists(filepath):
            logger.error(f"File not found: {filepath}")
            return False
            
        try:
            with open(filepath, "r") as f:
                state = json.load(f)
                
            self.mode = state.get("mode", self.mode)
            self.risk_aversion = state.get("risk_aversion", self.risk_aversion)
            self.max_allocation = state.get("max_allocation", self.max_allocation)
            self.min_allocation = state.get("min_allocation", self.min_allocation)
            self.method = state.get("method", self.method)
            self.historical_allocations = state.get("historical_allocations", [])
            self.historical_returns = state.get("historical_returns", [])
            
            logger.info(f"Portfolio allocator state loaded from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error loading allocator state: {str(e)}")
            return False

class AIPortfolioOptimizer:
    """
    Advanced portfolio optimizer that uses machine learning to 
    determine optimal allocations based on market conditions.
    """
    
    def __init__(
        self,
        lookback_window: int = 30,
        max_allocation: float = 0.5,
        min_allocation: float = 0.0,
        model_path: Optional[str] = None
    ):
        """
        Initialize the AI portfolio optimizer.
        
        Args:
            lookback_window: Number of days to look back for features
            max_allocation: Maximum allocation to a single asset
            min_allocation: Minimum allocation to included assets
            model_path: Path to pre-trained model file
        """
        self.lookback_window = lookback_window
        self.max_allocation = max_allocation
        self.min_allocation = min_allocation
        self.model = None
        
        if model_path and os.path.exists(model_path):
            try:
                self.model = joblib.load(model_path)
                logger.info(f"Loaded AI allocation model from {model_path}")
            except Exception as e:
                logger.error(f"Error loading model: {str(e)}")
    
    def prepare_features(
        self,
        asset_returns: Dict[str, pd.Series],
        market_features: Optional[pd.DataFrame] = None,
        agent_signals: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """
        Prepare features for the AI allocation model.
        
        Args:
            asset_returns: Dictionary mapping asset symbols to return series
            market_features: Optional DataFrame with market-wide features
            agent_signals: Optional dictionary with agent prediction signals
            
        Returns:
            DataFrame with features for allocation prediction
        """
        features = pd.DataFrame()
        
        # Process asset returns
        for symbol, returns in asset_returns.items():
            # Basic return features
            features[f"{symbol}_mean_return"] = returns.rolling(self.lookback_window).mean().iloc[-1:]
            features[f"{symbol}_volatility"] = returns.rolling(self.lookback_window).std().iloc[-1:]
            features[f"{symbol}_sharpe"] = (
                returns.rolling(self.lookback_window).mean() / 
                returns.rolling(self.lookback_window).std()
            ).iloc[-1:]
            
            # Trend features
            features[f"{symbol}_trend"] = returns.rolling(self.lookback_window).apply(
                lambda x: np.polyfit(np.arange(len(x)), x, 1)[0]
            ).iloc[-1:]
            
        # Add market features if provided
        if market_features is not None:
            for col in market_features.columns:
                features[f"market_{col}"] = market_features[col].iloc[-1:]
                
        # Add agent signals if provided
        if agent_signals is not None:
            for symbol, signal in agent_signals.items():
                features[f"{symbol}_signal"] = signal
                
        return features
    
    def predict_allocations(
        self,
        asset_returns: Dict[str, pd.Series],
        market_features: Optional[pd.DataFrame] = None,
        agent_signals: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Predict optimal allocations using the AI model.
        
        Args:
            asset_returns: Dictionary mapping asset symbols to return series
            market_features: Optional DataFrame with market-wide features
            agent_signals: Optional dictionary with agent prediction signals
            
        Returns:
            Dictionary mapping asset symbols to allocation weights
        """
        if self.model is None:
            logger.warning("No AI model loaded, falling back to heuristic allocation")
            # Fall back to simple allocation based on Sharpe ratios
            weights = self._allocate_by_sharpe(asset_returns)
            return weights
            
        # Prepare features for prediction
        features = self.prepare_features(asset_returns, market_features, agent_signals)
        
        try:
            # Predict raw allocations
            raw_allocations = self.model.predict(features)[0]
            
            # Convert to dictionary and apply constraints
            symbols = list(asset_returns.keys())
            allocations = {symbol: float(alloc) for symbol, alloc in zip(symbols, raw_allocations)}
            
            # Apply min/max constraints
            allocations = {symbol: min(max(alloc, self.min_allocation), self.max_allocation) 
                          for symbol, alloc in allocations.items()}
            
            # Normalize to sum to 1
            total_allocation = sum(allocations.values())
            if total_allocation > 0:
                allocations = {symbol: alloc / total_allocation 
                             for symbol, alloc in allocations.items()}
            else:
                # Fall back to equal weights if total is 0 or negative
                allocations = {symbol: 1.0 / len(symbols) for symbol in symbols}
                
            return allocations
            
        except Exception as e:
            logger.error(f"Error predicting allocations: {str(e)}")
            # Fall back to simple allocation
            return self._allocate_by_sharpe(asset_returns)
    
    def _allocate_by_sharpe(self, asset_returns: Dict[str, pd.Series]) -> Dict[str, float]:
        """Fall back method to allocate based on Sharpe ratios"""
        sharpes = {}
        
        for symbol, returns in asset_returns.items():
            mean_return = returns.mean()
            std_return = returns.std()
            if std_return > 0:
                sharpes[symbol] = mean_return / std_return
            else:
                sharpes[symbol] = 0
                
        # Avoid negative allocations by shifting values
        min_sharpe = min(sharpes.values())
        if min_sharpe < 0:
            # Shift all values to be positive
            for symbol in sharpes:
                sharpes[symbol] -= min_sharpe - 0.01
                
        # Normalize to sum to 1
        total_sharpe = sum(sharpes.values())
        if total_sharpe > 0:
            allocations = {symbol: min(sharpe / total_sharpe, self.max_allocation) 
                         for symbol, sharpe in sharpes.items()}
        else:
            # Equal weights if all sharpes are 0
            allocations = {symbol: 1.0 / len(asset_returns) for symbol in asset_returns}
            
        # Renormalize after applying max constraint
        total_allocation = sum(allocations.values())
        allocations = {symbol: alloc / total_allocation for symbol, alloc in allocations.items()}
        
        return allocations

if __name__ == "__main__":
    # Example usage
    
    # Simulate some returns data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    
    returns_history = {
        'BTC': pd.Series(np.random.normal(0.001, 0.05, size=100), index=dates),
        'ETH': pd.Series(np.random.normal(0.002, 0.06, size=100), index=dates),
        'SOL': pd.Series(np.random.normal(0.003, 0.08, size=100), index=dates),
        'ADA': pd.Series(np.random.normal(0.0005, 0.04, size=100), index=dates)
    }
    
    # Simulate agent predictions
    agent_predictions = {
        'BTC': 0.65,
        'ETH': 0.72,
        'SOL': 0.58,
        'ADA': 0.48
    }
    
    # Create allocator
    allocator = PortfolioAllocator(
        mode='sharpe',
        risk_aversion=1.2,
        max_allocation=0.4,
        min_allocation=0.05,
        method='risk_parity'
    )
    
    # Allocate portfolio
    allocation = allocator.allocate_portfolio(
        agent_predictions=agent_predictions,
        returns_history=returns_history,
        allocation_date='2023-04-11'
    )
    
    print(f"Portfolio allocation: {allocation}")
    
    # Simulate the next day's returns
    next_returns = {
        'BTC': 0.03,
        'ETH': -0.01,
        'SOL': 0.05,
        'ADA': 0.02
    }
    
    # Update with realized returns
    allocator.update_returns(next_returns, date='2023-04-12')
    
    # Save allocator state
    os.makedirs('models', exist_ok=True)
    allocator.save('models/portfolio_allocator.json') 