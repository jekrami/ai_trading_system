import os
import json
import logging
import requests
from typing import Dict, List, Any, Optional
import pandas as pd
from datetime import datetime

# Set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReasoningEngine:
    """
    Engine for generating explanations and reports using LLMs.
    Supports both local LLMs (via Ollama) and remote APIs.
    """
    
    def __init__(
        self,
        model_type: str = "template",  # "local", "remote", or "template"
        model_name: str = "llama2",  # For local: "llama2", "mistral", etc. For remote: provider's model name
        api_key: Optional[str] = None,
        api_endpoint: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024
    ):
        """
        Initialize the reasoning engine.
        
        Args:
            model_type: "local" for Ollama/llama.cpp, "remote" for API (OpenAI, etc.), or "template" for built-in templates
            model_name: Model name (depends on provider)
            api_key: API key for remote models
            api_endpoint: API endpoint for remote models
            temperature: Sampling temperature (higher = more creative)
            max_tokens: Maximum tokens in response
        """
        self.model_type = model_type
        self.model_name = model_name
        self.api_key = api_key
        self.api_endpoint = api_endpoint
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Default local endpoint (Ollama) - using chat API
        if model_type == "local" and not api_endpoint:
            self.api_endpoint = "http://localhost:11434/api/chat"  # Updated to use chat endpoint
    
    def generate_text(self, prompt: str) -> str:
        """
        Generate text based on the prompt using the configured LLM.
        
        Args:
            prompt: The prompt text
            
        Returns:
            Generated text
        """
        try:
            if self.model_type == "template":
                return self._generate_template_response(prompt)
            elif self.model_type == "local":
                return self._generate_local(prompt)
            else:
                return self._generate_remote(prompt)
        except Exception as e:
            logger.error(f"Error generating text: {str(e)}")
            return f"[ERROR: {str(e)}]"
    
    def _generate_local(self, prompt: str) -> str:
        """Generate text using local Ollama model with the chat API"""
        try:
            # Check if Ollama is running
            if not self.api_endpoint:
                logger.error("No API endpoint configured for local model")
                return "[ERROR: No API endpoint configured]"
                
            # Prepare request using the chat API format
            data = {
                "model": self.model_name,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "options": {
                    "temperature": self.temperature
                },
                "stream": False
            }
            
            # Make request
            response = requests.post(self.api_endpoint, json=data)
            
            if response.status_code == 405:  # Method Not Allowed
                logger.error(f"405 Method Not Allowed: Check that you're using the correct Ollama API endpoint format")
                return "[ERROR: 405 Method Not Allowed - Check Ollama API documentation]"
                
            # Raise for other HTTP errors
            response.raise_for_status()
            
            # Extract response from the chat format
            result = response.json()
            return result.get("message", {}).get("content", "[No response generated]")
            
        except requests.RequestException as e:
            if "Connection refused" in str(e):
                logger.error("Could not connect to Ollama. Is it running?")
                # Fall back to template response
                return self._generate_template_response(prompt)
            else:
                logger.error(f"API request error: {str(e)}")
                return f"[ERROR: {str(e)}]"
    
    def _generate_remote(self, prompt: str) -> str:
        """Generate text using remote API"""
        try:
            # Check if API key is provided
            if not self.api_key:
                logger.error("No API key configured for remote model")
                return "[ERROR: No API key configured]"
                
            # Check if endpoint is provided
            if not self.api_endpoint:
                logger.error("No API endpoint configured for remote model")
                return "[ERROR: No API endpoint configured]"
            
            # Prepare headers
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            # Prepare request
            data = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens
            }
            
            # Make request
            response = requests.post(self.api_endpoint, headers=headers, json=data)
            response.raise_for_status()
            
            # Extract response
            result = response.json()
            return result.get("choices", [{}])[0].get("message", {}).get("content", "[No response generated]")
            
        except Exception as e:
            logger.error(f"Error with remote API: {str(e)}")
            # Fall back to template response
            return self._generate_template_response(prompt)
    
    def _generate_template_response(self, prompt: str) -> str:
        """Generate a template response when LLM is unavailable"""
        logger.warning("Falling back to template response")
        
        if "market analysis" in prompt.lower():
            return """
# Market Analysis

## Current Market Conditions
The market has shown mixed signals over the past week with some volatility in key assets.

## Key Observations
- Bitcoin has seen a consolidation pattern
- Ethereum has outperformed major assets
- Overall market sentiment appears neutral to slightly bullish
- Volume indicators suggest increased interest in altcoins

## Conclusion
Monitor key support and resistance levels while maintaining a balanced portfolio.
"""
        elif "trade explanation" in prompt.lower():
            return """
# Trade Explanation

The system recommended this trade based on several factors:

1. Price action showing a break of recent resistance
2. Increasing volume suggesting stronger buyer interest
3. Positive momentum indicators (RSI, MACD)
4. Favorable risk-reward ratio at current levels

This aligns with the broader market trend which currently appears bullish for this asset.
"""
        elif "portfolio allocation" in prompt.lower():
            return """
# Portfolio Allocation Rationale

The current portfolio allocation is designed to:

- Maximize risk-adjusted returns (Sharpe ratio)
- Diversify across different asset types
- Overweight assets showing stronger momentum
- Maintain limited exposure to highly correlated assets

This allocation aims to balance growth potential with downside protection.
"""
        else:
            return """
# AI Trading System Report

The system has analyzed market data and generated recommendations based on 
technical indicators, historical patterns, and machine learning models.

Current signals show a mixed market with opportunities in specific assets.
Risk management remains crucial in the current environment.
"""
    
    def analyze_market_data(self, asset_data: Dict[str, pd.DataFrame]) -> str:
        """
        Generate market analysis based on asset data.
        
        Args:
            asset_data: Dictionary mapping symbols to DataFrames with OHLCV data
            
        Returns:
            Market analysis text
        """
        # Prepare summary of market data
        data_summary = []
        
        for symbol, df in asset_data.items():
            if len(df) < 2:
                continue
                
            # Calculate basic metrics
            latest_price = df['close'].iloc[-1]
            prev_price = df['close'].iloc[-2]
            pct_change = ((latest_price / prev_price) - 1) * 100
            
            # 7-day metrics
            days_7 = min(7, len(df) - 1)
            price_7d_ago = df['close'].iloc[-days_7-1]
            pct_change_7d = ((latest_price / price_7d_ago) - 1) * 100
            
            # Volume
            latest_volume = df['volume'].iloc[-1]
            avg_volume_7d = df['volume'].iloc[-days_7:].mean()
            volume_change = ((latest_volume / avg_volume_7d) - 1) * 100
            
            # Add to summary
            data_summary.append(f"- {symbol}: ${latest_price:.2f} ({pct_change:.2f}% 1d, {pct_change_7d:.2f}% 7d), Vol: {volume_change:.2f}% vs 7d avg")
        
        # Create prompt
        prompt = f"""
You are an AI trading advisor. Generate a market analysis report based on the following cryptocurrency data.
Focus on key trends, notable price movements, and overall market sentiment.

Current date: {datetime.now().strftime('%Y-%m-%d')}

Asset Summary:
{''.join(data_summary)}

Your analysis should include:
1. Overall market sentiment and trend
2. Notable performers and underperformers
3. Volume analysis and what it suggests
4. Key levels to watch

Format your response as a Markdown document with appropriate sections and bullet points.
"""
        
        # Generate analysis
        return self.generate_text(prompt)
    
    def explain_trade(self, symbol: str, action: str, asset_data: pd.DataFrame) -> str:
        """
        Generate explanation for a specific trade.
        
        Args:
            symbol: Asset symbol
            action: Trade action (buy, sell, hold)
            asset_data: DataFrame with OHLCV data for the asset
            
        Returns:
            Trade explanation text
        """
        # Calculate some basic metrics
        if len(asset_data) < 30:
            return f"Insufficient data to explain trade for {symbol}"
            
        # Get recent price action
        latest_price = asset_data['close'].iloc[-1]
        prev_day = asset_data['close'].iloc[-2]
        week_ago = asset_data['close'].iloc[-7]
        month_ago = asset_data['close'].iloc[-30]
        
        # Calculate changes
        day_change = ((latest_price / prev_day) - 1) * 100
        week_change = ((latest_price / week_ago) - 1) * 100
        month_change = ((latest_price / month_ago) - 1) * 100
        
        # Volume changes
        latest_vol = asset_data['volume'].iloc[-1]
        avg_vol_week = asset_data['volume'].iloc[-7:].mean()
        vol_change = ((latest_vol / avg_vol_week) - 1) * 100
        
        # Create prompt
        prompt = f"""
As an AI trading advisor, explain the reasoning behind a {action.upper()} recommendation for {symbol} based on these metrics:

- Current price: ${latest_price:.2f}
- 24h change: {day_change:.2f}%
- 7d change: {week_change:.2f}%
- 30d change: {month_change:.2f}%
- Volume vs 7d avg: {vol_change:.2f}%

Current date: {datetime.now().strftime('%Y-%m-%d')}

Explain:
1. What technical indicators likely triggered this {action} signal
2. Key support and resistance levels that may be relevant
3. Risk factors to consider
4. Historical patterns that might be repeating

Format as a concise Markdown report with clear sections.
"""
        
        # Generate explanation
        return self.generate_text(prompt)
    
    def explain_portfolio_allocation(
        self, 
        allocation: Dict[str, float],
        returns_history: Dict[str, pd.Series]
    ) -> str:
        """
        Generate explanation for portfolio allocation.
        
        Args:
            allocation: Dictionary mapping symbols to allocation weights
            returns_history: Dictionary mapping symbols to return series
            
        Returns:
            Portfolio allocation explanation text
        """
        # Prepare allocation summary
        allocation_summary = []
        for symbol, weight in sorted(allocation.items(), key=lambda x: x[1], reverse=True):
            allocation_summary.append(f"- {symbol}: {weight*100:.2f}%")
        
        # Calculate some metrics on returns
        return_metrics = []
        for symbol, returns in returns_history.items():
            if symbol not in allocation:
                continue
                
            # Calculate metrics if we have enough data
            if len(returns) >= 30:
                vol = returns.std() * np.sqrt(252)  # Annualized volatility
                sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
                max_dd = (returns.cumsum() - returns.cumsum().cummax()).min()
                
                return_metrics.append(f"- {symbol}: Vol: {vol:.2f}, Sharpe: {sharpe:.2f}, Max DD: {max_dd:.2f}")
        
        # Create prompt
        prompt = f"""
As an AI portfolio manager, explain the reasoning behind the following portfolio allocation:

Allocation:
{''.join(allocation_summary)}

Asset Metrics:
{''.join(return_metrics)}

Current date: {datetime.now().strftime('%Y-%m-%d')}

Your explanation should include:
1. The likely rationale for each weight allocation
2. Risk management considerations
3. Diversification benefits in this allocation
4. How this allocation might perform in different market scenarios

Format as a Markdown report with clear sections.
"""
        
        # Generate explanation
        return self.generate_text(prompt)
    
    def generate_trading_report(
        self,
        asset_data: Dict[str, pd.DataFrame],
        agent_predictions: Dict[str, str],
        portfolio_allocation: Dict[str, Any],
        backtest_results: Dict[str, Any]
    ) -> str:
        """
        Generate comprehensive trading report.
        
        Args:
            asset_data: Dictionary mapping symbols to DataFrames
            agent_predictions: Dictionary mapping symbols to prediction labels
            portfolio_allocation: Portfolio allocation information
            backtest_results: Backtest results
            
        Returns:
            Complete trading report text
        """
        # Format predictions
        prediction_summary = []
        for symbol, prediction in agent_predictions.items():
            prediction_summary.append(f"- {symbol}: {prediction}")
        
        # Format allocation
        allocation_summary = []
        if portfolio_allocation and "allocation" in portfolio_allocation:
            for symbol, weight in sorted(portfolio_allocation["allocation"].items(), key=lambda x: x[1], reverse=True):
                allocation_summary.append(f"- {symbol}: {float(weight)*100:.2f}%")
        
        # Format backtest results
        backtest_summary = []
        if backtest_results and "portfolio" in backtest_results:
            portfolio = backtest_results["portfolio"]
            backtest_summary.extend([
                f"- Total Return: {portfolio.get('total_return', 0)*100:.2f}%",
                f"- Sharpe Ratio: {portfolio.get('sharpe_ratio', 0):.2f}",
                f"- Max Drawdown: {portfolio.get('max_drawdown', 0)*100:.2f}%",
                f"- Total Trades: {portfolio.get('total_trades', 0)}"
            ])
        
        # Create prompt
        prompt = f"""
As an AI trading advisor, generate a comprehensive trading report based on the following information:

Current date: {datetime.now().strftime('%Y-%m-%d')}

Agent Predictions:
{''.join(prediction_summary)}

Portfolio Allocation:
{''.join(allocation_summary)}

Backtest Results:
{''.join(backtest_summary)}

Your report should include:
1. Executive summary of current market conditions
2. Analysis of agent predictions and why they make sense given market data
3. Evaluation of the portfolio allocation and its risk/reward balance
4. Interpretation of backtest results and what they suggest for future performance
5. Recommendations for adjustments or considerations going forward

Format as a professional Markdown report with clear sections, bullet points, and concise explanations.
"""
        
        # Generate report
        return self.generate_text(prompt)

if __name__ == "__main__":
    # Example usage
    engine = ReasoningEngine(model_type="local", model_name="llama2")
    
    # Generate sample report
    report = engine.generate_text("Generate a short market analysis for Bitcoin and Ethereum.")
    
    print(report) 