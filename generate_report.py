import os
import json
import logging
import pandas as pd
from datetime import datetime
from llm.reasoning_engine import ReasoningEngine

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Generate a trading report using template-based ReasoningEngine"""
    output_dir = "outputs"
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Create basic results for the template
        backtest_results = {
            "portfolio": {
                "total_return": 0.6821,  # Based on the console output
                "sharpe_ratio": 0.21,
                "max_drawdown": -0.4543,
                "total_trades": 2218
            }
        }
            
        # Create a reasoning engine with template mode
        engine = ReasoningEngine(
            model_type="local",  # Use local Ollama model
            model_name="llama3",  # Using llama3 instead of llama2
            temperature=0.7,
            max_tokens=2048
        )
        
        # Generate the report
        report = engine.generate_trading_report(
            asset_data={},  # No asset data needed for template
            agent_predictions={"BTC_USD": "bullish", "ETH_USD": "bullish"},
            portfolio_allocation={"allocation": {"BTC_USD": 0.6, "ETH_USD": 0.4}},
            backtest_results=backtest_results
        )
        
        # Save the report
        report_path = os.path.join(output_dir, "trading_report.md")
        with open(report_path, "w") as f:
            f.write(report)
            
        logger.info(f"Trading report saved to {report_path}")
        
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        print(f"Error generating report: {str(e)}")

if __name__ == "__main__":
    main() 