import gradio as gr
import requests
import time
import threading
import subprocess
from datetime import datetime

# Configuration
COIN_API_URL = "https://api.coingecko.com/api/v3/simple/price"
COINS = {"bitcoin": "BTC", "ethereum": "ETH"}
UPDATE_INTERVAL = 60  # seconds
TRIGGER_HISTORY = []

# Global variables for prices and triggers
current_prices = {"BTC": 0, "ETH": 0}
triggers = {
    "BTC": {"up": None, "down": None},
    "ETH": {"up": None, "down": None}
}

def fetch_prices():
    """Fetch current prices from CoinGecko API"""
    try:
        params = {
            'ids': ','.join(COINS.keys()),
            'vs_currencies': 'usd'
        }
        response = requests.get(COIN_API_URL, params=params)
        data = response.json()
        
        for coin_id, symbol in COINS.items():
            current_prices[symbol] = data[coin_id]['usd']
        return True
    except Exception as e:
        print(f"Error fetching prices: {e}")
        return False

def check_triggers():
    """Check if price triggers have been hit"""
    for coin in ["BTC", "ETH"]:
        current_price = current_prices[coin]
        
        # Check upward trigger
        if triggers[coin]["up"] and current_price >= triggers[coin]["up"]:
            execute_trigger(coin, "up", current_price)
            triggers[coin]["up"] = None  # Reset trigger after activation
        
        # Check downward trigger
        if triggers[coin]["down"] and current_price <= triggers[coin]["down"]:
            execute_trigger(coin, "down", current_price)
            triggers[coin]["down"] = None  # Reset trigger after activation

def execute_trigger(coin, direction, price):
    """Execute the external script and log the trigger"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message = f"{timestamp} - {coin} {direction} trigger activated at ${price:,.2f}"
    TRIGGER_HISTORY.append(message)
    
    # Execute the external script
    try:
        subprocess.run(["python", "goget.py", coin, direction, str(price)], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing goget.py: {e}")

def monitoring_thread():
    """Background thread for continuous monitoring"""
    while True:
        if fetch_prices():
            check_triggers()
        time.sleep(UPDATE_INTERVAL)

def set_trigger(coin, direction, price):
    """Set a new price trigger"""
    try:
        price_float = float(price)
        if direction == "up":
            triggers[coin]["up"] = price_float
        else:
            triggers[coin]["down"] = price_float
        return f"{coin} {direction} trigger set at ${price_float:,.2f}"
    except (ValueError, TypeError):
        return f"Error: Please enter a valid price"

def get_dashboard():
    """Generate dashboard content"""
    dashboard = "## Current Prices\n"
    dashboard += f"- BTC: ${current_prices['BTC']:,.2f}\n"
    dashboard += f"- ETH: ${current_prices['ETH']:,.2f}\n\n"
    
    dashboard += "## Active Triggers\n"
    for coin in ["BTC", "ETH"]:
        up_trigger = triggers[coin]["up"] or "Not set"
        down_trigger = triggers[coin]["down"] or "Not set"
        dashboard += f"- {coin} Up: ${up_trigger if isinstance(up_trigger, str) else f'{up_trigger:,.2f}'}\n"
        dashboard += f"- {coin} Down: ${down_trigger if isinstance(down_trigger, str) else f'{down_trigger:,.2f}'}\n\n"
    
    dashboard += "## Trigger History\n"
    dashboard += "\n".join(TRIGGER_HISTORY[-5:][::-1]) if TRIGGER_HISTORY else "No triggers activated yet"
    
    return dashboard

# Start monitoring thread
thread = threading.Thread(target=monitoring_thread, daemon=True)
thread.start()

# Gradio Interface
with gr.Blocks(title="Crypto Price Monitor") as app:
    gr.Markdown("# Cryptocurrency Price Monitor")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("## Set Price Triggers")
            coin_select = gr.Dropdown(["BTC", "ETH"], label="Select Coin")
            direction_select = gr.Dropdown(["up", "down"], label="Trigger Direction")
            price_input = gr.Number(label="Trigger Price (USD)")
            set_trigger_btn = gr.Button("Set Trigger")
            trigger_output = gr.Textbox(label="Trigger Status")            set_trigger_btn.click(
                fn=set_trigger,
                inputs=[coin_select, direction_select, price_input],
                outputs=[trigger_output, dashboard]
            )
        
        with gr.Column():
            gr.Markdown("## Dashboard")
            dashboard = gr.Markdown(value=get_dashboard())
            
            # Auto-refresh dashboard every 10 seconds
            app.add_periodic_callback(get_dashboard, 10, outputs=dashboard)

if __name__ == "__main__":
    # Initial price fetch
    fetch_prices()
    app.launch()