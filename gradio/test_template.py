import requests
import json

def process_price_change(coin, direction, price):
    """
    Template for handling price triggers.
    This is an example of what your goget.py could look like.
    """
    print(f"Price alert triggered!")
    print(f"Coin: {coin}")
    print(f"Direction: {direction}")
    print(f"Price: ${price:,.2f}")
    
    # Add your custom logic here
    # For example:
    # - Send notifications
    # - Execute trades
    # - Log to database
    # - etc.

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 4:
        print("Usage: python goget.py <coin> <direction> <price>")
        sys.exit(1)
    
    coin = sys.argv[1]
    direction = sys.argv[2]
    price = float(sys.argv[3])
    
    process_price_change(coin, direction, price)
