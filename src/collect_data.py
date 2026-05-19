import asyncio
import json
import csv
import os
import websockets
from datetime import datetime, UTC

SYMBOLS = [
    "btcusdt", "ethusdt", "solusdt", "bnbusdt", "xrpusdt",
    "adausdt", "dogeusdt", "avaxusdt", "linkusdt", "ltcusdt"
]

streams = "/".join([f"{s}@depth20@100ms/{s}@trade" for s in SYMBOLS])

URL = f"wss://data-stream.binance.vision/stream?streams={streams}"

MAX_ROWS = 1_000_000
OUT_FILE = "data/orderbook.csv"


async def collect():
    os.makedirs("data", exist_ok=True)

    file_exists = os.path.exists(OUT_FILE)
    count = 0

    if file_exists:
        with open(OUT_FILE, "r") as f:
            count = max(0, sum(1 for _ in f) - 1)

    latest_trade = {}

    with open(OUT_FILE, "a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp", "symbol", "mid", "spread",
                "best_bid", "best_ask", "bid_qty", "ask_qty",
                "top5_bid", "top5_ask", "imbalance",
                "microprice_dev", "trade_side", "trade_qty"
            ],
        )

        if not file_exists:
            writer.writeheader()

        while count < MAX_ROWS:
            try:
                async with websockets.connect(
                    URL,
                    ping_interval=20,
                    ping_timeout=60,
                    max_size=None,
                    close_timeout=5,
                ) as ws:
                    print(f"Connected. Current rows: {count}")

                    async for message in ws:
                        data = json.loads(message)
                        stream = data["stream"]
                        payload = data["data"]
                        symbol = stream.split("@")[0]

                        if "@trade" in stream:
                            latest_trade[symbol] = payload
                            continue

                        if "@depth20" in stream:
                            bids = payload["bids"]
                            asks = payload["asks"]

                            best_bid = float(bids[0][0])
                            bid_qty = float(bids[0][1])
                            best_ask = float(asks[0][0])
                            ask_qty = float(asks[0][1])

                            mid = (best_bid + best_ask) / 2
                            spread = best_ask - best_bid

                            top5_bid = sum(float(x[1]) for x in bids[:5])
                            top5_ask = sum(float(x[1]) for x in asks[:5])

                            imbalance = (top5_bid - top5_ask) / (
                                top5_bid + top5_ask + 1e-9
                            )

                            microprice = (
                                best_ask * bid_qty + best_bid * ask_qty
                            ) / (bid_qty + ask_qty + 1e-9)

                            trade = latest_trade.get(symbol)

                            writer.writerow({
                                "timestamp": datetime.now(UTC),
                                "symbol": symbol,
                                "mid": mid,
                                "spread": spread,
                                "best_bid": best_bid,
                                "best_ask": best_ask,
                                "bid_qty": bid_qty,
                                "ask_qty": ask_qty,
                                "top5_bid": top5_bid,
                                "top5_ask": top5_ask,
                                "imbalance": imbalance,
                                "microprice_dev": microprice - mid,
                                "trade_side": int(trade["m"]) if trade else -1,
                                "trade_qty": float(trade["q"]) if trade else 0,
                            })

                            count += 1

                            if count % 10_000 == 0:
                                print(f"Collected {count} rows")
                                f.flush()

                            if count >= MAX_ROWS:
                                break

            except Exception as e:
                print(f"Connection dropped: {e}")
                print("Reconnecting in 5 seconds...")
                f.flush()
                await asyncio.sleep(5)

    print(f"Done. Saved {count} rows to {OUT_FILE}")


asyncio.run(collect())